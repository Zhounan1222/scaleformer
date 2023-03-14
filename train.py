# torchrun --standalone --nproc_per_node=2 train.py
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributied.launch --nproc_per_node=1 train.py

import argparse
import random
import logging
import numpy as np
import time
import setproctitle

from torch.cuda.amp import autocast

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from models.TransBTS.TransBTS_downsample8x_skipconnection import TransBTS
# from models.TransBTS.Unet_skipconnection import Unet
from models.TransBTS.Unet_v2 import Unet1, Unet2
# from models.TransBTS.resunet import ResUNet
from models.criterions import *

from models.BraTS import BraTS
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from ranger import Ranger

import torch.multiprocessing as mp  # multi-process wrapper
from torch.utils.data.distributed import DistributedSampler  # distribute data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group  # init and destroy process group
import os
from networks.ScaleFormer import ScaleFormer
# It is usually one process for a GPU
# worldsize: the total number of processes in a group
# rank: a unique identifier that is assigned to each process, global, ranges from 0 to wordsize-1
# MASTER_ADDR: the ip addr for a machine that running the rank 0 process.

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
parser = argparse.ArgumentParser()
# Basic Information
parser.add_argument('--user', default='name of user', type=str)
parser.add_argument('--experiment', default='scaleformer_c16_b4a6', type=str)
parser.add_argument('--date',
                    # default='2023-02-13',
                    default=local_time.split(' ')[0],
                    type=str)
parser.add_argument('--description',
                    default='TransBTS,'
                            'training on train.txt!',
                    type=str)
# DataSet Information
parser.add_argument('--local_rank',type=int)
parser.add_argument('--root', default='/workspace/multimodal/datasets/BraTs2019', type=str)
parser.add_argument('--train_dir', default='training', type=str)
parser.add_argument('--valid_dir', default='validation', type=str)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--train_file', default='train_split.txt', type=str)
parser.add_argument('--test_file', default='test_split.txt', type=str)
parser.add_argument('--valid_file', default='valid.txt', type=str)
parser.add_argument('--dataset', default='brats', type=str)
parser.add_argument('--model_name', default='TransBTS', type=str)
parser.add_argument('--input_C', default=4, type=int)
parser.add_argument('--input_H', default=240, type=int)
parser.add_argument('--input_W', default=240, type=int)
parser.add_argument('--input_D', default=160, type=int)
parser.add_argument('--crop_H', default=128, type=int)
parser.add_argument('--crop_W', default=128, type=int)
parser.add_argument('--crop_D', default=128, type=int)
parser.add_argument('--output_D', default=155, type=int)

# Training Information
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--amsgrad', default=True, type=bool)
parser.add_argument('--num_class', default=3, type=int)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--grad_accum_freq', default=6, type=int)
parser.add_argument('--max_epochs', default=300, type=int)
parser.add_argument('--save_freq', default=100, type=int)
parser.add_argument('--snapshot_file', default='', type=str)
parser.add_argument('--debug', action="store_true",default=True)

args = parser.parse_args()


class Trainer:
    def __init__(self,
                 gpu_id: int,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 train_dataset: DataLoader,
                 test_dataset: DataLoader,
                 snapshot_path: str,
                 snapshot_file: str,
                 max_epochs: int,
                 init_lr: float,
                 snapshot_save_freq,
                 writer,
                 test_freq=5,
                 fp_val=None,
                 ):

        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id) 
        self.train_data = train_dataset
        self.test_data = test_dataset
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.snapshot_path = snapshot_path
        self.snapshot_save_freq = snapshot_save_freq
        self.writer = writer
        self.test_freq = test_freq
        self.use_amp = True
        self.sigmoid = nn.Sigmoid()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        # loss func
        self.dice_loss = Criterion()
        self.bce = nn.BCEWithLogitsLoss()
        
        self.max_epochs = max_epochs
        self.epochs_run = 0
        self.iters_run = 0
        self.best_loss = 1e8
        self.val_txt = fp_val

        # snapshot_file = os.path.join(self.snapshot_path, snapshot_file)
        if os.path.exists(snapshot_file) and os.path.isfile(snapshot_file):
            logging.info('loading checkpoint {}'.format(snapshot_file))
            self._load_snapshot(snapshot_file)
            logging.info('Successfully loading checkpoint {} and training from epoch: {}, learning rate: {:.6f}'
                         .format(snapshot_file, self.epochs_run, self.optimizer.param_groups[0]['lr']))
        else:
            logging.info('Training from scratch!!!')
        
        if not args.debug: 
            self.model = DDP(module=model, device_ids=[self.gpu_id])
    
    def _criterion(self, pred, target1, target2, mode='train'):
        b = pred.size(0)
        # et, others = pred[:, 0:1, ...], pred[:, 1:, ...]
        # if mode == 'train':
        #     new_et = et * self.sigmoid(et_pred.view(b, 1, 1, 1, 1))
        # else:
        #     new_et = et * (et_pred.squeeze() > 0.5)
        # pred = torch.cat([new_et, others], dim=1)
        loss = self.dice_loss.sigmoid_dice(pred, target2)

        # exist_loss = self.bce(et_pred.flatten(), et_label)

        if mode != 'train':
            et, tc, wt = self.dice_loss.dice_metrix(pred, target2)
            
            msk = (pred > 0.5)
            et_msk = msk[:, 0:1, ...] 
            net_msk = torch.logical_and(torch.logical_not(et_msk), msk[:, 1:2, ...])
            ed_msk = torch.logical_and(torch.logical_not(msk[:, 1:2, ...]), msk[:, 2:3, ...])
            pred_split = torch.cat([net_msk, ed_msk, et_msk], dim=1)
            net, ed, _ = self.dice_loss.dice_metrix(pred_split, target1)
            # msk = (pred2 > 0.5)
            # et_msk = msk[:, 0:1, ...]
            # net_msk = torch.logical_and(torch.logical_not(et_msk), msk[:, 1:2, ...])
            # ed_msk = torch.logical_and(torch.logical_not(net_msk), msk[:, 2:3, ...])
            # pred2_split = torch.cat([net_msk, ed_msk, et_msk], dim=1)

            # et, tc, wt = self.dice_loss.dice_metrix(pred2, target2)    
            # net, ed, _ = self.dice_loss.dice_metrix(pred2_split, target1)
            return loss,  et, tc, wt, net, ed
        else:
            return loss
        
    def _load_snapshot(self, snapshot_file):
        snapshot = torch.load(snapshot_file, map_location='cuda:{}'.format(self.gpu_id))
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.optimizer.load_state_dict(snapshot['OPTIMIZER_STATE'])
        self.epochs_run = snapshot['EPOCHS_RUN']
        self.iters_run = self.epochs_run * len(self.train_data)
        if 'SCALER_STATE' in snapshot.keys():
            self.scaler.load_state_dict(snapshot['SCALER_STATE'])
            self.use_amp = True
        else:
            self.use_amp = False
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch, info_append=None):
        snapshot = {
            'MODEL_STATE': self.model.module.state_dict(),
            'OPTIMIZER_STATE': self.optimizer.state_dict(),
            'SCALER_STATE': self.scaler.state_dict(),
            'EPOCHS_RUN': epoch,
        }

        if info_append is None:
            snapshot_file = os.path.join(self.snapshot_path, str(epoch) + ".pth")
        else:
            snapshot_file = os.path.join(self.snapshot_path, str(epoch) + '_' + info_append + ".pth")
        torch.save(snapshot, snapshot_file)
        print(f"Traning snapshot saved at {snapshot_file}")              

    def _adjust_learning_rate(self, epoch, power=0.9):
        if epoch < 5:
            lr = (self.init_lr / 5) * (epoch + 1)
        else:
            lr = round(self.init_lr * np.power(1- ((epoch - 5) / (self.max_epochs - 5)), power), 8)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
   
    def _run_batch(self, data, target1, target2):
        with torch.autocast(device_type='cuda', enabled=self.use_amp):
            pred = self.model(data)
            loss = self._criterion(pred,target1, target2)
        return loss
    
    def _run_epoch(self, epoch):
        b_sz = len(self.train_data)
        
        if not args.debug:
            self.train_data.sampler.set_epoch(epoch)
        self._adjust_learning_rate(epoch)
        
        for b_idx, batch in enumerate(self.train_data):
            self.iters_run += 1 

            data = batch['image'].to(self.gpu_id, non_blocking=True)
            target1 = batch['label1'].to(self.gpu_id, non_blocking=True)
            target2 = batch['label2'].to(self.gpu_id, non_blocking=True)
            # et_label = batch['et_present'].float().to(self.gpu_id, non_blocking=True)
            
            # if b_idx == 1:
            #     break
            dice_loss = self._run_batch(data, target1, target2)
            loss = dice_loss
            self.scaler.scale(loss).backward()
            if (self.iters_run + 1) % args.grad_accum_freq == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if self.gpu_id == 0 and self.iters_run % args.grad_accum_freq == 0:
                logging.info('Epoch: {}  Iter:{}/{}  loss: {:.5f}'
                             .format(epoch+1, b_idx+1, b_sz, loss))

                self.writer.add_scalar('lr:', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('loss/loss', loss, self.iters_run)
                self.writer.add_scalar('loss/dice_loss', dice_loss, self.iters_run)
                self.writer.add_scalar('loss/exist_loss',  self.iters_run)
    
    def train(self):
        try:
            for epoch in range(self.epochs_run, self.max_epochs):
                
                start_epoch = time.time()
                self._run_epoch(epoch)
                end_epoch = time.time()
                
                if self.gpu_id == 0:
                    epoch_time_minute = (end_epoch-start_epoch)/60
                    remaining_time_hour = (self.max_epochs-epoch-1)*epoch_time_minute/60
                    logging.info('Current epoch time consumption: {:.2f} minutes!'.format(epoch_time_minute))
                    logging.info('Estimated remaining training time: {:.2f} hours!'.format(remaining_time_hour))

                    if (epoch + 1) % self.snapshot_save_freq == 0:
                        self._save_snapshot(epoch + 1)
                    epoch = 4
                    if (epoch + 1) % self.test_freq == 0:
                        self.model.eval()
                        self._test(epoch)
                        self.model.train()
                        
            if self.gpu_id == 0:
                self.val_txt.close()

        except Exception as e:
            print(e)
            self._save_snapshot(epoch, info_append="EXCEPTION") 
            if self.gpu_id == 0:
                self.val_txt.close()
        
    def _test(self, epoch):
        dice_loss_sum,et_sum, tc_sum, wt_sum, net_sum, ed_sum = 0., 0., 0., 0., 0,0.
        self.val_txt.write('\n\n\n<<< Test Epoch >>>: {}\n'.format(epoch))

        for batch in self.test_data:
            data = batch['image'].to(self.gpu_id)
            target1 = batch['label1'].to(self.gpu_id)
            target2 = batch['label2'].to(self.gpu_id)
            patient_idx = batch['patient_idx'][0]
            # et_label = batch['et_present'].float().to(self.gpu_id, non_blocking=True)
            
            with torch.no_grad():
                pred= self.model(data)
                # loss, et, tc, wt, net, ed = self._criterion(pred, target1, target2, mode='val') 
                dice_loss,et, tc, wt, net, ed = self._criterion(pred, target1, target2, mode='val')
                self.val_txt.write(f"[{patient_idx}]   ET: {et:.3f}  TC: {tc:.3f}  WT: {wt:.3f}  NET:{net:.3f}  ED: {ed:.3f}\n")
                          
            dice_loss_sum += dice_loss
            et_sum += et
            tc_sum += tc
            wt_sum += wt
            net_sum += net
            ed_sum += ed

        dataset_len = len(self.test_data)
  
        dice_loss_avg = dice_loss_sum /dataset_len
        # exit_et_loss_avg = exist_et_loss_sum / dataset_len
        
        total_loss_avg = dice_loss_avg
        et_avg = et_sum / dataset_len
        tc_avg = tc_sum / dataset_len
        wt_avg = wt_sum / dataset_len
        net_avg = net_sum / dataset_len
        ed_avg = ed_sum / dataset_len
        
        if total_loss_avg < self.best_loss:
            self._save_snapshot(epoch + 1, info_append="loss {:.3f}_diceloss{:.3f}_et{:.3f}_tc{:.3f}_wt{:.3f}_net{:.3f}_ed{:.3f}".format(total_loss_avg, dice_loss_avg,et_avg, tc_avg, wt_avg, net_avg, ed_avg))
            self.best_loss = total_loss_avg
        logging.info('<<< Test Epoch >>>: {}  loss: {:.5f} || dice_loss:{:.4f} |  ET:{:.4f} | TC:{:.4f} | WT:{:.4f} | NET:{:.4f} | ED:{:.4f}||\n'.format(epoch+1, total_loss_avg,dice_loss_avg, et_avg, tc_avg, wt_avg, net_avg, ed_avg))
        
        self.writer.add_scalar('test/dice_loss_total', total_loss_avg, epoch) 
        self.writer.add_scalar('test/dice_loss', dice_loss_avg, epoch)
        # self.writer.add_scalar('test/exit)_et_loss', exit_et_loss_avg, epoch)
        self.writer.add_scalar('test/dice_ET', et_avg, epoch)
        self.writer.add_scalar('test/dice_TC', tc_avg, epoch)
        self.writer.add_scalar('test/dice_WT', wt_avg, epoch)
        self.writer.add_scalar('test/dice_NET', net_avg, epoch)
        self.writer.add_scalar('test/dice_ED', ed_avg, epoch)


def main():
    
    if args.debug:
        local_rank = 0
    else:
        init_process_group(backend='nccl')
        # local_rank = int(args.local_rank)
        local_rank = int(os.environ['LOCAL_RANK']) 
        print(f"local_rank:{local_rank} is working ...")
       
    

    if args.debug:
        args.experiment = 'debug'

    exp_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'experiments', args.experiment + args.date)
    snapshot_path = os.path.join(exp_dir, 'snapshot')
    if local_rank==0 and not os.path.exists(exp_dir):
        os.makedirs(snapshot_path)

    tb_dir = os.path.join(exp_dir, 'runs')
    if local_rank==0 and not os.path.exists(tb_dir):
        os.mkdir(tb_dir)

    writer = SummaryWriter(tb_dir)
    if local_rank == 0:
        log_file = os.path.join(exp_dir, 'log.txt')
        log_args(log_file)
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description)) 

    val_file = os.path.join(exp_dir, 'val.txt')
    if local_rank == 0:
        fp_val = open(val_file, mode='w+')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(local_rank)
    # model = Unet1(base_channels=16)
    model = ScaleFormer(n_classes=3)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, betas=[0.9, 0.999], amsgrad=args.amsgrad)
    optimizer = Ranger(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # criterion = getattr(criterions, args.criterion)
    train_list = os.path.join(args.root, args.train_dir, args.train_file)
    train_root = os.path.join(args.root, args.train_dir)
    train_set = BraTS(train_list, train_root, 'train')

    test_list = os.path.join(args.root, args.train_dir, args.test_file)
    test_set = BraTS(test_list, train_root, 'valid')
    
    if args.debug: 
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, drop_last=True, num_workers=0, pin_memory=True, shuffle=True)
    else:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_loader = DataLoader(dataset=train_set, sampler=train_sampler, batch_size=args.batch_size,
                                drop_last=True, num_workers=args.num_workers, pin_memory=True, shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=1, pin_memory=True)
    logging.info('Samples for train = {}'.format(len(train_set)))
    logging.info('Samples for test = {}'.format(len(test_set)))


    trainer = Trainer(local_rank, model, optimizer, train_loader, test_loader, snapshot_path, args.snapshot_file, args.max_epochs, args.lr, args.save_freq, writer, fp_val=fp_val if local_rank == 0 else None)
    
    torch.set_grad_enabled(True)
    start_time = time.time()

    trainer.train()
    end_time = time.time()

    total_time = (end_time-start_time)/3600
    logging.info('The total training time is {:.2f} hours'.format(total_time))
    logging.info('----------------------------------The training process finished!-----------------------------------')
    
    if not args.debug:
        destroy_process_group()


def log_args(log_file):
    
    from pathlib import Path
    file = Path(log_file).touch(exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


if __name__ == '__main__':
    main()  # rank will be assigned automatically by mp.spawn
