# import models
# from models.TransBTS.Unet_skipconnection import Unet
from models.TransBTS.Unet_v2 import Cascaded_Unet, Unet1
import torch
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random
import argparse

from data.BraTS import BraTS

def main():
    gpu_id = 0
    valid_list = "/workspace/multimodal/datasets/BraTs2019/validation/valid.txt"
    valid_root = "/workspace/multimodal/datasets/BraTs2019/validation"
    rslt_save_path = "./experiments/cascad_withWarmUP_diffLoss2023-02-18/rslt"
    model_checkpoint = "./experiments/cascad_withWarmUP_diffLoss2023-02-18/snapshot/245_loss 0.318_et0.860_tc0.906_wt0.851.pth"
    test_log = os.path.join(rslt_save_path, 'test_log.txt')
    
    if not os.path.exists(rslt_save_path):
        os.mkdir(rslt_save_path)
    
    pf_log = open(test_log, "w+")

    infe_thresh = 0.5
    valid_data = BraTS(valid_list, valid_root, mode='test')
    valid_loader = DataLoader(valid_data, batch_size=1, pin_memory=True)

    # model = Unet()
    model = Cascaded_Unet()
    model.load_state_dict(torch.load(model_checkpoint)['MODEL_STATE'])
    model.to(gpu_id)
    model.eval()

    for data, img_name, affine in valid_loader:
        data = data.to(gpu_id)
        img_name = img_name[0]
        with torch.no_grad():
            _, pred = model(data)
            pred = pred.squeeze()[0:3, :, :, :-5].detach().cpu().numpy()

        pred = (pred > infe_thresh)
        et = pred[0] 
        tc = pred[1]
        wt = pred[2]
        
        seg = np.zeros_like(pred[0], dtype=np.int16)
        if et.sum() < 100:
            net = tc
        else:
            net = np.logical_and(tc, np.logical_not(et))
            seg[et] = 4
        ed  = np.logical_and(wt, np.logical_not(tc))
        seg[net] = 1
        seg[ed]  = 2

        pf_log.write(f'[{img_name}]   et voxel num: {et.sum()}\n')
        # seg_name = os.path.join(rslt_save_path, img_name + ".nii.gz")
        # nib.save(nib.Nifti1Image(seg, affine=affine[0].numpy()), seg_name)
        # print(img_name, " processed.")

    pf_log.close()



if __name__ == "__main__":
    main()