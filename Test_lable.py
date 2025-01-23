import os
from argparse import ArgumentParser
import glob
import itertools
import numpy as np
import torch
import torch.utils.data as Data
import nibabel as nib

from WATFunctions import generate_grid_unit, transform_unit_flow_to_flow, load_4D
from model_stage import (
    WavletMono_unit_add_lvl1, 
    WavletMono_unit_add_lvl2, 
    WavletMono_unit_add_lvl3, 
    WavletMono_unit_add_lvl4, 
    SpatialTransform_unit, 
    SpatialTransformNearest_unit
)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='./WD_OAS_NCC_lvl4_80000.pth',
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='./xxx',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--data", type=str,
                    dest="fixed", default='./fixed/',
                    help="fixed image")
opt = parser.parse_args()

savepath = opt.savepath
data_path = opt.fixed
savepath_seg = './output_image/seg/'
savepath_img = './output_image/img/'
label_path = './T1_mask/'

if not os.path.isdir(savepath_img):
    os.makedirs(savepath_img)

if not os.path.isdir(savepath_seg):
    os.makedirs(savepath_seg)

start_channel = opt.start_channel

def test():
    imgshape4 = (128, 128, 128)
    imgshape3 = (64, 64, 64)
    imgshape2 = (32, 32, 32)
    imgshape1 = (16, 16, 16)

    model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape1=imgshape1).to(device)
    model2 = WavletMono_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2, model1=model1).to(device)
    model3 = WavletMono_unit_add_lvl3(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2,
                                      imgshape3=imgshape3, model2=model2).to(device)

    model4 = WavletMono_unit_add_lvl4(2, 3, start_channel, is_train=False, imgshape1=imgshape1, imgshape2=imgshape2,
                                      imgshape3=imgshape3, imgshape4=imgshape4, model3=model3).to(device)

    transform_near = SpatialTransformNearest_unit().to(device)
    transform_tri = SpatialTransform_unit().to(device)


    model4.load_state_dict(torch.load(opt.modelpath))
    model4.eval()
    transform_tri.eval()

    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    names = sorted(glob.glob(data_path + 'test/*.nii.gz'))

    index_pair = list(itertools.permutations(names, 2))

    def save_nifti(data, filename, affine):
                try:
                    img = nib.Nifti1Image(data, affine)
                    nib.save(img, filename)
                    print(f"Saved {filename}")
                except Exception as e:
                    print(f"Error saving {filename}: {e}")
    
    for pair in index_pair:

        moving = pair[0]
        moving_img = load_4D(moving)
        moving_img = torch.from_numpy(moving_img).float().to(device).unsqueeze(dim=0)
        fixed = pair[1]
        fixed_img = load_4D(fixed)
        fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)

        filename = os.path.basename(moving)
        num1 = filename.split(".")[0]
        filename = os.path.basename(fixed)
        num2 = filename.split(".")[0]

        label1 = load_4D(label_path + num1 + '_mask.nii.gz')
        label1 = torch.from_numpy(label1).float().to(device).unsqueeze(dim=0)

        label2 = load_4D(label_path + num2 + '_mask.nii.gz')
        label2 = torch.from_numpy(label2).float().to(device).unsqueeze(dim=0)

        ########## warp segmentation maps ########
        label1 = load_4D(label_path + num1 + '_mask.nii.gz')
        label1 = torch.from_numpy(label1).float().to(device).unsqueeze(dim=0)

        label2 = load_4D(label_path + num2 + '_mask.nii.gz')
        label2 = torch.from_numpy(label2).float().to(device).unsqueeze(dim=0)

        LLL3_M = os.path.join(data_path + 'LLL3/test/' + num1 + '.nii.gz')
        img3_LLL_M = load_4D(LLL3_M)
        M3_LLL = torch.from_numpy(img3_LLL_M).float().to(device).unsqueeze(dim=0)
        LLL3_F = os.path.join(data_path + 'LLL3/test/' + num2 + '.nii.gz')
        img3_LLL_F = load_4D(LLL3_F)
        F3_LLL = torch.from_numpy(img3_LLL_F).float().to(device).unsqueeze(dim=0)

        LLL2_M = os.path.join(data_path + 'LLL2/test/' + num1 + '.nii.gz')
        img2_LLL_M = load_4D(LLL2_M)
        M2_LLL = torch.from_numpy(img2_LLL_M).float().to(device).unsqueeze(dim=0)
        LLL2_F = os.path.join(data_path + 'LLL2/test/' + num2 + '.nii.gz')
        img2_LLL_F = load_4D(LLL2_F)
        F2_LLL = torch.from_numpy(img2_LLL_F).float().to(device).unsqueeze(dim=0)

        EM2_M = os.path.join(data_path + 'EM2/test/' + num1 + '.nii.gz')
        img2_EM_M = load_4D(EM2_M)
        M2_EM = torch.from_numpy(img2_EM_M).float().to(device).unsqueeze(dim=0)
        EM2_F = os.path.join(data_path + 'EM2/test/' + num2 + '.nii.gz')
        img2_EM_F = load_4D(EM2_F)
        F2_EM = torch.from_numpy(img2_EM_F).float().to(device).unsqueeze(dim=0)

        LLL1_M = os.path.join(data_path + 'LLL1/test/' + num1 + '.nii.gz')
        img1_LLL_M = load_4D(LLL1_M)
        M1_LLL = torch.from_numpy(img1_LLL_M).float().to(device).unsqueeze(dim=0)
        LLL1_F = os.path.join(data_path + 'LLL1/test/' + num2 + '.nii.gz')
        img1_LLL_F = load_4D(LLL1_F)
        F1_LLL = torch.from_numpy(img1_LLL_F).float().to(device).unsqueeze(dim=0)

        EM1_M = os.path.join(data_path + 'EM1/test/' + num1 + '.nii.gz')
        img1_EM_M = load_4D(EM1_M)
        M1_EM = torch.from_numpy(img1_EM_M).float().to(device).unsqueeze(dim=0)
        EM1_F = os.path.join(data_path + 'EM1/test/' + num2 + '.nii.gz')
        img1_EM_F = load_4D(EM1_F)
        F1_EM = torch.from_numpy(img1_EM_F).float().to(device).unsqueeze(dim=0)
        with torch.no_grad():
            field4, field4_in = model4(M3_LLL, F3_LLL, M2_LLL, F2_LLL, M2_EM, F2_EM, M1_LLL, F1_LLL, M1_EM, F1_EM, moving_img, fixed_img, label1, label2)
                        
            X_Y = transform_tri(moving_img, field4.permute(0, 2, 3, 4, 1), grid)
            warp_label = transform_near(label1, field4.permute(0, 2, 3, 4, 1), grid)

            X_Y = transform_tri(moving_img, field4.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]
            warp_label = transform_near(label1, field4.permute(0, 2, 3, 4, 1), grid).data.cpu().numpy()[0, 0, :, :, :]

            F_X_Y_cpu = field4.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            F_X_Y_cpu = transform_unit_flow_to_flow(F_X_Y_cpu)

            F_in_cpu = field4_in.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            F_in_cpu = transform_unit_flow_to_flow(F_in_cpu)

            save_nifti(F_X_Y_cpu, savepath_img + 'field-' + num1 + '--' + num2 + '.nii.gz', np.eye(4)) 
            save_nifti(F_in_cpu, savepath_img + 'infield-' + num1 + '--' + num2 + '.nii.gz', np.eye(4)) 
            save_nifti(X_Y, savepath_img + 'warped-' + num1 + '--' + num2 + '.nii.gz', np.eye(4)) 
            save_nifti(warp_label, savepath_seg + 'warped_label-' + num1 + '--' + num2 + '.nii.gz', np.eye(4)) 

if __name__ == '__main__':
    imgshape = (128, 128, 128)
    range_flow = 1
    test()
