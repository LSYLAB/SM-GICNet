import glob
import os
import torch.nn as nn
from argparse import ArgumentParser
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import torch
import torch.utils.data as Data
import nibabel as nib
import SimpleITK as sitk
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# from asdnet.utils import OneHotEncode
plt.switch_backend('agg')
from WATFunctions import generate_grid, Dataset_epoch_train, transform_unit_flow_to_flow_cuda, generate_grid_unit
from model_stage import WavletMono_unit_add_lvl1, WavletMono_unit_add_lvl2, WavletMono_unit_add_lvl3, WavletMono_unit_add_lvl4, SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, antifoldloss

'''
    Takes a Tensor of size 1xHxW and create one-hot encoding of size nclassxHxW
    OneHotEncoding: [2,1,3,0]->[0 0 1 0; 0 1 0 0; 0 0 0 1; 1 0 0 0]
    label: NxWxHxD or NxWxH
'''

def get_middle_slice(image):
    """Extract the middle slice along each axis."""
    slices = []
    for axis in range(3):
        mid_index = image.shape[axis] // 2
        slices.append(torch.index_select(image, axis, torch.tensor(mid_index)).squeeze())
    return slices

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


saveImgPath = './'

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration_lvl1", type=int,
                    dest="iteration_lvl1", default=30001,
                    help="number of lvl1 iterations")
parser.add_argument("--iteration_lvl2", type=int,
                    dest="iteration_lvl2", default=40001,
                    help="number of lvl2 iterations")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=60001,
                    help="number of lvl3 iterations")
parser.add_argument("--iteration_lvl4", type=int,
                    dest="iteration_lvl4", default=80001,
                    help="number of lvl4 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=1,
                    help="Anti-fold loss: suggested range 0 to 1000")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=1,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=1000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='./LLL3/train/',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=3000,
                    help="Number step for freezing the previous level")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath
freeze_step = opt.freeze_step

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3
iteration_lvl4 = opt.iteration_lvl4


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

transform_near = SpatialTransformNearest_unit().to(device)



def train_lvl1():
    model_name = "WD_OAS_NCC_lvl1_"
    print("Training lvl1...")

    model = WavletMono_unit_add_lvl1(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1).to(device)

    loss_similarity1 = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_antifold = antifoldloss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    # OASIS
    names = sorted(glob.glob(datapath + '*.nii.gz'))

    grid = generate_grid(imgshape1)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './xxx'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl1+1))
    loss_total = []
    loss_smo = []
    loss_sim = []
    loss_Jaco = []
    loss_to = 0
    loss_sm=0
    loss_si = 0
    loss_Ja = 0

    training_generator = Data.DataLoader(Dataset_epoch_train(names, norm=False), batch_size=1,
                                         shuffle=True, num_workers=4)
    

    load_model = False
    if load_model is True:
        model_path = "/xxx/WD_OAS_NCC_reg_5.pth"
        print("Loading weight: ", model_path)
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("/XXX/WD_OAS_NCC_reg_5.npy")
        lossall[:, 0:1000] = temp_lossall[:, 0:1000]
    # epoch = 0
    step = 1
    while step <= iteration_lvl1:
        for X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2, label1, label2 in training_generator:

            X1_LLL = X1_LLL.to(device).float()
            Y1_LLL = Y1_LLL.to(device).float()

# 正向
            field1, field1_in, warped_x1, fixed1, mov1, warped_x1_in = model(X1_LLL, Y1_LLL)
            
# 逆向
            field1_re, field1_in_re, warped_x1_re, fixed1_re, mov1_re, warped_x1_in_re = model(Y1_LLL, X1_LLL)

# 正逆向loss
            field2_in_re_clone = field1_in_re.clone()
            field2_re_clone = field1_re.clone()

            field2_in_clone = field1_in.clone()
            field2_clone = field1.clone()

            warp_filed = transform_near(field2_clone, field1.permute(0, 2, 3, 4, 1), grid)
            warp_filed_shape = warp_filed.shape
            warp_ones = torch.ones(warp_filed_shape[0], warp_filed_shape[1], warp_filed_shape[2],warp_filed_shape[3],warp_filed_shape[4]).to(device)
            criterion_f = nn.MSELoss()
            loss_filed = criterion_f(warp_filed, warp_ones)


#--------------------------一致性--------------------------------------------------
            loss_inv = criterion_f(warped_x1_in, warped_x1_re)

            loss_inv_re = criterion_f(warped_x1_in_re, warped_x1)


            # 1 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC2 = loss_similarity1(warped_x1_re, fixed1_re)
            loss_multiNCC3 = loss_similarity1(warped_x1_in, fixed1_re)
            loss_multiNCC4 = loss_similarity1(warped_x1_in_re, fixed1)
            loss_multiNCC  = loss_multiNCC1 + loss_multiNCC2 + 0.1 * loss_multiNCC3 + 0.1*loss_multiNCC4


            # reg2 - use velocity
            _, _, x, y, z = field1.shape
            field1[:, 0, :, :, :] = field1[:, 0, :, :, :] * z
            field1[:, 1, :, :, :] = field1[:, 1, :, :, :] * y
            field1[:, 2, :, :, :] = field1[:, 2, :, :, :] * x
            loss_regulation1 = loss_smooth(field1)
            loss_regulation2 = loss_smooth(field1_in)
            loss_regulation3 = loss_smooth(field1_re)
            loss_regulation4 = loss_smooth(field1_in_re)
            loss_regulation = loss_regulation1 + 0.1* loss_regulation2 + loss_regulation3 + 0.1* loss_regulation4

            loss_fold = loss_antifold(field1) + loss_antifold(field1_re)

            loss = loss_multiNCC + smooth * loss_regulation + antifold*loss_fold + 0.01* loss_filed + 0.001*(loss_inv + loss_inv_re)  #+ 0*loss_Jacobian
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(),  loss_regulation.item(), loss_fold.item()])
            loss_to = loss_to + loss.item()
            loss_si = loss_si + loss_multiNCC.item()
            loss_sm = loss_sm + loss_regulation.item()
            loss_Ja = loss_Ja + loss_fold.item()
            loss_total.append(loss.item())
            loss_sim.append(loss_multiNCC.item())
            loss_smo.append(loss_regulation.item())
            loss_Jaco.append(loss_fold.item())

            if (step % 1000 == 0):
                total = format(float(loss_to)/float(1000), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si)/float(1000), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(1000), '.6f')
                #loss_smo.append(smo)
                Jaco =  format((float(loss_Ja)/float(1000)), '.9f')
                #loss_Jaco.append(Jaco)
                Fil = format(float(loss_filed) / float(1000), '.9f')
                L_i = format(float(loss_inv*1) / float(1000), '.9f')
                L_i_r = format(float(loss_inv_re*1) / float(1000), '.9f')

                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco),\
                      'Filed:'+str(Fil), 'inv:'+str(L_i), 'invr:'+str(L_i_r))
                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0
            if (step % 5000 == 0):
                modelname = model_dir + model_name + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + model_name + str(step) + '.npy', lossall)


                x1 = range(0, len(loss_total))
                x2 = range(0, len(loss_sim))
                x3 = range(0, len(loss_smo))
                x4 = range(0, len(loss_Jaco))
                y1 = loss_total
                y2 = loss_sim
                y3 = loss_smo
                y4 = loss_Jaco
                plt.switch_backend('agg')
                plt.subplot(2, 2, 1)
                plt.plot(x1, y1, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_total')
                plt.subplot(2, 2, 2)
                plt.plot(x2, y2, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_sim')
                plt.subplot(2, 2, 3)
                plt.plot(x3, y3, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_smo')
                plt.subplot(2, 2, 4)
                plt.plot(x4, y4, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_Jac')
                plt.tight_layout()
                # plt.show()
                plt.savefig(model_dir + str(step) + "_lv1.jpg")

            step += 1

            if step > iteration_lvl1:
                break
        print("-------------------------- level 1 epoch pass-------------------------")
    print("level 1 Finish!")

def train_lvl2():
    model_name = "WD_OAS_NCC_lvl2_"
    print("Training lvl2...")
    model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1).to(device)
    model1_path = "./xxx/WD_OAS_NCC_lvl1_30000.pth"
    model1.load_state_dict(torch.load(model1_path))
    print("Loading weight for model_lvl1...", model1_path)

    # Freeze model_lvl1 weight
    for param in model1.parameters():
        param.requires_grad = False

    model2 = WavletMono_unit_add_lvl2(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2, model1=model1).to(device)

    loss_similarity1 = NCC(win=3)
    loss_similarity2 = NCC(win=5)

    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_antifold = antifoldloss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    # OASIS
    names = sorted(glob.glob(datapath + '*.nii.gz'))

    grid = generate_grid(imgshape2)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model2.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './xxx'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl2+1))
    loss_total = []
    loss_smo = []
    loss_sim = []
    loss_Jaco = []
    loss_to = 0
    loss_sm = 0
    loss_si = 0
    loss_Ja = 0

    training_generator = Data.DataLoader(Dataset_epoch_train(names, norm=False), batch_size=1,
                                         shuffle=True, num_workers=4)

    
    load_model = False
    if load_model is True:
        model_path = "/XXX/WD_OAS_NCC_reg_5.pth"
        print("Loading weight: ", model_path)
        model2.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("/XXX/WD_OAS_NCC_reg_5.npy")
        lossall[:, 0:1000] = temp_lossall[:, 0:1000]

    step = 1
    while step <= iteration_lvl2:
        for X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2, label1, label2 in training_generator:

            X1_LLL = X1_LLL.to(device).float()
            Y1_LLL = Y1_LLL.to(device).float()
            X2_LLL = X2_LLL.to(device).float()
            Y2_LLL = Y2_LLL.to(device).float()
            X2_HHH = X2_HHH.to(device).float()
            Y2_HHH = Y2_HHH.to(device).float()

            
# 正向
            field1, field2, field2_in, warped_x1, warped_x2_lll, \
                warped_x2_hhh, fixed1, fixed2_lll, fixed2_hhh, \
                    mov1, mov2, warped_x2_input, warped_x2_in\
               = model2(X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH)
            
# 逆向
            field1_re, field2_re, field2_in_re, warped_x1_re, warped_x2_lll_re, \
                   warped_x2_hhh_re, fixed1_re, fixed2_lll_re, fixed2_hhh_re, \
                   mov1_re, mov2_re, warped_x2_input_re, warped_x2_in_re\
               = model2(Y1_LLL, X1_LLL, Y2_LLL, X2_LLL, Y2_HHH, X2_HHH)

# 正逆向loss
            field2_in_re_clone = field2_in_re.clone()
            field2_re_clone = field2_re.clone()

            field2_in_clone = field2_in.clone()
            field2_clone = field2.clone()

            warp_filed = transform_near(field2_clone, field2.permute(0, 2, 3, 4, 1), grid)
            warp_filed_shape = warp_filed.shape
            warp_ones = torch.ones(warp_filed_shape[0], warp_filed_shape[1], warp_filed_shape[2],warp_filed_shape[3],warp_filed_shape[4]).to(device)
            criterion_f = nn.MSELoss()
            loss_filed = criterion_f(warp_filed, warp_ones)


#--------------------------一致性--------------------------------------------------
            warped_mean = (warped_x2_input + warped_x2_lll) / 2
            loss_mean = criterion_f(warped_x2_lll, warped_mean)

            warped_mean_re = (warped_x2_input_re + warped_x2_lll_re) / 2
            loss_mean_re = criterion_f(warped_x2_lll_re, warped_mean_re)

            loss_inv = criterion_f(warped_x2_in, warped_x2_lll_re)

            loss_inv_re = criterion_f(warped_x2_in_re, warped_x2_lll)



            # 3 level deep supervision NCC
            loss_multiNCC1_re = loss_similarity1(warped_x1_re, fixed1_re)
            loss_multiNCC2_re = loss_similarity2(warped_x2_lll_re, fixed2_lll_re)
            loss_multiNCC22_re = loss_similarity2(warped_x2_hhh_re, fixed2_hhh_re)

            loss_multiNCC_re  = 0.5*loss_multiNCC1_re + (loss_multiNCC2_re + loss_multiNCC22_re)

            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC2 = loss_similarity2(warped_x2_lll, fixed2_lll)
            loss_multiNCC22 = loss_similarity2(warped_x2_hhh, fixed2_hhh)

            loss_multiNCC  = 0.5*loss_multiNCC1 + (loss_multiNCC2 + loss_multiNCC22)

            loss_multiNCC_in = loss_similarity1(warped_x2_in, fixed2_lll_re)

            loss_multiNCC1_in_re = loss_similarity1(warped_x2_in_re, fixed2_lll)

            loss_ncc = loss_multiNCC + loss_multiNCC_re + 0.1* loss_multiNCC_in + 0.1* loss_multiNCC1_in_re


            # field_norm = transform_unit_flow_to_flow_cuda(field2.permute(0,2,3,4,1).clone())
            # loss_Jacobian = loss_Jdet(field_norm, grid)

            _, _, x1, y1, z1 = field1.shape
            field1[:, 0, :, :, :] = field1[:, 0, :, :, :] * z1
            field1[:, 1, :, :, :] = field1[:, 1, :, :, :] * y1
            field1[:, 2, :, :, :] = field1[:, 2, :, :, :] * x1
            _, _, x2, y2, z2 = field2.shape
            field2[:, 0, :, :, :] = field2[:, 0, :, :, :] * z2
            field2[:, 1, :, :, :] = field2[:, 1, :, :, :] * y2
            field2[:, 2, :, :, :] = field2[:, 2, :, :, :] * x2

            _, _, x2, y2, z2 = field1_re.shape
            field2_re[:, 0, :, :, :] = field2_re[:, 0, :, :, :] * z2
            field2_re[:, 1, :, :, :] = field2_re[:, 1, :, :, :] * y2
            field2_re[:, 2, :, :, :] = field2_re[:, 2, :, :, :] * x2

            _, _, x2, y2, z2 = field2_in.shape
            field2_in[:, 0, :, :, :] = field2_in[:, 0, :, :, :] * z2
            field2_in[:, 1, :, :, :] = field2_in[:, 1, :, :, :] * y2
            field2_in[:, 2, :, :, :] = field2_in[:, 2, :, :, :] * x2

            _, _, x2, y2, z2 = field2_re.shape
            field2_re[:, 0, :, :, :] = field2_re[:, 0, :, :, :] * z2
            field2_re[:, 1, :, :, :] = field2_re[:, 1, :, :, :] * y2
            field2_re[:, 2, :, :, :] = field2_re[:, 2, :, :, :] * x2

            loss_regulation1 = loss_smooth(field1)
            loss_regulation2 = loss_smooth(field2)
            loss_regulation3 = loss_smooth(field2_re)
            loss_regulation4 = loss_smooth(field2_in)
            loss_regulation5 = loss_smooth(field1_re)

            loss_regulation =  0.5*loss_regulation1 + loss_regulation2 + loss_regulation3 + 0.1* loss_regulation4 + 0.5*loss_regulation5

            loss_fold1 = loss_antifold(field1)
            loss_fold2 = loss_antifold(field2)
            loss_fold3 = loss_antifold(field2_re)
            loss_fold = 0.5*loss_fold1+loss_fold2+loss_fold3

            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss = loss_ncc + loss_regulation + loss_fold + 0.01 * loss_filed + 0.001 *(1* loss_inv + 1*loss_inv_re + 0.5*loss_mean + 0.5*loss_mean_re)#loss_Jacobian
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


            lossall[:, step] = np.array(
                [loss.item(), loss_ncc.item(),  loss_regulation.item(), loss_fold.item()])

            loss_total.append(loss.item())
            loss_sim.append(loss_ncc.item())
            loss_smo.append(loss_regulation.item())
            loss_Jaco.append(loss_fold.item())

            loss_to = loss_to + loss.item()
            loss_si = loss_si + loss_ncc.item()
            loss_sm = loss_sm + loss_regulation.item()
            loss_Ja = loss_Ja + loss_fold.item()

            if (step % 500 == 0):

                total = format(float(loss_to) / float(1000), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si) / float(1000), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(1000), '.6f')
                #loss_smo.append(smo)
                Jaco = format(float(loss_Ja) / float(1000), '.9f')
                #loss_Jaco.append(Jaco)
                Fil = format(float(loss_filed) / float(1000), '.9f')
                L_m = format(float(loss_mean*0.5) / float(1000), '.9f')
                L_m_r = format(float(loss_mean_re*0.5) / float(1000), '.9f')
                L_i = format(float(loss_inv*1) / float(1000), '.9f')
                L_i_r = format(float(loss_inv_re*1) / float(1000), '.9f')

                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco),\
                      'Filed:'+str(Fil), 'mean:'+str(L_m), 'meanre:'+str(L_m_r), 'inv:'+str(L_i), 'invr:'+str(L_i_r))


            # 添加图像到 TensorBoard
                x_tb = list()
                y_tb = list()
                z_tb = list()

                for tag, image in [
                        ("image_lv2/lv1", warped_x2_input[:1].squeeze()),
                        ("image_lv2/lv2", warped_x2_lll[:1].squeeze()),
                        ("image_lv2/fake", warped_x2_in_re[:1].squeeze()),
                        ("image_lv2/fixed_2", warped_x2_lll_re[:1].squeeze()),
                        ("image_lv2/real", warped_x2_in[:1].squeeze()),
                        # ("image/grid", grid.squeeze())
                    ]:
                        image = image.detach().cpu()
                        # Check if the image is a 3D MRI image
                        slices = get_middle_slice(image)  # remove channel dimension and get slices
                        # for i, slice_img in enumerate(slices):
                        x_tb.append(slices[0].unsqueeze(0).unsqueeze(0))
                        y_tb.append(slices[1].unsqueeze(0).unsqueeze(0))
                        z_tb.append(slices[2].unsqueeze(0).unsqueeze(0))
                        
                # 将每个内部列表拼接
                x_tb_concat = [torch.cat([x], dim=0) for x in x_tb]
                x_ = torch.cat(x_tb_concat, dim=0)
                y_tb_concat = [torch.cat([y], dim=0) for y in y_tb]
                y_ = torch.cat(y_tb_concat, dim=0)
                z_tb_concat = [torch.cat([z], dim=0) for z in z_tb]
                z_ = torch.cat(z_tb_concat, dim=0)

                writer.add_image(f"slice_2/x", make_grid(x_, nrow=3), step)
                writer.add_image(f"slice_2/y", make_grid(y_, nrow=3), step)
                writer.add_image(f"slice_2/z", make_grid(z_, nrow=3), step)


            if (step % 5000 == 0):
                modelname = model_dir + model_name + str(step) + '.pth'
                torch.save(model2.state_dict(), modelname)
                np.save(model_dir + model_name + str(step) + '.npy', lossall)

                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0

                x1 = range(0, len(loss_total))
                x2 = range(0, len(loss_sim))
                x3 = range(0, len(loss_smo))
                x4 = range(0, len(loss_Jaco))
                y1 = loss_total
                y2 = loss_sim
                y3 = loss_smo
                y4 = loss_Jaco
                plt.switch_backend('agg')
                plt.subplot(2, 2, 1)
                plt.plot(x1, y1, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_total')
                plt.subplot(2, 2, 2)
                plt.plot(x2, y2, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_sim')
                plt.subplot(2, 2, 3)
                plt.plot(x3, y3, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_smo')
                plt.subplot(2, 2, 4)
                plt.plot(x4, y4, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_Jac')
                plt.tight_layout()
                # plt.show()
                plt.savefig(model_dir + str(step) + "_lv2.jpg")

            if step == freeze_step:
                model2.unfreeze_modellvl1()
            step += 1

            if step > iteration_lvl2:
                break
        print("-------------------------- level 2 epoch pass-------------------------")
    print("level 2 Finish!")

def train_lvl3():
    model_name = "WD_OAS_NCC_lvl3_"
    print("Training lvl3...")
    model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1).to(device)
    model2 = WavletMono_unit_add_lvl2(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2,
                                      model1=model1).to(device)
    model2_path = "./xxx/WD_OAS_NCC_lvl2_40000.pth"
    model2.load_state_dict(torch.load(model2_path))
    print("Loading weight for model_lvl2...", model2_path)

    # Freeze model_lvl1 weight
    for param in model2.parameters():
        param.requires_grad = False

    model3 = WavletMono_unit_add_lvl3(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2, imgshape3=imgshape3, model2=model2).to(device)

    loss_similarity1 = NCC(win=3)
    loss_similarity2 = NCC(win=5)
    loss_similarity3 = NCC(win=7)

    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_antifold = antifoldloss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    # OASIS
    names = sorted(glob.glob(datapath + '*.nii.gz'))

    grid = generate_grid(imgshape3)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model3.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './xxx'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl3+1))
    loss_total = []
    loss_smo = []
    loss_sim = []
    loss_Jaco = []
    loss_to = 0
    loss_sm = 0
    loss_si = 0
    loss_Ja = 0

    training_generator = Data.DataLoader(Dataset_epoch_train(names, norm=False), batch_size=1,
                                         shuffle=True, num_workers=4)
    

    load_model = False
    if load_model is True:
        model_path = "./xxx/WD_oas_NCC_reg_5.pth"
        print("Loading weight: ", model_path)
        model2.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("/xxx/WD_LPBA_OAS_reg_5.npy")
        lossall[:, 0:1000] = temp_lossall[:, 0:1000]

    step = 1
    while step <= iteration_lvl3:
        for X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2, label1, label2 in training_generator:

            X1_LLL = X1_LLL.to(device).float()
            Y1_LLL = Y1_LLL.to(device).float()
            X2_LLL = X2_LLL.to(device).float()
            Y2_LLL = Y2_LLL.to(device).float()
            X2_HHH = X2_HHH.to(device).float()
            Y2_HHH = Y2_HHH.to(device).float()
            X3_LLL = X3_LLL.to(device).float()
            Y3_LLL = Y3_LLL.to(device).float()
            X3_HHH = X3_HHH.to(device).float()
            Y3_HHH = Y3_HHH.to(device).float()

            
# 正向
            field1, field2, field3, field3_in, warped_x1, warped_x2_lll,  warped_x3_lll, warped_x2_hhh, warped_x3_hhh,\
                   fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, mov1, mov2, mov3, warped_x3_input, warped_x3_in\
               = model3(X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH)
            
# 逆向
            field1_re, field2_re, field3_re, field3_in_re, warped_x1_re, warped_x2_lll_re,  warped_x3_lll_re, \
                   warped_x2_hhh_re, warped_x3_hhh_re, fixed1_re, fixed2_lll_re, fixed2_hhh_re, fixed3_lll_re, \
                   fixed3_hhh_re, mov1_re, mov2_re, mov3_re, warped_x3_input_re, warped_x3_in_re\
               = model3(Y1_LLL, X1_LLL, Y2_LLL, X2_LLL, Y2_HHH, X2_HHH, Y3_LLL, X3_LLL, Y3_HHH, X3_HHH)

# 正逆向loss
            field3_in_re_clone = field3_in_re.clone()
            field3_re_clone = field3_re.clone()

            field3_in_clone = field3_in.clone()
            field3_clone = field3.clone()

            warp_filed = transform_near(field3_clone, field3.permute(0, 2, 3, 4, 1), grid)
            warp_filed_shape = warp_filed.shape
            warp_ones = torch.ones(warp_filed_shape[0], warp_filed_shape[1], warp_filed_shape[2],warp_filed_shape[3],warp_filed_shape[4]).to(device)
            criterion_f = nn.MSELoss()
            loss_filed = criterion_f(warp_filed, warp_ones)


#--------------------------一致性--------------------------------------------------
            warped_mean = (warped_x3_input + warped_x3_lll) / 2
            loss_mean = criterion_f(warped_x3_lll, warped_mean)

            warped_mean_re = (warped_x3_input_re + warped_x3_lll_re) / 2
            loss_mean_re = criterion_f(warped_x3_lll_re, warped_mean_re)

            loss_inv = criterion_f(warped_x3_in, warped_x3_lll_re)

            loss_inv_re = criterion_f(warped_x3_in_re, warped_x3_lll)




            # 3 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC2 = loss_similarity2(warped_x2_lll, fixed2_lll)
            loss_multiNCC4 = loss_similarity3(warped_x3_lll, fixed3_lll)
            loss_multiNCC22 = loss_similarity2(warped_x2_hhh, fixed2_hhh)
            loss_multiNCC44 = loss_similarity3(warped_x3_hhh, fixed3_hhh)
            loss_ncc1  = 0.25*loss_multiNCC1 + 0.5*(loss_multiNCC2+loss_multiNCC22) + (loss_multiNCC4+loss_multiNCC44)

            loss_multiNCC1_re = loss_similarity1(warped_x1_re, fixed1_re)
            loss_multiNCC2_re = loss_similarity2(warped_x2_lll_re, fixed2_lll_re)
            loss_multiNCC4_re = loss_similarity3(warped_x3_lll_re, fixed3_lll_re)
            loss_multiNCC22_re = loss_similarity2(warped_x2_hhh_re, fixed2_hhh_re)
            loss_multiNCC44_re = loss_similarity3(warped_x3_hhh_re, fixed3_hhh_re)
            loss_ncc2  = 0.25*loss_multiNCC1_re + 0.5*(loss_multiNCC2_re+loss_multiNCC22_re) + (loss_multiNCC4_re+loss_multiNCC44_re)


            loss_ncc3 = loss_similarity3(warped_x3_in, fixed3_lll_re)

            loss_ncc4 = loss_similarity3(warped_x3_in_re, fixed3_lll)

            loss_multiNCC = loss_ncc1 + loss_ncc2 + 0.1*(loss_ncc3 + loss_ncc4) 

            
            # field_norm = transform_unit_flow_to_flow_cuda(field3.permute(0,2,3,4,1).clone())
            # loss_Jacobian = loss_Jdet(field_norm, grid)

            _, _, x1, y1, z1 = field1.shape
            field1[:, 0, :, :, :] = field1[:, 0, :, :, :] * z1
            field1[:, 1, :, :, :] = field1[:, 1, :, :, :] * y1
            field1[:, 2, :, :, :] = field1[:, 2, :, :, :] * x1
            _, _, x2, y2, z2 = field2.shape
            field2[:, 0, :, :, :] = field2[:, 0, :, :, :] * z2
            field2[:, 1, :, :, :] = field2[:, 1, :, :, :] * y2
            field2[:, 2, :, :, :] = field2[:, 2, :, :, :] * x2
            _, _, x3, y3, z3 = field3.shape
            field3[:, 0, :, :, :] = field3[:, 0, :, :, :] * z3
            field3[:, 1, :, :, :] = field3[:, 1, :, :, :] * y3
            field3[:, 2, :, :, :] = field3[:, 2, :, :, :] * x3

            _, _, x2, y2, z2 = field1_re.shape
            field2_re[:, 0, :, :, :] = field2_re[:, 0, :, :, :] * z2
            field2_re[:, 1, :, :, :] = field2_re[:, 1, :, :, :] * y2
            field2_re[:, 2, :, :, :] = field2_re[:, 2, :, :, :] * x2

            _, _, x2, y2, z2 = field2_re.shape
            field2_re[:, 0, :, :, :] = field2_re[:, 0, :, :, :] * z2
            field2_re[:, 1, :, :, :] = field2_re[:, 1, :, :, :] * y2
            field2_re[:, 2, :, :, :] = field2_re[:, 2, :, :, :] * x2

            _, _, x3, y3, z3 = field3_re.shape
            field3_re[:, 0, :, :, :] = field3_re[:, 0, :, :, :] * z3
            field3_re[:, 1, :, :, :] = field3_re[:, 1, :, :, :] * y3
            field3_re[:, 2, :, :, :] = field3_re[:, 2, :, :, :] * x3

            _, _, x3, y3, z3 = field3_in.shape
            field3_in[:, 0, :, :, :] = field3_in[:, 0, :, :, :] * z3
            field3_in[:, 1, :, :, :] = field3_in[:, 1, :, :, :] * y3
            field3_in[:, 2, :, :, :] = field3_in[:, 2, :, :, :] * x3


            loss_regulation1 = loss_smooth(field1)
            loss_regulation2 = loss_smooth(field2)
            loss_regulation3 = loss_smooth(field3)
            loss_regulation4 = loss_smooth(field2_re)
            loss_regulation5 = loss_smooth(field1_re)
            loss_regulation6 = loss_smooth(field3_re)
            loss_regulation7 = loss_smooth(field3_in)
            loss_regulation = 0.25*(loss_regulation1+loss_regulation5) + 0.5*(loss_regulation2+loss_regulation4) + loss_regulation3+loss_regulation6+loss_regulation7

            loss_fold1 = loss_antifold(field1)
            loss_fold2 = loss_antifold(field2)
            loss_fold3 = loss_antifold(field3)
            loss_fold4 = loss_antifold(field3_re)

            loss_fold = 0.25 * loss_fold1 + 0.5*loss_fold2 + loss_fold3 +loss_fold4

            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss = loss_multiNCC + smooth * loss_regulation + loss_fold + 0.01 * loss_filed + 0.001*(1* loss_inv + 1*loss_inv_re + 0.5*loss_mean + 0.5*loss_mean_re) #loss_Jacobian
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(),  loss_regulation.item(), loss_fold.item()])

            loss_total.append(loss.item())
            loss_sim.append(loss_multiNCC.item())
            loss_smo.append(loss_regulation.item())
            loss_Jaco.append(loss_fold.item())

            loss_to = loss_to + loss.item()
            loss_si = loss_si + loss_multiNCC.item()
            loss_sm = loss_sm + loss_regulation.item()
            loss_Ja = loss_Ja + loss_fold.item()
            
            
            if (step % 500 == 0):
                x_tb = list()
                y_tb = list()
                z_tb = list()

                for tag, image in [
                        ("image_lv2/lv1", warped_x3_input.squeeze()),
                        ("image_lv2/lv2", warped_x3_lll.squeeze()),
                        ("image_lv2/fake", warped_x3_in_re.squeeze()),
                        ("image_lv2/fixed_2", warped_x3_lll_re.squeeze()),
                        ("image_lv2/real", warped_x3_in.squeeze()),
                        # ("image/grid", grid.squeeze())
                    ]:
                        image = image.detach().cpu()
                        # Check if the image is a 3D MRI image
                        slices = get_middle_slice(image)  # remove channel dimension and get slices
                        # for i, slice_img in enumerate(slices):
                        x_tb.append(slices[0].unsqueeze(0).unsqueeze(0))
                        y_tb.append(slices[1].unsqueeze(0).unsqueeze(0))
                        z_tb.append(slices[2].unsqueeze(0).unsqueeze(0))
                        
                # 将每个内部列表拼接
                x_tb_concat = [torch.cat([x], dim=0) for x in x_tb]
                x_ = torch.cat(x_tb_concat, dim=0)
                y_tb_concat = [torch.cat([y], dim=0) for y in y_tb]
                y_ = torch.cat(y_tb_concat, dim=0)
                z_tb_concat = [torch.cat([z], dim=0) for z in z_tb]
                z_ = torch.cat(z_tb_concat, dim=0)

                writer.add_image(f"slice_3/x", make_grid(x_, nrow=3), step)
                writer.add_image(f"slice_3/y", make_grid(y_, nrow=3), step)
                writer.add_image(f"slice_3/z", make_grid(z_, nrow=3), step)


                total = format(float(loss_to) / float(1000), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si) / float(1000), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(1000), '.6f')
                #loss_smo.append(smo)
                Jaco = format(float(loss_Ja) / float(1000), '.9f')
                #loss_Jaco.append(Jaco)
                Fil = format(float(loss_filed) / float(1000), '.9f')
                L_m = format(float(loss_mean*0.5) / float(1000), '.9f')
                L_m_r = format(float(loss_mean_re*0.5) / float(1000), '.9f')
                L_i = format(float(loss_inv*1) / float(1000), '.9f')
                L_i_r = format(float(loss_inv_re*1) / float(1000), '.9f')

                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco),\
                      'Filed:'+str(Fil), 'mean:'+str(L_m), 'meanre:'+str(L_m_r), 'inv:'+str(L_i), 'invr:'+str(L_i_r))
            
                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0


            if (step % 5000 == 0):
                modelname = model_dir + model_name + str(step) + '.pth'
                torch.save(model3.state_dict(), modelname)
                np.save(model_dir + model_name + str(step) + '.npy', lossall)

                x1 = range(0, len(loss_total))
                x2 = range(0, len(loss_sim))
                x3 = range(0, len(loss_smo))
                x4 = range(0, len(loss_Jaco))
                y1 = loss_total
                y2 = loss_sim
                y3 = loss_smo
                y4 = loss_Jaco
                plt.switch_backend('agg')
                plt.subplot(2, 2, 1)
                plt.plot(x1, y1, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_total')
                plt.subplot(2, 2, 2)
                plt.plot(x2, y2, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_sim')
                plt.subplot(2, 2, 3)
                plt.plot(x3, y3, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_smo')
                plt.subplot(2, 2, 4)
                plt.plot(x4, y4, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_Jac')
                plt.tight_layout()
                # plt.show()
                plt.savefig(model_dir + str(step) + "_lv3.jpg")

            if step == freeze_step:
                model3.unfreeze_modellvl2()
            step += 1

            if step > iteration_lvl3:
                break
        print("-------------------------- level 3 epoch pass-------------------------")
    print("level 3 Finish!")

def train_lvl4():
    torch.autograd.set_detect_anomaly(True)

    model_name = "WD_OAS_NCC_lvl4_"
    print("Training lvl4...")
    model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape1=imgshape1).to(device)
    model2 = WavletMono_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2,
                                      model1=model1).to(device)
    model3 = WavletMono_unit_add_lvl3(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2, imgshape3=imgshape3,
                                      model2=model2).to(device)

    model3_path = "./xxx/WD_OAS_NCC_lvl3_60000.pth"
    model3.load_state_dict(torch.load(model3_path))
    print("Loading weight for model_lvl3...", model3_path)

    # Freeze model_lvl3 weight
    for param in model3.parameters():
        param.requires_grad = False

    model4 = WavletMono_unit_add_lvl4(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2, imgshape3=imgshape3, imgshape4=imgshape4, model3=model3).to(device)

    loss_similarity1 = NCC(win=3)
    loss_similarity2 = NCC(win=5)
    loss_similarity3 = NCC(win=7)
    loss_similarity4 = NCC(win=9)

    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss
    loss_antifold = antifoldloss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    # OASIS
    names = sorted(glob.glob(datapath + '*.nii.gz'))

    grid = generate_grid(imgshape4)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model3.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './xxx'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl4+1))
    loss_total = []
    loss_smo = []
    loss_sim = []
    loss_Jaco = []
    loss_F = []
    loss_to = 0
    loss_sm = 0
    loss_si = 0
    loss_Ja = 0
    loss_f = 0
    loss_D = 0

    training_generator = Data.DataLoader(Dataset_epoch_train(names, norm=False), batch_size=1,
                                         shuffle=True, num_workers=4)

    

    load_model = False
    if load_model is True:
        model_path = "/XXX/WD_OAS_NCC_reg_5.pth"
        print("Loading weight: ", model_path)
        model2.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("/XXX/WD_OAS_NCC_reg_5.npy")
        lossall[:, 0:1000] = temp_lossall[:, 0:1000]

    step = 1
    while step <= iteration_lvl4:
        for X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2, label1, label2 in training_generator:

            X1_LLL = X1_LLL.to(device).float()
            Y1_LLL = Y1_LLL.to(device).float()
            X2_LLL = X2_LLL.to(device).float()
            Y2_LLL = Y2_LLL.to(device).float()
            X2_HHH = X2_HHH.to(device).float()
            Y2_HHH = Y2_HHH.to(device).float()
            X3_LLL = X3_LLL.to(device).float()
            Y3_LLL = Y3_LLL.to(device).float()
            X3_HHH = X3_HHH.to(device).float()
            Y3_HHH = Y3_HHH.to(device).float()
            source1 = source1.to(device).float()
            source2 = source2.to(device).float()
            label1 = label1.to(device)
            label2 = label2.to(device)

# 正向
            field1, field2, field3, field4, field4_in, warped_x1, warped_x2_lll,  warped_x3_lll, \
                   warped_x2_hhh, warped_x3_hhh, warped_source1, fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, \
                   fixed3_hhh, fixed4_source2, mov1, mov2, mov3, source1, diff_up4, warped_source1_input, warped_source1_in\
                = model4(X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1,
                        source2, label1, label2)
            
# 逆向
            field1_re, field2_re, field3_re, field4_re, field4_re_in, warped_x1_re, warped_x2_lll_re,  warped_x3_lll_re, \
                   warped_x2_hhh_re, warped_x3_hhh_re, warped_source1_re, fixed1_re, fixed2_lll_re, fixed2_hhh_re, fixed3_lll_re, \
                   fixed3_hhh_re, fixed4_source2_re, mov1_re, mov2_re, mov3_re, source1_re, diff_up4_re, warped_source1_input_re, warped_source1_in_re\
                = model4(Y1_LLL, X1_LLL, Y2_LLL, X2_LLL, Y2_HHH, X2_HHH, Y3_LLL, X3_LLL, Y3_HHH, X3_HHH, source2,
                        source1, label2, label1)

# 正逆向loss
            field4_in_re_clone = field4_re_in.clone()
            field4_re_clone = field4_re.clone()

            field4_in_clone = field4_in.clone()
            field4_clone = field4.clone()

            warp_filed = transform_near(field4_clone, field4.permute(0, 2, 3, 4, 1), grid)
            warp_filed_shape = warp_filed.shape
            warp_ones = torch.ones(warp_filed_shape[0], warp_filed_shape[1], warp_filed_shape[2],warp_filed_shape[3],warp_filed_shape[4]).to(device)
            criterion_f = nn.MSELoss()
            loss_filed = criterion_f(warp_filed, warp_ones)


#--------------------------一致性--------------------------------------------------
            warped_mean = (warped_source1_input + warped_source1) / 2
            loss_mean = criterion_f(warped_source1, warped_mean)

            warped_mean_re = (warped_source1_input_re + warped_source1_re) / 2
            loss_mean_re = criterion_f(warped_source1_re, warped_mean_re)

            loss_inv = criterion_f(warped_source1_in, warped_source1_re)

            loss_inv_re = criterion_f(warped_source1_in_re, warped_source1)


            # 3 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC2 = loss_similarity2(warped_x2_lll, fixed2_lll)
            loss_multiNCC4 = loss_similarity3(warped_x3_lll, fixed3_lll)
            loss_multiNCC6 = loss_similarity4(warped_source1, fixed4_source2)
            loss_multiNCC22 = loss_similarity2(warped_x2_hhh, fixed2_hhh)
            loss_multiNCC44 = loss_similarity3(warped_x3_hhh, fixed3_hhh)

            loss_ncc1  = 0.125*loss_multiNCC1 + 0.25*(loss_multiNCC2 + loss_multiNCC22) + 0.5*(loss_multiNCC4 + loss_multiNCC44) + loss_multiNCC6
            

            loss_multiNCC1_re = loss_similarity1(warped_x1_re, fixed1_re)
            loss_multiNCC2_re = loss_similarity2(warped_x2_lll_re, fixed2_lll_re)
            loss_multiNCC4_re = loss_similarity3(warped_x3_lll_re, fixed3_lll_re)
            loss_multiNCC6_re = loss_similarity4(warped_source1_re, fixed4_source2_re)
            loss_multiNCC22_re = loss_similarity2(warped_x2_hhh_re, fixed2_hhh_re)
            loss_multiNCC44_re = loss_similarity3(warped_x3_hhh_re, fixed3_hhh_re)

            loss_ncc2  = 0.125*loss_multiNCC1_re + 0.25*(loss_multiNCC2_re + loss_multiNCC22_re) + 0.5*(loss_multiNCC4_re + loss_multiNCC44_re) + loss_multiNCC6_re


            loss_ncc3 = loss_similarity3(warped_source1_in, fixed4_source2_re)

            loss_ncc4 = loss_similarity3(warped_source1_in_re, fixed4_source2)

            loss_multiNCC = loss_ncc1 + loss_ncc2 + 0.1*(loss_ncc3 + loss_ncc4) 

            
            # field_norm = transform_unit_flow_to_flow_cuda(field4.permute(0,2,3,4,1).clone())
            # loss_Jacobian = loss_Jdet(field_norm, grid)

            _, _, x1, y1, z1 = field1.shape
            field1[:, 0, :, :, :] = field1[:, 0, :, :, :] * z1
            field1[:, 1, :, :, :] = field1[:, 1, :, :, :] * y1
            field1[:, 2, :, :, :] = field1[:, 2, :, :, :] * x1
            _, _, x2, y2, z2 = field2.shape
            field2[:, 0, :, :, :] = field2[:, 0, :, :, :] * z2
            field2[:, 1, :, :, :] = field2[:, 1, :, :, :] * y2
            field2[:, 2, :, :, :] = field2[:, 2, :, :, :] * x2
            _, _, x3, y3, z3 = field3.shape
            field3[:, 0, :, :, :] = field3[:, 0, :, :, :] * z3
            field3[:, 1, :, :, :] = field3[:, 1, :, :, :] * y3
            field3[:, 2, :, :, :] = field3[:, 2, :, :, :] * x3
            _, _, x4, y4, z4 = field4.shape
            field4[:, 0, :, :, :] = field4[:, 0, :, :, :] * z4
            field4[:, 1, :, :, :] = field4[:, 1, :, :, :] * y4
            field4[:, 2, :, :, :] = field4[:, 2, :, :, :] * x4

            _, _, x2, y2, z2 = field1_re.shape
            field2_re[:, 0, :, :, :] = field2_re[:, 0, :, :, :] * z2
            field2_re[:, 1, :, :, :] = field2_re[:, 1, :, :, :] * y2
            field2_re[:, 2, :, :, :] = field2_re[:, 2, :, :, :] * x2

            _, _, x2, y2, z2 = field2_re.shape
            field2_re[:, 0, :, :, :] = field2_re[:, 0, :, :, :] * z2
            field2_re[:, 1, :, :, :] = field2_re[:, 1, :, :, :] * y2
            field2_re[:, 2, :, :, :] = field2_re[:, 2, :, :, :] * x2

            _, _, x3, y3, z3 = field3_re.shape
            field3_re[:, 0, :, :, :] = field3_re[:, 0, :, :, :] * z3
            field3_re[:, 1, :, :, :] = field3_re[:, 1, :, :, :] * y3
            field3_re[:, 2, :, :, :] = field3_re[:, 2, :, :, :] * x3

            _, _, x3, y3, z3 = field4_re.shape
            field4_re[:, 0, :, :, :] = field4_re[:, 0, :, :, :] * z3
            field4_re[:, 1, :, :, :] = field4_re[:, 1, :, :, :] * y3
            field4_re[:, 2, :, :, :] = field4_re[:, 2, :, :, :] * x3

            _, _, x4, y4, z4 = field4_in.shape
            field4_in[:, 0, :, :, :] = field4_in[:, 0, :, :, :] * z4
            field4_in[:, 1, :, :, :] = field4_in[:, 1, :, :, :] * y4
            field4_in[:, 2, :, :, :] = field4_in[:, 2, :, :, :] * x4


            loss_regulation1 = loss_smooth(field1)
            loss_regulation2 = loss_smooth(field2)
            loss_regulation3 = loss_smooth(field3)
            loss_regulation4 = loss_smooth(field4)
            loss_regulation5 = loss_smooth(field1_re)
            loss_regulation6 = loss_smooth(field3_re)
            loss_regulation7 = loss_smooth(field2_re)
            loss_regulation8 = loss_smooth(field4_re)
            loss_regulation9 = loss_smooth(field4_in)

            loss_regulation = 0.125*(loss_regulation1+loss_regulation5) + 0.25*(loss_regulation2+loss_regulation7)\
                + 0.5*(loss_regulation3+loss_regulation6) + loss_regulation4 + loss_regulation8 + loss_regulation9

            loss_fold1 = loss_antifold(field1)
            loss_fold2 = loss_antifold(field2)
            loss_fold3 = loss_antifold(field3)
            loss_fold4 = loss_antifold(field4)

            loss_fold1_re = loss_antifold(field1_re)
            loss_fold2_re = loss_antifold(field2_re)
            loss_fold3_re = loss_antifold(field3_re)
            loss_fold4_re = loss_antifold(field4_re)

 
            loss_fold = 0.125 * loss_fold1 + 0.25*loss_fold2 + 0.5*loss_fold3 + loss_fold4 + 0.125 * loss_fold1_re + 0.25*loss_fold2_re + 0.5*loss_fold3_re + loss_fold4_re

            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss =  loss_multiNCC + loss_regulation + loss_fold + 0.01 * loss_filed + 0.001*(1* loss_inv + 1*loss_inv_re + 0.5*loss_mean + 0.5*loss_mean_re)
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients


            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(),  loss_regulation.item(), loss_fold.item()])

            loss_total.append(loss.item())
            loss_sim.append(loss_multiNCC.item())
            loss_smo.append(loss_regulation.item())
            loss_Jaco.append(loss_fold.item())
            loss_F.append(loss_filed.item())

            loss_to = loss_to + loss.item()
            loss_si = loss_si + loss_multiNCC.item()
            loss_sm = loss_sm + (loss_regulation).item()
            loss_Ja = loss_Ja + loss_fold.item()
            loss_f = loss_f + loss_filed.item()
            # loss_D = loss_D + lossGD.item()

            if (step % 500 == 0) or step == 1:
                total = format(float(loss_to) / float(1000), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si) / float(1000), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(1000), '.6f')
                #loss_smo.append(smo)
                Jaco = format(float(loss_Ja) / float(1000), '.9f')
                #loss_Jaco.append(Jaco)
                Fil = format(float(loss_f) / float(1000), '.9f')
                L_m = format(float(loss_mean*0.5) / float(1000), '.9f')
                L_m_r = format(float(loss_mean_re*0.5) / float(1000), '.9f')
                L_i = format(float(loss_inv*1) / float(1000), '.9f')
                L_i_r = format(float(loss_inv_re*1) / float(1000), '.9f')

                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco), 'Filed:'+str(Fil), 'mean:'+str(L_m), 'meanre:'+str(L_m_r), 'inv:'+str(L_i), 'invr:'+str(L_i_r))
                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0
                loss_f = 0
                loss_D = 0
                

                x_tb = list()
                y_tb = list()
                z_tb = list()

                for tag, image in [
                        ("image_lv4/lv1", warped_source1_input[:1].squeeze()),
                        ("image_lv4/label_1", warped_source1_input_re[:1].squeeze()),
                        ("image_lv4/lv2", warped_source1[:1].squeeze()),
                        ("image_lv4/lv2", warped_source1_in_re[:1].squeeze()),
                        ("image_lv4/lv2", warped_source1_in[:1].squeeze()),
                        ("image_lv4/lv2", warped_source1_re[:1].squeeze()),
                        # ("image/grid", grid.squeeze())
                    ]:
                        image = image.detach().cpu()
                        # Check if the image is a 3D MRI image
                        slices = get_middle_slice(image)  # remove channel dimension and get slices
                        # for i, slice_img in enumerate(slices):
                        x_tb.append(slices[0].unsqueeze(0).unsqueeze(0))
                        y_tb.append(slices[1].unsqueeze(0).unsqueeze(0))
                        z_tb.append(slices[2].unsqueeze(0).unsqueeze(0))


                # 将每个内部列表拼接
                x_tb_concat = [torch.cat([x], dim=0) for x in x_tb]
                x_ = torch.cat(x_tb_concat, dim=0)
                y_tb_concat = [torch.cat([y], dim=0) for y in y_tb]
                y_ = torch.cat(y_tb_concat, dim=0)
                z_tb_concat = [torch.cat([z], dim=0) for z in z_tb]
                z_ = torch.cat(z_tb_concat, dim=0)

                writer.add_image(f"slice_4/x", make_grid(x_, nrow=3), step)
                writer.add_image(f"slice_4/y", make_grid(y_, nrow=3), step)
                writer.add_image(f"slice_4/z", make_grid(z_, nrow=3), step)


            if (step % 20000 != 0):
                del X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2, field1, field2, field3, field4, warped_x1, warped_x2_lll, warped_x3_lll, warped_x2_hhh, warped_x3_hhh, warped_source1, fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, fixed4_source2, mov1, mov2, mov3
                torch.cuda.empty_cache()


            if (step % 5000 == 0):
                modelname = model_dir + model_name + str(step) + '.pth'
                torch.save(model4.state_dict(), modelname)
                np.save(model_dir + model_name + str(step) + '.npy', lossall)
                
                
            if (step % 20000 == 0):

                x1 = range(0, len(loss_total))
                x2 = range(0, len(loss_sim))
                x3 = range(0, len(loss_smo))
                x4 = range(0, len(loss_Jaco))
                y1 = loss_total
                y2 = loss_sim
                y3 = loss_smo
                y4 = loss_Jaco
                plt.switch_backend('agg')
                plt.subplot(2, 2, 1)
                plt.plot(x1, y1, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_total')
                plt.subplot(2, 2, 2)
                plt.plot(x2, y2, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_sim')
                plt.subplot(2, 2, 3)
                plt.plot(x3, y3, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_smo')
                plt.subplot(2, 2, 4)
                plt.plot(x4, y4, 'r-')
                plt.xlabel('iteration')
                plt.ylabel('loss_Jac')
                plt.tight_layout()
                # plt.show()
                plt.savefig(model_dir + str(step) + "_lv4.jpg")
                del X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2, field1, field2, field3, field4, warped_x1, warped_x2_lll, warped_x3_lll, warped_x2_hhh, warped_x3_hhh, warped_source1, fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, fixed4_source2, mov1, mov2, mov3
                torch.cuda.empty_cache()

            if step == freeze_step:
                model4.unfreeze_modellvl3()

            step += 1

            if step > iteration_lvl4:
                break
        print("-------------------------- level 4 epoch pass-------------------------")
    print("level 4 Finish!")



range_flow = 7
imgshape4 = (128, 128, 128)
imgshape3 = (64, 64, 64)
imgshape2 = (32, 32, 32)
imgshape1 = (16, 16, 16)

tb_dir =  "./tensorboard"
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
writer = SummaryWriter(log_dir=str(tb_dir))


train_lvl1()
train_lvl2()
train_lvl3()
train_lvl4()

