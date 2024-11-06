import os
from argparse import ArgumentParser
from thop import profile
import numpy as np
import torch

from WATFunctions import generate_grid_unit, save_img, save_flow, transform_unit_flow_to_flow, load_4D
from model_stage1_6_mix import WavletMono_unit_add_lvl1, WavletMono_unit_add_lvl2, WavletMono_unit_add_lvl3, WavletMono_unit_add_lvl4, SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, antifoldloss
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='/data1/ctt/Projects/WaveletReg/Wavelet_mono_reg/Code_WAT1-6_mix/Model/stage/lvl4/comp_WD_LPBA_NCC_lvl4_60000.pth',
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='/data1/ctt/Projects/WaveletReg/Wavelet_mono_reg/Code_WAT1-6_mix/Results-OAS-comp/',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='/data1/ctt/TestData/OASIS1-img/',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='data1/ctt/TestData/OASIS1-img/',
                    help="moving image")
opt = parser.parse_args()

savepath = opt.savepath
fixed_path = opt.fixed
moving_path = opt.moving
if not os.path.isdir(savepath):
    os.mkdir(savepath)

imgshape4 = (160, 184, 144)
imgshape3 = (80, 92, 72)
imgshape2 = (40, 46, 36)
imgshape1 = (20, 23, 18)

range_flow = 0.4

start_channel = opt.start_channel

model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape1=imgshape1).to(device)
model2 = WavletMono_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2,
                                  model1=model1).to(device)
model3 = WavletMono_unit_add_lvl3(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2,
                                  imgshape3=imgshape3, model2=model2).to(device)

model4 = WavletMono_unit_add_lvl4(2, 3, start_channel, is_train=False, imgshape1=imgshape1, imgshape2=imgshape2,
                                  imgshape3=imgshape3, imgshape4=imgshape4, model3=model3).to(device)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def FLOPs(model,size):
#     dummy_input = torch.randn(size).to(device).float()
#     flops, params = profile(model, (dummy_input, dummy_input))
#     #print('flops: ', flops, 'params: ', params)
#     return flops,params
total_trainable_params = sum(
    p.numel() for p in model4.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters_A.')
print("Model1 number of parameters: ", count_parameters(model1))
print("Model2 number of parameters: ", count_parameters(model2))
print("Model3 number of parameters: ", count_parameters(model3))
print("Total number of parameters: ", count_parameters(model4))
print('-------------------------------------------------------')
dummy_input = torch.randn(1,1,20, 23, 18).to(device).float()
flops, params = profile(model1, (dummy_input, dummy_input))
print('Model1: flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
dummy_input1 = torch.randn(1,1,20, 23, 18).to(device).float()
dummy_input2 = torch.randn(1,1,40, 46, 36).to(device).float()
flops, params = profile(model2, (dummy_input1, dummy_input1,dummy_input2, dummy_input2,dummy_input2, dummy_input2))
print('Model2: flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
dummy_input1 = torch.randn(1,1,20, 23, 18).to(device).float()
dummy_input2 = torch.randn(1,1,40, 46, 36).to(device).float()
dummy_input3 = torch.randn(1,1,80, 92, 72).to(device).float()
flops, params = profile(model3, (dummy_input1, dummy_input1,dummy_input2, dummy_input2,dummy_input2, dummy_input2,dummy_input3, dummy_input3,dummy_input3, dummy_input3))
print('Model3: flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
dummy_input1 = torch.randn(1,1,20, 23, 18).to(device).float()
dummy_input2 = torch.randn(1,1,40, 46, 36).to(device).float()
dummy_input3 = torch.randn(1,1,80, 92, 72).to(device).float()
dummy_input4 = torch.randn(1,1,160, 184, 144).to(device).float()
flops, params = profile(model4, (dummy_input1, dummy_input1,dummy_input2, dummy_input2,dummy_input2, dummy_input2,dummy_input3, dummy_input3,dummy_input3, dummy_input3,dummy_input4, dummy_input4))
print('Model4: flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))