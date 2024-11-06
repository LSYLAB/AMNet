import math
import torch
import numpy as np
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F


# RCN
def jacobian_det(flow):
    _, std = torch.std(torch.det(torch.det([
        flow[:, :, 1:, :-1, :-1] - flow[:, :, :-1, :-1, :-1] +
        torch.Tensor([1, 0, 0], dtype=torch.float32),
        flow[:, :, :-1, 1:, :-1, :] - flow[:, :, :-1, :-1, :-1] +
        torch.Tensor([0, 1, 0], dtype=torch.float32),
        flow[:, :, :-1, :-1, 1:] - flow[:, :, :-1, :-1, :-1] +
        torch.Tensor([0, 0, 1], dtype=torch.float32)
    ], dim=-1)), dim=[2, 3, 4])
    return std


# FAIM
def Get_Ja(flow):
    print(flow.shape)
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(ypred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)


# torch.random.manual_seed(0)
path_m2f='/home/che/Temp/field4_epoch20.nii'
m2f_flow=sitk.GetArrayFromImage(sitk.ReadImage(path_m2f))[np.newaxis,...]
m2f_flow=torch.from_numpy(m2f_flow)
m2f_flow=m2f_flow.permute(0,2,3,4,1)
print("shape: ",Get_Ja(m2f_flow).shape)
print("m2f_NJ_loss: ",NJ_loss(m2f_flow).data.item())


