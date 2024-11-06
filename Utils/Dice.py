import SimpleITK as sitk
import torch
import torch.nn as nn
import numpy as np
import glob

########################## Tensor-based Dice#############################
img_path =  '/home/che/Temp/'
def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")

    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N

img_A = img = sitk.ReadImage(img_path+'fixed4_source2-epoch20.nii')
A = sitk.GetArrayFromImage(img_A)
tensor_A = torch.from_numpy(A)
img_B = sitk.ReadImage(img_path+'warped_source1-epoch20.nii')
B = sitk.GetArrayFromImage(img_B)
tensor_B = torch.from_numpy(B)
dice_tumor1 = diceCoeff(tensor_A[:, :, :], tensor_B[:, :, :], smooth=1, activation=None)
dice_tumor2 = diceCoeff(tensor_A[:, :, :], tensor_B[:, :, :], smooth=1e-5, activation=None)
print('AMRNet-smooth=1 : dice={:.4}'.format(dice_tumor1.item()))
print('AMRNet-smooth=1e-5 : dice={:.4}'.format(dice_tumor2.item()))

########################## Numpy-based Dice#############################
# files =  sorted(glob.glob(img_path + '*.nii'))[0:255]
# img_A  = sitk.ReadImage(img_path+'F.nii')
# A = sitk.GetArrayFromImage(img_A)
# row, col, high = A.shape[0], A.shape[1], A.shape[2]
# print(row, col, high)
# d = []
# for file in files:
#     img_B  = sitk.ReadImage(img_path + 'M.nii')
#     B = sitk.GetArrayFromImage(img_B)
#     s = []
#     for r in range(row):
#         for c in range(col):
#             for h in range(high):
#                 if A[r][c][h] == B[r][c][h]:#计算图像像素交集
#                     s.append(A[r][c][h])
#     m1 = np.linalg.norm(s)
#     print("m1",m1)
#     m2 = np.linalg.norm(A.flatten()) + np.linalg.norm(B.flatten())
#     print("m2", m2)
#     d.append(2*m1/m2)
# #     print(2*m1/m2)
# print("DSC",d)

