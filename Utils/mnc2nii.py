
#!/usr/bin/python
import os
import subprocess
import glob
inPath = '/data1/ctt/TrainingData/OASIS1/orignal/'
outPath = '/data1/ctt/TrainingData/OASIS1/mnc2nii/'

names = sorted(glob.glob(inPath + '*.mnc'))[0:410]

for i in names:
    name1 = i.split('/')[6]
    name2 = i.split('/')[6].split('.mnc')[0]
    os.system('mri_convert '  + str(i) + "/" + str(name1) + " " + outPath + str(name2) + ".nii")

#subprocess.Popen


# import torch
# import SimpleITK as sitk
# import os
# import random
# import numpy
# import network
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# import Losses.gradientLoss as GL
# from matplotlib import pyplot as plt
# import warp_image
# from PIL import Image
#
# path = '/data/tongtong/ANTS-test/NT/'  # 数据所在路径
# savePath='/data/tongtong/ANTS-test/NT/NT-DL/'
# Gnet=network.unet_3d()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Gnet.to(device)
#
# # Define a Loss function and optimizer
# Gcriterion = nn.MSELoss().to(device)
# optimizer = torch.optim.Adam(
#     filter(lambda p: p.requires_grad, Gnet.parameters()), lr=0.0001)
#
# zero = numpy.zeros((1,3, 24, 24, 24), dtype=numpy.int)
# zero = torch.from_numpy(zero).type(torch.FloatTensor)
#
# GimgF = sitk.ReadImage(os.path.join(path, 'T_hist.nii.gz'))
# GarrayF = sitk.GetArrayFromImage(GimgF)
# GtensorF = torch.from_numpy(GarrayF).type(torch.FloatTensor)
# GnumpyF = GtensorF.numpy()
# GpadF = numpy.pad(GnumpyF, ((29, 29), (11, 11), (29, 29)), 'constant')
# GpadF = (GpadF - numpy.min(GpadF)) / (numpy.max(GpadF) - numpy.min(GpadF)) * 255
# if GpadF.shape == (239, 239, 239):
#     GpadF = numpy.pad(GpadF, ((1, 0), (1, 0), (1, 0)), 'constant')
# im = Image.fromarray(GpadF[152])
# #im.save(savePath+"1.jpg")
# plt.imsave(savePath+"1.jpg",GpadF[152, :, :])
# #plt.show()
