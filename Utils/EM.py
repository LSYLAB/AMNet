import nibabel as nib
import torch
import numpy as np
import SimpleITK as sitk
import glob
import os

########## Energy map ##########################
save_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD2/EM2/'
LLH_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD2/LLH2/'
LHL_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD2/LHL2/'
HLL_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD2/HLL2/'
LHH_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD2/LHH2/'
HLH_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD2/HLH2/'
HHL_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD2/HHL2/'
HHH_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD2/HHH2/'
names = sorted(glob.glob(LLH_path + '*.nii'))[0:405]
for i in names:
    name = i.split('/')[8].split('_')[0]
    llh = sitk.ReadImage(LLH_path + name + '_LLH2.nii')
    llh_arr = sitk.GetArrayFromImage(llh)
    lhl = sitk.ReadImage(LHL_path + name + '_LHL2.nii')
    lhl_arr = sitk.GetArrayFromImage(lhl)
    hll = sitk.ReadImage(HLL_path + name + '_HLL2.nii')
    hll_arr = sitk.GetArrayFromImage(hll)
    hlh = sitk.ReadImage(HLH_path + name + '_HLH2.nii')
    hlh_arr = sitk.GetArrayFromImage(hlh)
    lhh = sitk.ReadImage(LHH_path + name + '_LHH2.nii')
    lhh_arr = sitk.GetArrayFromImage(lhh)
    hhl = sitk.ReadImage(HHL_path + name + '_HHL2.nii')
    hhl_arr = sitk.GetArrayFromImage(hhl)
    hhh = sitk.ReadImage(HHH_path + name + '_HHH2.nii')
    hhh_arr = sitk.GetArrayFromImage(hhh)
    E = (llh_arr * llh_arr) + (lhl_arr * lhl_arr) + (hll_arr * hll_arr) + (hlh_arr * hlh_arr) + (lhh_arr * lhh_arr) + (
                hhl_arr * hhl_arr) + (hhh_arr * hhh_arr)
    img = sitk.GetImageFromArray(E)
    sitk.WriteImage(img, save_path + name + '_EM2.nii')
print("Finish!")





