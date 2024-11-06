import numpy as np
import pywt
import matplotlib.pyplot as plt
import nibabel as nib

# 读取3D图像
image_path = 'BraTS2021_00495/BraTS2021_00495_flair.nii.gz'
image = nib.load(image_path).get_fdata()

# 小波变换
coeffs = pywt.dwtn(image, 'db2')

# 分离低频和高频子图
low_freq = coeffs['aaa']
high_freq = [coeffs['aad'], coeffs['ada'], coeffs['add'], coeffs['daa'], coeffs['dda'], coeffs['dad'], coeffs['ddd']]

# 保存子图
for i, freq in enumerate(high_freq):
    nib.save(nib.Nifti1Image(freq, np.eye(4)), f'results/high_freq_{i+1}.nii.gz')
nib.save(nib.Nifti1Image(low_freq, np.eye(4)), 'results/low_freq.nii.gz')

# 可视化子图
fig, axs = plt.subplots(2, 4, figsize=(10, 5))
axs[0][0].imshow(low_freq[:, :, 10],cmap='gray')
axs[0][0].set_title('Low Frequency')
for i, freq in enumerate(high_freq):
    axs[(i+1)//4][(i+1)%4].imshow(freq[:, :, 10],cmap='gray')
    axs[(i+1)//4][(i+1)%4].set_title(f'High Frequency {i+1}')
plt.show()