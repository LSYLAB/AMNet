import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import itertools
import os
import SimpleITK as sitk

def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * z
    flow[:, :, :, 1] = flow[:, :, :, 1] * y
    flow[:, :, :, 2] = flow[:, :, :, 2] * x

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * z
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * y
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * x

    return flow


def load_4D(name):
    # X = nib.load(name)
    # X = X.get_fdata()
    X = sitk.ReadImage(name)
    X = sitk.GetArrayFromImage(X)
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_5D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + (1,) + X.shape)
    return X


def imgnorm(N_I, index1=0.0001, index2=0.0001):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1 * len(I_sort))]
    I_max = I_sort[-int(index2 * len(I_sort))]

    N_I = 1.0 * (N_I - I_min) / (I_max - I_min)
    N_I[N_I > 1.0] = 1.0
    N_I[N_I < 0.0] = 0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2


def Norm_Zscore(img):
    img = (img - np.mean(img)) / np.std(img)
    return img


def save_img(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_img_nii(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


class Dataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, iterations, norm=True):
        'Initialization'
        self.names = names
        self.norm = norm
        self.iterations = iterations

    def __len__(self):
        'Denotes the total number of samples'
        return self.iterations

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        index_pair = np.random.permutation(len(self.names))[0:2]
        img_A = load_4D(self.names[index_pair[0]])
        img_B = load_4D(self.names[index_pair[1]])
        if self.norm:
            return Norm_Zscore(imgnorm(img_A)), Norm_Zscore(imgnorm(img_B))
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, norm=True):
        'Initialization'
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        LLL2_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD2/LLL2/'
        EM2_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD2/EM2/'
        LLL1_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD1/LLL1/'
        EM1_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/WD1/EM1/'
        source_path = '/data1/ctt/TrainingData/OASIS1/WaveImg/Source/'
        # Select sample
        LLL3_A = self.index_pair[step][0]
        img3_A = load_4D(LLL3_A)

        LLL3_B = self.index_pair[step][1]
        img3_B = load_4D(LLL3_B)


        num1 = LLL3_A.split('/')[8].split('_')[0]
        LLL2_A = os.path.join(LLL2_path,num1+'_LLL2.nii')
        img2_LLL_A= load_4D(LLL2_A)

        num2 = LLL3_B.split('/')[8].split('_')[0]
        LLL2_B = os.path.join(LLL2_path, num2+ '_LLL2.nii')
        img2_LLL_B = load_4D(LLL2_B)


        EM2_A = os.path.join(EM2_path, num1+'_EM2.nii')
        img2_EM_A = load_4D(EM2_A)

        EM2_B = os.path.join(EM2_path, num2+'_EM2.nii')
        img2_EM_B = load_4D(EM2_B)


        LLL1_A = os.path.join(LLL1_path, num1 + '_LLL1.nii')
        img1_LLL_A = load_4D(LLL1_A)

        LLL1_B = os.path.join(LLL1_path, num2 + '_LLL1.nii')
        img1_LLL_B = load_4D(LLL1_B)


        EM1_A = os.path.join(EM1_path, num1 + '_EM1.nii')
        img1_EM_A = load_4D(EM1_A)

        EM1_B = os.path.join(EM1_path, num2 + '_EM1.nii')
        img1_EM_B = load_4D(EM1_B)

        source_A = os.path.join(source_path,  num1 + '_source.nii')
        img1_source_A = load_4D(source_A)
        source_B = os.path.join(source_path,  num2 + '_source.nii')
        img1_source_B = load_4D(source_B)
        # tensor1_source_B = torch.from_numpy(img1_source_B)
        # tensor1_source_B = torch.squeeze(tensor1_source_B, dim=4)
        # img1_source_B = tensor1_source_B.numpy()
        # if img1_source_B.shape == (1, 181, 217, 181):
        #     #img1_source_B = np.pad(img1_source_B, ((0, 0), (5, 6), (3, 4), (5, 6)), 'constant')
        #     img1_source_B = img1_source_B[0:1, 11:171, 13:205, 11:171]

        if self.norm:
            return Norm_Zscore(imgnorm(img3_A)), Norm_Zscore(imgnorm(img3_B))
        else:
            return torch.from_numpy(img3_A).float(), torch.from_numpy(img3_B).float(), \
                   torch.from_numpy(img2_LLL_A).float(),torch.from_numpy(img2_LLL_B).float(), \
                   torch.from_numpy(img2_EM_A).float(), torch.from_numpy(img2_EM_B).float(),\
                   torch.from_numpy(img1_LLL_A).float(), torch.from_numpy(img1_LLL_B).float(), \
                   torch.from_numpy(img1_EM_A).float(), torch.from_numpy(img1_EM_B).float(), \
                   torch.from_numpy(img1_source_A).float(), torch.from_numpy(img1_source_B).float(),



class Predict_dataset(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=True):
        super(Predict_dataset, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list)
        moved_img = load_4D(self.move_list[index])
        fixed_label = load_4D(self.fixed_label_list)
        moved_label = load_4D(self.move_label_list[index])

        if self.norm:
            fixed_img = Norm_Zscore(imgnorm(fixed_img))
            moved_img = Norm_Zscore(imgnorm(moved_img))

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)
        fixed_label = torch.from_numpy(fixed_label)
        moved_label = torch.from_numpy(moved_label)

        if self.norm:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output
        else:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output