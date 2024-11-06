import glob
import os
import sys
import torch.nn as nn
from argparse import ArgumentParser
from torch.autograd import Variable
import numpy as np
import torch
import torch.utils.data as Data
import SimpleITK as sitk
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from WATFunctions import generate_grid, Dataset_epoch, transform_unit_flow_to_flow_cuda, \
    generate_grid_unit
from model_stage import WavletMono_unit_add_lvl1, WavletMono_unit_add_lvl2, WavletMono_unit_add_lvl3, WavletMono_unit_add_lvl4, SpatialTransform_unit, SpatialTransformNearest_unit, smoothloss, \
    neg_Jdet_loss, NCC, antifoldloss

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
saveImgPath = '/xxxxx'

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
                    default='/xxx',
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

def train_lvl1():
    model_name = "WD_OAS_NCC_lvl1_"
    print("Training lvl1...")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
    names = sorted(glob.glob(datapath + 'WD3/LLL3/*.nii'))[0:300]

    grid = generate_grid(imgshape1)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '/xxx/Model'

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

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
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
        for X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2 in training_generator:

            X1_LLL = X1_LLL.to(device).float()
            Y1_LLL = Y1_LLL.to(device).float()

            field1, warped_x1,fixed1,mov1 = model(X1_LLL, Y1_LLL)

            # 1 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC  = loss_multiNCC1

            field_norm = transform_unit_flow_to_flow_cuda(field1.permute(0,2,3,4,1).clone())
            loss_Jacobian = loss_Jdet(field_norm, grid)

            # reg2 - use velocity
            _, _, x, y, z = field1.shape
            field1[:, 0, :, :, :] = field1[:, 0, :, :, :] * z
            field1[:, 1, :, :, :] = field1[:, 1, :, :, :] * y
            field1[:, 2, :, :, :] = field1[:, 2, :, :, :] * x
            loss_regulation1 = loss_smooth(field1)
            loss_regulation = loss_regulation1

            loss_fold = loss_antifold(field1)

            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss = loss_multiNCC + smooth * loss_regulation + antifold*loss_fold #+ 0*loss_Jacobian
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
                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco))
                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0
            if (step % 30000 == 0):
                modelname = model_dir + '/stage/lvl1/' + model_name + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss/lvl1/' + model_name + str(step) + '.npy', lossall)


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
                plt.savefig(model_dir + '/loss/lvl1/' + str(step) + "_lv1.jpg")

            step += 1

            if step > iteration_lvl1:
                break
        print("-------------------------- level 1 epoch pass-------------------------")
    print("level 1 Finish!")

def train_lvl2():
    model_name = "WD_OAS_NCC_lvl2_"
    print("Training lvl2...")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1).to(device)
    model1_path = "/XXX/WD_OAS_NCC_lvl1_30000.pth"
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
    names = sorted(glob.glob(datapath + 'WD3/LLL3/*.nii'))[0:300]

    grid = generate_grid(imgshape2)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model2.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = 'XXX'

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

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
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
        for X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2 in training_generator:

            X1_LLL = X1_LLL.to(device).float()
            Y1_LLL = Y1_LLL.to(device).float()
            X2_LLL = X2_LLL.to(device).float()
            Y2_LLL = Y2_LLL.to(device).float()
            X2_HHH = X2_HHH.to(device).float()
            Y2_HHH = Y2_HHH.to(device).float()

            field1, field2,  warped_x1, warped_x2_lll, warped_x2_hhh, fixed1, fixed2_lll, fixed2_hhh, mov1, mov2\
               = model2(X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH)

            # 3 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC2 = loss_similarity2(warped_x2_lll, fixed2_lll)
            loss_multiNCC22 = loss_similarity2(warped_x2_hhh, fixed2_hhh)

            loss_multiNCC  = 0.5*loss_multiNCC1 + (loss_multiNCC2 + loss_multiNCC22)
            field_norm = transform_unit_flow_to_flow_cuda(field2.permute(0,2,3,4,1).clone())
            loss_Jacobian = loss_Jdet(field_norm, grid)

            _, _, x1, y1, z1 = field1.shape
            field1[:, 0, :, :, :] = field1[:, 0, :, :, :] * z1
            field1[:, 1, :, :, :] = field1[:, 1, :, :, :] * y1
            field1[:, 2, :, :, :] = field1[:, 2, :, :, :] * x1
            _, _, x2, y2, z2 = field2.shape
            field2[:, 0, :, :, :] = field2[:, 0, :, :, :] * z2
            field2[:, 1, :, :, :] = field2[:, 1, :, :, :] * y2
            field2[:, 2, :, :, :] = field2[:, 2, :, :, :] * x2

            loss_regulation1 = loss_smooth(field1)
            loss_regulation2 = loss_smooth(field2)
            loss_regulation =  0.5*loss_regulation1 + loss_regulation2

            loss_fold1 = loss_antifold(field1)
            loss_fold2 = loss_antifold(field2)
            loss_fold = 0.5*loss_fold1+loss_fold2

            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss = loss_multiNCC + smooth * loss_regulation + antifold*loss_fold#loss_Jacobian
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
            if (step % 1000 == 0):
                total = format(float(loss_to) / float(1000), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si) / float(1000), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(1000), '.6f')
                #loss_smo.append(smo)
                Jaco = format(float(loss_Ja) / float(1000), '.9f')
                #loss_Jaco.append(Jaco)
                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco))
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
                plt.savefig(model_dir + '/loss/lvl2/' + str(step) + "_lv2.jpg")

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
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1).to(device)
    model2 = WavletMono_unit_add_lvl2(2, 3, start_channel, range_flow=range_flow, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2,
                                      model1=model1).to(device)
    model2_path = "/XXX/WD_OAS_NCC_lvl2_40000.pth"
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
    names = sorted(glob.glob(datapath + 'WD3/LLL3/*.nii'))[0:300]

    grid = generate_grid(imgshape3)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model3.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = 'xxx'

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

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
                                         shuffle=True, num_workers=4)
    load_model = False
    if load_model is True:
        model_path = "/xxx/WD_oas_NCC_reg_5.pth"
        print("Loading weight: ", model_path)
        model2.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("/xxx/WD_LPBA_OAS_reg_5.npy")
        lossall[:, 0:1000] = temp_lossall[:, 0:1000]

    step = 1
    while step <= iteration_lvl3:
        for X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2 in training_generator:

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

            field1, field2, field3, warped_x1, warped_x2_lll,  warped_x3_lll, warped_x2_hhh, warped_x3_hhh,fixed1, \
            fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, mov1, mov2, mov3,diff_up3, diff_Mhigh_x\
               = model3(X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH)

            # 3 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC2 = loss_similarity2(warped_x2_lll, fixed2_lll)
            loss_multiNCC4 = loss_similarity3(warped_x3_lll, fixed3_lll)
            loss_multiNCC22 = loss_similarity2(warped_x2_hhh, fixed2_hhh)
            loss_multiNCC44 = loss_similarity3(warped_x3_hhh, fixed3_hhh)

            loss_multiNCC  = 0.25*loss_multiNCC1 + 0.5*(loss_multiNCC2+loss_multiNCC22) + (loss_multiNCC4+loss_multiNCC44)
            field_norm = transform_unit_flow_to_flow_cuda(field3.permute(0,2,3,4,1).clone())
            loss_Jacobian = loss_Jdet(field_norm, grid)

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
            loss_regulation1 = loss_smooth(field1)
            loss_regulation2 = loss_smooth(field2)
            loss_regulation3 = loss_smooth(field3)
            loss_regulation = 0.25*loss_regulation1 + 0.5*loss_regulation2 + loss_regulation3

            loss_fold1 = loss_antifold(field1)
            loss_fold2 = loss_antifold(field2)
            loss_fold3 = loss_antifold(field3)
            loss_fold = 0.25 * loss_fold1 + 0.5*loss_fold2 + loss_fold3

            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss = loss_multiNCC + smooth * loss_regulation + antifold*loss_fold#loss_Jacobian
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
            if (step % 1000 == 0):
                total = format(float(loss_to) / float(1000), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si) / float(1000), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(1000), '.6f')
                #loss_smo.append(smo)
                Jaco = format(float(loss_Ja) / float(1000), '.9f')
                #loss_Jaco.append(Jaco)
                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco))
                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0
            if (step % 30000 == 0):
                modelname = model_dir + '/stage/lvl3/' + model_name + str(step) + '.pth'
                torch.save(model3.state_dict(), modelname)
                np.save(model_dir + '/loss/lvl3/' + model_name + str(step) + '.npy', lossall)


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
                plt.savefig(model_dir + '/loss/lvl3/' + str(step) + "_lv3.jpg")

            # if step == freeze_step:
            #     model3.unfreeze_modellvl2()
            step += 1

            if step > iteration_lvl3:
                break
        print("-------------------------- level 3 epoch pass-------------------------")
    print("level 3 Finish!")

def train_lvl4():
    model_name = "WD_OAS_NCC_lvl4_"
    print("Training lvl4...")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model1 = WavletMono_unit_add_lvl1(2, 3, start_channel, is_train=True, imgshape1=imgshape1).to(device)
    model2 = WavletMono_unit_add_lvl2(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2,
                                      model1=model1).to(device)
    model3 = WavletMono_unit_add_lvl3(2, 3, start_channel, is_train=True, imgshape1=imgshape1, imgshape2=imgshape2, imgshape3=imgshape3,
                                      model2=model2).to(device)

    model3_path = "/XXX/WD_LPBA_NCC_lvl3_60000.pth"
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
    names = sorted(glob.glob(datapath + 'WD3/LLL3/*.nii'))[0:300]

    grid = generate_grid(imgshape4)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model3.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = ''

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl4+1))
    loss_total = []
    loss_smo = []
    loss_sim = []
    loss_Jaco = []
    loss_to = 0
    loss_sm = 0
    loss_si = 0
    loss_Ja = 0

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
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
        for X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2 in training_generator:

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

            field1, field2, field3, field4, warped_x1, warped_x2_lll, warped_x3_lll, warped_x2_hhh, warped_x3_hhh, \
            warped_source1, fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, fixed4_source2, mov1, mov2, mov3, source1, diff_up4, diff_Mhigh_x \
                = model4(X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1,
                        source2)

            # 3 level deep supervision NCC
            loss_multiNCC1 = loss_similarity1(warped_x1, fixed1)
            loss_multiNCC2 = loss_similarity2(warped_x2_lll, fixed2_lll)
            loss_multiNCC4 = loss_similarity3(warped_x3_lll, fixed3_lll)
            loss_multiNCC6 = loss_similarity4(warped_source1, fixed4_source2)
            loss_multiNCC22 = loss_similarity2(warped_x2_hhh, fixed2_hhh)
            loss_multiNCC44 = loss_similarity3(warped_x3_hhh, fixed3_hhh)

            loss_multiNCC  = 0.125*loss_multiNCC1 + 0.25*(loss_multiNCC2 + loss_multiNCC22) + 0.5*(loss_multiNCC4 + loss_multiNCC44) + loss_multiNCC6
            field_norm = transform_unit_flow_to_flow_cuda(field4.permute(0,2,3,4,1).clone())
            loss_Jacobian = loss_Jdet(field_norm, grid)

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
            loss_regulation1 = loss_smooth(field1)
            loss_regulation2 = loss_smooth(field2)
            loss_regulation3 = loss_smooth(field3)
            loss_regulation4 = loss_smooth(field4)
            loss_regulation = 0.125*loss_regulation1 + 0.25*loss_regulation2 + 0.5*loss_regulation3 + loss_regulation4

            loss_fold1 = loss_antifold(field1)
            loss_fold2 = loss_antifold(field2)
            loss_fold3 = loss_antifold(field3)
            loss_fold4 = loss_antifold(field4)
            loss_fold = 0.125 * loss_fold1 + 0.25*loss_fold2 + 0.5*loss_fold3 + loss_fold4

            #loss = loss_multiNCC + antifold*loss_Jacobian + smooth*loss_regulation
            loss = loss_multiNCC + smooth * loss_regulation + antifold*loss_fold#loss_Jacobian
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            if (step % 20000 != 0):
                del X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2, field1, field2, field3, field4, warped_x1, warped_x2_lll, warped_x3_lll, warped_x2_hhh, warped_x3_hhh, warped_source1, fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, fixed4_source2, mov1, mov2, mov3
                torch.cuda.empty_cache()

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
            if (step % 1000 == 0):
                total = format(float(loss_to) / float(1000), '.6f')
                #loss_total.append(total)
                sim = format(float(loss_si) / float(1000), '.6f')
                #loss_sim.append(sim)
                smo = format(float(loss_sm) / float(1000), '.6f')
                #loss_smo.append(smo)
                Jaco = format(float(loss_Ja) / float(1000), '.9f')
                #loss_Jaco.append(Jaco)
                print('step->'+str(step), 'total:'+str(total), 'sim_NCC:'+str(sim), 'smooth:'+str(smo), 'Jacob:'+str(Jaco))
                loss_to = 0
                loss_sm = 0
                loss_si = 0
                loss_Ja = 0
            if (step % 20000 == 0):
                modelname = model_dir + '/stage/lvl4/' + model_name + str(step) + '.pth'
                torch.save(model2.state_dict(), modelname)
                np.save(model_dir + '/loss/lvl4/' + model_name + str(step) + '.npy', lossall)

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
                plt.savefig(model_dir + '/loss/lvl4/' + str(step) + "_lv4.jpg")
                del X1_LLL, Y1_LLL, X2_LLL, Y2_LLL, X2_HHH, Y2_HHH, X3_LLL, Y3_LLL, X3_HHH, Y3_HHH, source1, source2, field1, field2, field3, field4, warped_x1, warped_x2_lll, warped_x3_lll, warped_x2_hhh, warped_x3_hhh, warped_source1, fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, fixed4_source2, mov1, mov2, mov3
                torch.cuda.empty_cache()

            # if step == freeze_step:
            #     model4.unfreeze_modellvl3()

            step += 1

            if step > iteration_lvl4:
                break
        print("-------------------------- level 4 epoch pass-------------------------")
    print("level 4 Finish!")

range_flow = 7
imgshape4 = (160, 192, 160)
imgshape3 = (80, 96, 80)
imgshape2 = (40, 48, 40)
imgshape1 = (20, 24, 20)

train_lvl1()
train_lvl2()
train_lvl3()
train_lvl4()

