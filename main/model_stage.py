import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from WATFunctions import generate_grid_unit

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class WavletMono_unit_add_lvl1(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, range_flow=0.4, is_train=True, imgshape1=(20, 24, 20)):
        super(WavletMono_unit_add_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.range_flow = range_flow

        self.is_train = is_train

        self.imgshape1 = imgshape1

        self.grid_1 = generate_grid_unit(self.imgshape1)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).to(device).float()

        #self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().to(device)

        bias_opt = False

        self.imgInput1 = nn.Conv3d(self.in_channel, self.start_channel * 4, 3, stride=1, padding=1, bias=bias_opt)

        self.resblock3 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv3 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock4 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        # self.down_conv4 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
        #                             bias=bias_opt)
        self.resblock5 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up1 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        # self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        # self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        # self.resblock6 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        # self.up2 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
        #                               padding=0, output_padding=0, bias=bias_opt)
        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=1, padding=1,
                                    bias=bias_opt)


    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            Block_down(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
        )
        return layer


    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x1, y1):
        #level 1
        cat_input_lvl1 = torch.cat((x1, y1), 1)

        fixed1 = cat_input_lvl1[:, 1:2, :, :, :]#template
        mov1 = cat_input_lvl1[:, 0:1, :, :, :]  #moving

        fea_input1=self.imgInput1(cat_input_lvl1)
        fea_block3=self.resblock3(fea_input1)
        fea_down3= self.down_conv3(fea_block3)
        fea_block4 = self.resblock3(fea_down3)
        #fea_down4 = self.down_conv4(fea_block4)
        fea_block5 = self.resblock5(fea_block4)
        fea_up1 = self.up1(fea_block5)
        # fea_block6 = self.resblock5(fea_up1)
        # fea_up2 = self.up2(fea_block6)
        # #fea_up2 = fea_up2[:,:,1:19,:,:]
        skip1 = self.skip(fea_block3)
        skip2 = self.skip(skip1)
        field1=self.output_lvl1(torch.cat([skip2,fea_up1],dim=1)) * self.range_flow
        #output_disp_e0 = self.diff_transform(fea_output, self.grid_1) #deformation field
        warped_x1 = self.transform(x1, field1.permute(0, 2, 3, 4, 1), self.grid_1)

        if self.is_train is True:
            return field1,  warped_x1, fixed1,  mov1
        else:
            return field1



class WavletMono_unit_add_lvl2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, range_flow=0.4, is_train=True, imgshape1=(20, 24, 20),imgshape2=(40, 48, 40),model1=None):
        super(WavletMono_unit_add_lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.range_flow = range_flow

        self.is_train = is_train

        self.imgshape1 = imgshape1
        self.imgshape2 = imgshape2
        self.model1 = model1

        self.grid_1 = generate_grid_unit(self.imgshape1)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).to(device).float()

        self.grid_2 = generate_grid_unit(self.imgshape2)
        self.grid_2 = torch.from_numpy(np.reshape(self.grid_2, (1,) + self.grid_2.shape)).to(device).float()

        #self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().to(device)

        bias_opt = False

        self.imgInput1 = nn.Conv3d(self.in_channel, self.start_channel * 4, 3, stride=1, padding=1, bias=bias_opt)
        self.imgInput2 = nn.Conv3d(self.in_channel + 3, self.start_channel * 2, 3, stride=1, padding=1, bias=bias_opt)

        self.resblock2 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv2 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock3 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv3 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock4 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv4 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock5 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up1 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        # self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.resblock6 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up2 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        self.resblock7 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up3 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=1, padding=1,
                              bias=bias_opt)

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            Block_down(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
        )
        return layer


    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x1, y1, x2_lll, y2_lll, x2_hhh, y2_hhh):
        #level 1
        field1, warped_x1, fixed1, mov1 = self.model1(x1, y1)

        #Level 2
        field1_up = self.up_tri(field1)
        warped_x2_lll_input = self.transform(x2_lll, field1_up.permute(0, 2, 3, 4, 1), self.grid_2)

        cat_input_lvl2_lll = torch.cat((warped_x2_lll_input, y2_lll, field1_up), 1)
        fixed2_lll = cat_input_lvl2_lll[:, 1:2, :, :, :]  # template
        mov2 = x2_lll  # moving
        fea_input2_lll = self.imgInput2(cat_input_lvl2_lll)

        warped_x2_hhh_input = self.transform(x2_hhh, field1_up.permute(0, 2, 3, 4, 1), self.grid_2)
        cat_input_lvl2_hhh = torch.cat((warped_x2_hhh_input, y2_hhh), 1)
        fixed2_hhh = cat_input_lvl2_hhh[:, 1:2, :, :, :]  # template
        fea_input2_hhh = self.imgInput2(cat_input_lvl2_hhh)

        fea_block2 = self.resblock2(torch.cat([fea_input2_lll,fea_input2_hhh],dim=1))
        fea_down2 = self.down_conv2(fea_block2)
        fea_block32 = self.resblock3(fea_down2)
        fea_down32 = self.down_conv3(fea_block32)
        fea_block42 = self.resblock4(fea_down32)
        #fea_down42 = self.down_conv4(fea_block42)
        fea_block52 = self.resblock5(fea_block42)
        fea_up12 = self.up1(fea_block52)
        fea_block62 = self.resblock6(fea_up12)
        fea_up22 = self.up2(fea_block62)
        #fea_up22 = fea_up22[:, :, 1:19, :, :]
        # fea_block7 = self.resblock7(fea_up22)
        # fea_up3 = self.up3(fea_block7)
        skip1 = self.skip(fea_block2)
        # skip2 = self.skip(skip1)
        field2 = self.output_lvl1(torch.cat([skip1, fea_up22], dim=1)) * self.range_flow
        ##addition
        # field2 = field2 + field1_up

        ##composition
        field1_up_warp1 = self.transform(field1_up[:, 0:1, :, :, :], field2.permute(0, 2, 3, 4, 1), self.grid_2)
        field1_up_warp2 = self.transform(field1_up[:, 1:2, :, :, :], field2.permute(0, 2, 3, 4, 1), self.grid_2)
        field1_up_warp3 = self.transform(field1_up[:, 2:3, :, :, :], field2.permute(0, 2, 3, 4, 1), self.grid_2)
        field1_up_warp = torch.cat((field1_up_warp1, field1_up_warp2, field1_up_warp3), 1)
        field2 = field2 + field1_up_warp

        warped_x2_lll = self.transform(x2_lll, field2.permute(0, 2, 3, 4, 1), self.grid_2)
        warped_x2_hhh = self.transform(x2_hhh, field2.permute(0, 2, 3, 4, 1), self.grid_2)

        if self.is_train is True:
            return field1, field2,  warped_x1, warped_x2_lll, warped_x2_hhh, fixed1, fixed2_lll, fixed2_hhh, mov1, mov2
        else:
            return field2

class WavletMono_unit_add_lvl3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, range_flow=0.4, is_train=True, imgshape1=(20, 24, 20),imgshape2=(40, 48, 40),imgshape3=(80, 96, 80), model2 = None):
        super(WavletMono_unit_add_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.range_flow = range_flow

        self.is_train = is_train
        self.model2 = model2

        self.imgshape1 = imgshape1
        self.imgshape2 = imgshape2
        self.imgshape3 = imgshape3

        self.grid_1 = generate_grid_unit(self.imgshape1)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).to(device).float()

        self.grid_2 = generate_grid_unit(self.imgshape2)
        self.grid_2 = torch.from_numpy(np.reshape(self.grid_2, (1,) + self.grid_2.shape)).to(device).float()

        self.grid_3 = generate_grid_unit(self.imgshape3)
        self.grid_3 = torch.from_numpy(np.reshape(self.grid_3, (1,) + self.grid_3.shape)).to(device).float()

        #self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().to(device)

        bias_opt = False

        self.imgInput1 = nn.Conv3d(self.in_channel, self.start_channel * 4, 3, stride=1, padding=1, bias=bias_opt)
        self.imgInput2 = nn.Conv3d(self.in_channel, self.start_channel * 2, 3, stride=1, padding=1, bias=bias_opt)
        self.imgInput3_6 = nn.Conv3d(self.in_channel + 3, self.start_channel * 4, 3, stride=1, padding=1, bias=bias_opt)
        self.imgInput3_2 = nn.Conv3d(self.in_channel + 3, self.start_channel * 4, 3, stride=1, padding=1, bias=bias_opt)
        self.imgInput3_cat = nn.Conv3d(self.start_channel * 4, self.start_channel * 2, 3, stride=1, padding=1,
                                       bias=bias_opt)

        self.resblock1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv1 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock2 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv2 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock3 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv3 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock4 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv4 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock5 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up1 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        # self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.resblock6 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up2 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        self.resblock7 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up3 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        self.resblock8 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up4 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=1, padding=1,
                              bias=bias_opt)

    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            Block_down(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
        )
        return layer


    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x1, y1, x2_lll, y2_lll, x2_hhh, y2_hhh,  x3_lll, y3_lll, x3_hhh, y3_hhh):
        #level 1
        #Level 2
        field1, field2, warped_x1, warped_x2_lll, warped_x2_hhh, fixed1, fixed2_lll, fixed2_hhh, mov1, mov2 \
            = self.model2(x1, y1, x2_lll, y2_lll, x2_hhh, y2_hhh)

        # Level 3
        field2_up = self.up_tri(field2)
        warped_x3_lll_input = self.transform(x3_lll, field2_up.permute(0, 2, 3, 4, 1), self.grid_3)

        field1_up = self.up_tri(field1)
        diff = torch.abs(warped_x3_lll_input) - torch.abs(y3_lll)
        ###diff norm(0-1)
        d_min = diff.min()
        if d_min < 0:
            diff = diff + torch.abs(d_min)
            d_min = diff.min()
        d_max = diff.max()
        dst = d_max - d_min
        diff_norm = (diff - d_min).true_divide(dst)

        diff_Mhigh = ((1 - 0.5) * diff_norm + 0.5) * warped_x3_lll_input
        diff_Fhigh = ((1 - 0.5) * diff_norm + 0.5) * y3_lll
        cat_input_lvl3_lll = torch.cat((diff_Mhigh, diff_Fhigh, field2_up), 1)

        # cat_input_lvl3_lll = torch.cat(
        #     (diff_Mhigh_x, diff_Mhigh_y, diff_Mhigh_z, diff_Fhigh_x, diff_Fhigh_y, diff_Fhigh_z), 1)
        fixed3_lll = y3_lll  # template
        mov3 = x3_lll  # moving
        fea_input3_lllx = self.imgInput3_6(cat_input_lvl3_lll)
        fea_input3_lll = self.imgInput3_cat(fea_input3_lllx)

        warped_x3_hhh_input = self.transform(x3_hhh, field2_up.permute(0, 2, 3, 4, 1), self.grid_3)
        cat_input_lvl3_hhh = torch.cat((warped_x3_hhh_input, y3_hhh, field2_up), 1)
        fixed3_hhh = y3_hhh  # cat_input_lvl3_hhh[:, 1:2, :, :, :]  # template
        fea_input3_hhhx = self.imgInput3_2(cat_input_lvl3_hhh)
        fea_input3_hhh = self.imgInput3_cat(fea_input3_hhhx)

        fea_block1 = self.resblock1(torch.cat([fea_input3_lll, fea_input3_hhh], dim=1))
        fea_down1 = self.down_conv1(fea_block1)
        fea_block23 = self.resblock2(fea_down1)
        fea_down23 = self.down_conv2(fea_block23)
        fea_block33 = self.resblock3(fea_down23)
        fea_down33 = self.down_conv3(fea_block33)
        fea_block43 = self.resblock4(fea_down33)
        #fea_down43 = self.down_conv4(fea_block43)
        fea_block53 = self.resblock5(fea_block43)
        fea_up13 = self.up1(fea_block53)
        fea_block63 = self.resblock6(fea_up13)
        fea_up23 = self.up2(fea_block63)
        #fea_up23 = fea_up23[:, :, 1:19, :, :]
        fea_block73 = self.resblock7(fea_up23)
        fea_up33 = self.up3(fea_block73)
        # fea_block8 = self.resblock8(fea_up33)
        # fea_up4 = self.up4(fea_block8)

        skip1 = self.skip(fea_block1)
        # skip2 = self.skip(skip1)
        # skip3 = self.skip(skip2)
        field3 = self.output_lvl1(torch.cat([skip1, fea_u23], dim=1)) * self.range_flow
        ##addition
        # field3 = field3 + field2_up

        ##composition
        field2_up_warp1 = self.transform(field2_up[:, 0:1, :, :, :], field3.permute(0, 2, 3, 4, 1), self.grid_3)
        field2_up_warp2 = self.transform(field2_up[:, 1:2, :, :, :], field3.permute(0, 2, 3, 4, 1), self.grid_3)
        field2_up_warp3 = self.transform(field2_up[:, 2:3, :, :, :], field3.permute(0, 2, 3, 4, 1), self.grid_3)
        field2_up_warp = torch.cat((field2_up_warp1, field2_up_warp2, field2_up_warp3), 1)
        field3 = field3 + field2_up_warp

        warped_x3_lll = self.transform(x3_lll, field3.permute(0, 2, 3, 4, 1), self.grid_3)
        warped_x3_hhh = self.transform(x3_hhh, field3.permute(0, 2, 3, 4, 1), self.grid_3)


        if self.is_train is True:
            return field1, field2, field3, warped_x1, warped_x2_lll,  warped_x3_lll, warped_x2_hhh, warped_x3_hhh,\
                   fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, mov1, mov2, mov3, diff_up3, diff_Mhigh_x
        else:
            return field3

class WavletMono_unit_add_lvl4(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, range_flow = 0.4, is_train=True, imgshape1=(20, 24, 20),imgshape2=(40, 48, 40),imgshape3=(80, 96, 80),imgshape4=(160, 192, 160), model3=None):
        super(WavletMono_unit_add_lvl4, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.range_flow = range_flow

        self.is_train = is_train
        self.model3 = model3

        self.imgshape1 = imgshape1
        self.imgshape2 = imgshape2
        self.imgshape3 = imgshape3
        self.imgshape4 = imgshape4

        self.grid_1 = generate_grid_unit(self.imgshape1)
        self.grid_1 = torch.from_numpy(np.reshape(self.grid_1, (1,) + self.grid_1.shape)).to(device).float()

        self.grid_2 = generate_grid_unit(self.imgshape2)
        self.grid_2 = torch.from_numpy(np.reshape(self.grid_2, (1,) + self.grid_2.shape)).to(device).float()

        self.grid_3 = generate_grid_unit(self.imgshape3)
        self.grid_3 = torch.from_numpy(np.reshape(self.grid_3, (1,) + self.grid_3.shape)).to(device).float()

        self.grid_4 = generate_grid_unit(self.imgshape4)
        self.grid_4 = torch.from_numpy(np.reshape(self.grid_4, (1,) + self.grid_4.shape)).to(device).float()

        #self.diff_transform = DiffeomorphicTransform_unit(time_step=7).cuda()
        self.transform = SpatialTransform_unit().to(device)

        bias_opt = False

        self.imgInput1 = nn.Conv3d(self.in_channel, self.start_channel * 4, 3, stride=1, padding=1, bias=bias_opt)
        self.imgInput2 = nn.Conv3d(self.in_channel + 3, self.start_channel * 2, 3, stride=1, padding=1, bias=bias_opt)
        self.imgInput3 = nn.Conv3d(self.in_channel + 3, self.start_channel * 2, 3, stride=1, padding=1, bias=bias_opt)
        self.imgInput4_6 = nn.Conv3d(self.in_channel + 3, self.start_channel * 4, 3, stride=1, padding=1, bias=bias_opt)

        self.resblock0 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv0 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv1 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock2 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv2 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock3 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv3 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock4 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.down_conv4 = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                    bias=bias_opt)
        self.resblock5 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up1 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        # self.down_avg = nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="trilinear")
        self.resblock6 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up2 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        self.resblock7 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up3 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        self.resblock8 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up4 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        self.resblock9 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)
        self.up5 = nn.ConvTranspose3d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                      padding=0, output_padding=0, bias=bias_opt)
        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip = nn.Conv3d(self.start_channel * 4, self.start_channel * 4, 3, stride=1, padding=1,
                              bias=bias_opt)

    def unfreeze_modellvl3(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model3.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.Sequential(
            Block_down(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
        )
        return layer


    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, int(in_channels/2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv3d(int(in_channels/2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x1, y1, x2_lll, y2_lll, x2_hhh, y2_hhh,  x3_lll, y3_lll, x3_hhh, y3_hhh, source1, source2):
        #level 1
        #Level 2
        # Level 3
        field1, field2, field3, warped_x1, warped_x2_lll, warped_x3_lll, warped_x2_hhh, warped_x3_hhh, fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, fixed3_hhh, mov1, mov2, mov3, _,_\
            = self.model3(x1, y1, x2_lll, y2_lll, x2_hhh, y2_hhh,  x3_lll, y3_lll, x3_hhh, y3_hhh)
        # Level 4(source)
        field3_up = self.up_tri(field3)
        warped_source1_input = self.transform(source1, field3_up.permute(0, 2, 3, 4, 1), self.grid_4)

        field2_up = self.up_tri(field2)
        diff = torch.abs(warped_source1_input) - torch.abs(source2)
        diff = torch.abs(diff)
        # print(diff.shape)
        ###diff norm(0-1)
        d_min = diff.min()
        if d_min < 0:
            diff = diff + torch.abs(d_min)
            d_min = diff.min()
        d_max = diff.max()
        dst = d_max - d_min
        diff_norm = (diff - d_min).true_divide(dst)
        diff_up = self.up_tri(diff_norm)
        diff_up4 = torch.squeeze(diff_up)

        diff_Mhigh = ((1 - 0.5) * diff_norm + 0.5) * warped_source1_input
        diff_Fhigh = ((1 - 0.5) * diff_norm + 0.5) * source2
        cat_input_lvl4_source = torch.cat((diff_Mhigh, diff_Fhigh, field2_up), 1)

        fixed4_source2 = source2  # template
        fea_input4_source = self.imgInput4_6(cat_input_lvl4_source)

        fea_block0 = self.resblock0(fea_input4_source)
        fea_down0 = self.down_conv0(fea_block0)
        fea_block14 = self.resblock1(fea_down0)
        fea_down14 = self.down_conv1(fea_block14)
        fea_block24 = self.resblock2(fea_down14)
        fea_down24 = self.down_conv2(fea_block24)
        fea_block34 = self.resblock3(fea_down24)
        fea_down34 = self.down_conv3(fea_block34)
        fea_block44 = self.resblock4(fea_down34)
        #fea_down44 = self.down_conv4(fea_block44)
        fea_block54 = self.resblock5(fea_block44)
        fea_up14 = self.up1(fea_block54)
        fea_block64 = self.resblock6(fea_up14)
        fea_up24 = self.up2(fea_block64)
        #fea_up24 = fea_up24[:, :, 1:19, :, :]
        fea_block74 = self.resblock7(fea_up24)
        fea_up34 = self.up3(fea_block74)
        fea_block84 = self.resblock8(fea_up34)
        fea_up44 = self.up4(fea_block84)
        # fea_block9 = self.resblock9(fea_up44)
        # fea_up5 = self.up5(fea_block9)

        skip1 = self.skip(fea_block0)
        # skip2 = self.skip(skip1)
        # skip3 = self.skip(skip2)
        # skip4 = self.skip(skip3)
        field4 = self.output_lvl1(torch.cat([skip1, fea_up44], dim=1)) * self.range_flow
        ## addition
        # field4 = field4 + field3_up

        ##composition
        field3_up_warp1 = self.transform(field3_up[:, 0:1, :, :, :], field4.permute(0, 2, 3, 4, 1), self.grid_4)
        field3_up_warp2 = self.transform(field3_up[:, 1:2, :, :, :], field4.permute(0, 2, 3, 4, 1), self.grid_4)
        field3_up_warp3 = self.transform(field3_up[:, 2:3, :, :, :], field4.permute(0, 2, 3, 4, 1), self.grid_4)
        field3_up_warp = torch.cat((field3_up_warp1, field3_up_warp2, field3_up_warp3), 1)
        field4 = field4 + field3_up_warp

        warped_source1 = self.transform(source1, field4.permute(0, 2, 3, 4, 1), self.grid_4)


        if self.is_train is True:
            return field1, field2, field3, field4,  warped_x1, warped_x2_lll,  warped_x3_lll, \
                   warped_x2_hhh, warped_x3_hhh, warped_source1, fixed1, fixed2_lll, fixed2_hhh, fixed3_lll, \
                   fixed3_hhh, fixed4_source2, mov1, mov2, mov3, source1, diff_up4, diff_Mhigh_x
        else:
            return field4




class Block_down(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False):
        super(Block_down, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=bias)
            )
    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(out, negative_slope=0.2))


        out += shortcut
        return out


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


class SpatialTransformNearest_unit(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border", align_corners=True)
        return flow


class DiffeomorphicTransform_unit(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform_unit, self).__init__()
        self.time_step = time_step

    def forward(self, velocity, sample_grid):
        flow = velocity/(2.0**self.time_step)
        for _ in range(self.time_step):
            grid = sample_grid + flow.permute(0,2,3,4,1)
            flow = flow + F.grid_sample(flow, grid, mode='bilinear', padding_mode="border", align_corners=True)
        return flow


def smoothloss(y_pred):
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :])
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :])
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1])
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0


def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet

def antifoldloss(y_pred):
    dy = y_pred[:, :, :-1, :, :] - y_pred[:, :, 1:, :, :]-1
    dx = y_pred[:, :, :, :-1, :] - y_pred[:, :, :, 1:, :]-1
    dz = y_pred[:, :, :, :, :-1] - y_pred[:, :, :, :, 1:]-1

    dy = F.relu(dy) * torch.abs(dy*dy)
    dx = F.relu(dx) * torch.abs(dx*dx)
    dz = F.relu(dz) * torch.abs(dz*dz)
    return (torch.mean(dx)+torch.mean(dy)+torch.mean(dz))/3.0


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)

def mse_loss(input, target):
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    diff = y_true_f-y_pred_f
    mse = torch.mul(diff,diff).mean()
    return mse


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=3, eps=1e-8):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)




class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def  __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i*2)))

    def forward(self, I, J):
        total_NCC = []

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC/(2**i))

            I = nn.functional.avg_pool3d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool3d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)

def diceLoss(y_true, y_pred):
    top = 2 * (y_true * y_pred, [1, 2, 3]).sum()
    bottom = torch.max((y_true + y_pred, [1, 2, 3]).sum(), 50)
    dice = torch.mean(top / bottom)
    return -dice