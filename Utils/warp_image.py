#coding=utf-8
import torch
from torch.autograd import Variable



def grid_generator(field):
    """
    generate the new coordinate
    :param field: tensor, 3*D*W*H
    :return:the new coordinate of x, y, z
    """

    field_x = field[ 0,:, :, :].clone()
    field_y = field[ 1,:, :, :].clone()
    field_z = field[ 2,:, :, :].clone()

    sz = field_x.size()
    for i in range(sz[0]):
        field_x[i, :, :] = field_x[i, :, :] + i
    for i in range(sz[1]):
        field_y[:, i, :] = field_y[:, i, :] + i
    for i in range(sz[2]):
        field_z[:, :, i] = field_z[:, :, i] + i
    return field_x, field_y, field_z

def warp_image_core(field, moving):
    """
    the core function of warp image with tri-linear interpolation
    :param field: tensor, 3*D*W*H
    :param moving:tensor, D*W*H
    :return: warped image, tensor, D*W*H
    """

    moving = moving.cuda()
    field = field.cuda()


    vx, vy, vz = grid_generator(field)

    lx = Variable(torch.floor(vx).type(torch.cuda.LongTensor))#floor:返回不大于输入参数的最大整数
    ly = Variable(torch.floor(vy).type(torch.cuda.LongTensor))
    lz = Variable(torch.floor(vz).type(torch.cuda.LongTensor))
    lx[lx < 0] = 0
    ly[ly < 0] = 0
    lz[lz < 0] = 0

    ux = lx + 1
    uy = ly + 1
    uz = lz + 1

    size = moving.data.size()
    ux[ux >= size[0]] = size[0] - 1
    uy[uy >= size[1]] = size[1] - 1
    uz[uz >= size[2]] = size[2] - 1


    lx[lx >= size[0]] = size[0] - 1
    ly[ly >= size[1]] = size[1] - 1
    lz[lz >= size[2]] = size[2] - 1


    uuu = moving[ux[:], uy[:], uz[:]]
    ulu = moving[ux[:], ly[:], uz[:]]
    llu = moving[lx[:], ly[:], uz[:]]
    luu = moving[lx[:], uy[:], uz[:]]
    uul = moving[ux[:], uy[:], lz[:]]
    ull = moving[ux[:], ly[:], lz[:]]
    lll = moving[lx[:], ly[:], lz[:]]
    lul = moving[lx[:], uy[:], lz[:]]

    lx = Variable(lx.type(torch.cuda.FloatTensor))
    ly = Variable(ly.type(torch.cuda.FloatTensor))
    lz = Variable(lz.type(torch.cuda.FloatTensor))
    ux = Variable(ux.type(torch.cuda.FloatTensor))
    uy = Variable(uy.type(torch.cuda.FloatTensor))
    uz = Variable(uz.type(torch.cuda.FloatTensor))

    moving_warped = (vx - lx) * (vy - ly) * (vz - lz) * uuu + \
                    (vx - lx) * (uy - vy) * (vz - lz) * ulu + \
                    (ux - vx) * (uy - vy) * (vz - lz) * llu + \
                    (ux - vx) * (vy - ly) * (vz - lz) * luu + \
                    (vx - lx) * (vy - ly) * (uz - vz) * uul + \
                    (vx - lx) * (uy - vy) * (uz - vz) * ull + \
                    (ux - vx) * (uy - vy) * (uz - vz) * lll + \
                    (ux - vx) * (vy - ly) * (uz - vz) * lul


    moving_warped[lz == uz] = moving[lz == uz]
    moving_warped[ly == uy] = moving[ly == uy]
    moving_warped[lx == ux] = moving[lx == ux]

    return moving_warped





def warp_image_nn_core(field, moving):
    """
    the core function of warp image with nearest neighbor interpolation
    :param field: tensor, 3*D*W*H
    :param moving:tensor, D*W*H
    :return: warped image, tensor, D*W*H
    """

    mov = moving.cuda()
    field = field.cuda()

    vx, vy, vz = grid_generator(field)

    rx = Variable(torch.round(vx).type(torch.cuda.LongTensor))
    ry = Variable(torch.round(vy).type(torch.cuda.LongTensor))
    rz = Variable(torch.round(vz).type(torch.cuda.LongTensor))
    size = mov.data.size()
    rx[rx < 0] = 0
    ry[ry < 0] = 0
    rz[rz < 0] = 0
    rx[rx >= size[0]] = size[0] - 1
    ry[ry >= size[1]] = size[1] - 1
    rz[rz >= size[2]] = size[2] - 1

    moving_warped = mov[rx[:], ry[:], rz[:]]

    return moving_warped


def warp_image_batch(field_batch, moving_batch):
    """
    warp moving image with tri-linear interpolation for a batch
    :param field_batch: tensor, five dimensions: B*3*D*W*H
    :param moving_batch: moving image, tensor, five dimensions: B*C*D*W*H
    :return: moving_batch_warped, tensor, five dimensions: B*C*D*W*H
    """

    moving_batch_warped = Variable(torch.empty_like(moving_batch))
    size = moving_batch.size()

    for batch_idx in range(size[0]):
        for channel_idx in range(size[1]):
            moving = moving_batch[batch_idx, channel_idx, :, :, :]
            field = field_batch[batch_idx, :, :, :, :]
            moving_batch_warped[batch_idx, channel_idx, :, :, :] = warp_image_core(field, moving)

    return moving_batch_warped


def warp_image_nn_batch(field_batch, moving_batch):
    """
    warp moving image with nearest neighbor interpolation
    :param field_batch, tensor, five dimensions: B*3*D*W*H
    :param moving_batch: moving image, tensor, five dimensions: B*C*D*W*H
    :return: moving_batch_warped, tensor, five dimensions: B*C*D*W*H
    """

    moving_batch_warped = Variable(torch.empty_like(moving_batch))
    size = moving_batch.size()

    for batch_idx in range(size[0]):
        for channel_idx in range(size[1]):
            moving = moving_batch[batch_idx, channel_idx, :, :, :]
            field = field_batch[batch_idx, :, :, :, :]
            moving_batch_warped[batch_idx, channel_idx, :, :, :] = warp_image_nn_core(field, moving)

    return moving_batch_warped

