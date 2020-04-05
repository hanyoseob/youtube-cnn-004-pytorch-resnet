"""
-------------------------------------
denoising
-------------------------------------
python  train.py \
        --mode train \
        --network resnet \
        --learning_type residual \
        --task denoising \
        --opts random 30.0

-------------------------------------
inpainting
-------------------------------------
python  train.py \
        --mode train \
        --network resnet \
        --learning_type residual \
        --task inpainting \
        --opts uniform 0.5

-------------------------------------
python  train.py \
        --mode train \
        --network resnet \
        --learning_type residual \
        --task inpainting \
        --opts random 0.5

-------------------------------------
super_resolution
-------------------------------------
python  train.py \
        --mode train \
        --network resnet \
        --learning_type residual \
        --task super_resolution \
        --opts bilinear 4.0
"""

## 라이브러리 추가하기
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms

## Parser 생성하기
parser = argparse.ArgumentParser(description="Regression Tasks such as inpainting, denoising, and super_resolution",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="test", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=8, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=300, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets/BSR/BSDS500/data/images", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint/srresnet/super_resolution", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log/srresnet/super_resolution", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result/srresnet/super_resolution", type=str, dest="result_dir")

parser.add_argument("--task", default="super_resolution", choices=["inpainting", "denoising", "super_resolution"], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['bilinear', 4.0, 0], dest='opts')

parser.add_argument("--ny", default=320, type=int, dest="ny")
parser.add_argument("--nx", default=480, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--network", default="srresnet", choices=["unet", "hourglass", "resnet", "srresnet"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

## 트레이닝 파라메터 설정하기
mode = args.mode
train_continue = args.train_continue

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("mode: %s" % mode)

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)

print("task: %s" % task)
print("opts: %s" % opts)

print("network: %s" % network)
print("learning type: %s" % learning_type)

print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)

print("device: %s" % device)

## 디렉토리 생성하기
result_dir_train = os.path.join(result_dir, 'train')
result_dir_val = os.path.join(result_dir, 'val')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    # os.makedirs(os.path.join(result_dir_train, 'numpy'))

    os.makedirs(os.path.join(result_dir_val, 'png'))
    # os.makedirs(os.path.join(result_dir_val, 'numpy'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))

## 네트워크 학습하기
if mode == 'train':
    # transform_train = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5), RandomFlip()])
    # transform_val = transforms.Compose([RandomCrop(shape=(ny, nx)), Normalization(mean=0.5, std=0.5)])

    transform_train = transforms.Compose([RandomCrop(shape=(ny, nx)), RandomFlip()])
    transform_val = transforms.Compose([RandomCrop(shape=(ny, nx))])

    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform_val, task=task, opts=opts)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    # transform_test = transforms.Compose([Normalization(mean=0.5, std=0.5)])
    # transform_test = None
    transform_test = transforms.Compose([RandomCrop(shape=(ny, nx))])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
if network == "unet":
    net = UNet(nin=nch, nout=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "hourglass":
    net = Hourglass(nin=nch, nout=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "resnet":
    net = ResNet(nin=nch, nout=nch, nker=nker, learning_type=learning_type).to(device)
elif network == "srresnet":
    net = SRResNet(nin=nch, nout=nch, nker=nker, learning_type=learning_type).to(device)

## 손실함수 정의하기
# fn_loss = nn.BCEWithLogitsLoss().to(device)
fn_loss = nn.MSELoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

cmap = None

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0

# TRAIN MODE
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_mse = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # 손실함수 계산
            loss_mse += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_mse)))

            if batch % 20 == 0:
              # Tensorboard 저장하기
              # label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
              # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
              # output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

              label = fn_tonumpy(label)
              input = fn_tonumpy(input)
              output = fn_tonumpy(output)

              label = np.clip(label, a_min=0, a_max=1)
              input = np.clip(input, a_min=0, a_max=1)
              output = np.clip(output, a_min=0, a_max=1)

              id = num_batch_train * (epoch - 1) + batch

              plt.imsave(os.path.join(result_dir_train, 'png', '%04d_label.png' % id), label[0].squeeze(), cmap=cmap)
              plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input.png' % id), input[0].squeeze(), cmap=cmap)
              plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)

              # writer_train.add_image('label', label, id, dataformats='NHWC')
              # writer_train.add_image('input', input, id, dataformats='NHWC')
              # writer_train.add_image('output', output, id, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_mse), epoch)

        with torch.no_grad():
            net.eval()
            loss_mse = []

            for batch, data in enumerate(loader_val, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산하기
                loss = fn_loss(output, label)

                loss_mse += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_mse)))

                if batch % 10 == 0:
                  # Tensorboard 저장하기
                  # label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
                  # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                  # output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

                  label = fn_tonumpy(label)
                  input = fn_tonumpy(input)
                  output = fn_tonumpy(output)

                  label = np.clip(label, a_min=0, a_max=1)
                  input = np.clip(input, a_min=0, a_max=1)
                  output = np.clip(output, a_min=0, a_max=1)

                  id = num_batch_val * (epoch - 1) + batch

                  plt.imsave(os.path.join(result_dir_val, 'png', '%04d_label.png' % id), label[0].squeeze(), cmap=cmap)
                  plt.imsave(os.path.join(result_dir_val, 'png', '%04d_input.png' % id), input[0].squeeze(), cmap=cmap)
                  plt.imsave(os.path.join(result_dir_val, 'png', '%04d_output.png' % id), output[0].squeeze(), cmap=cmap)

                  # writer_val.add_image('label', label, id, dataformats='NHWC')
                  # writer_val.add_image('input', input, id, dataformats='NHWC')
                  # writer_val.add_image('output', output, id, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_mse), epoch)

        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    with torch.no_grad():
        net.eval()
        loss_mse = []

        for batch, data in enumerate(loader_test, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            nimg_in = input.shape
            nimg_out = label.shape
            npatch_in = [1, nch, ny, nx] if not network == "srresnet" else [1, nch, int(ny//opts[1][0]), int(nx//opts[1][0])]
            npatch_out = [1, nch, ny, nx]
            nmargin = [0, 0, 0, 0]

            patch = image2patch(input, nimg_in, npatch_in, nmargin).to(device)

            nout = [patch.shape[0], nch, ny, nx]
            output = torch.zeros(nout)

            for i in range(patch.shape[0]):
                output[i] = net(patch[[i]])

            output = patch2image(output, nimg_out, npatch_out, nmargin).to(device)

            # 손실함수 계산하기
            loss = fn_loss(output, label)

            loss_mse += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_mse)))

            # Tensorboard 저장하기
            # label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5))
            # input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            # output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5))

            label = fn_tonumpy(label)
            input = fn_tonumpy(input)
            output = fn_tonumpy(output)

            for j in range(label.shape[0]):
                id = batch_size * (batch - 1) + j

                label_ = label[j]
                input_ = input[j]
                output_ = output[j]

                np.save(os.path.join(result_dir_test, 'numpy', '%04d_label.npy' % id), label_)
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_input.npy' % id), input_)
                np.save(os.path.join(result_dir_test, 'numpy', '%04d_output.npy' % id), output_)

                label_ = np.clip(label_, a_min=0, a_max=1)
                input_ = np.clip(input_, a_min=0, a_max=1)
                output_ = np.clip(output_, a_min=0, a_max=1)

                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_label.png' % id), label_, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_input.png' % id), input_, cmap=cmap)
                plt.imsave(os.path.join(result_dir_test, 'png', '%04d_output.png' % id), output_, cmap=cmap)

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
          (batch, num_batch_test, np.mean(loss_mse)))

