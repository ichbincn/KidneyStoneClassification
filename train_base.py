# -*- coding: utf-8 -*-
# Time    : 2023/10/30 20:35
# Author  : fanc
# File    : train_base.py

import warnings
warnings.filterwarnings("ignore")
import logging  # 引入logging模块
import os.path
import time
import os
import math
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import sys
from tqdm import tqdm
import torch
from torch.utils import data
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from src.dataloader.load_data import split_data, my_dataloader

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# 构建残差网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(-1, 512)
        out = self.fc(out)
        out = self.softmax(out)
        return out
class Logger:
    def __init__(self,mode='w'):
        # 第一步，创建一个logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        # 第二步，创建一个handler，用于写入日志文件
        rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
        log_path = os.getcwd() + '/Logs/'
        log_name = log_path + rq + '.log'
        logfile = log_name
        fh = logging.FileHandler(logfile, mode=mode)
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
        # 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        # 第四步，将logger添加到handler里面
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

def train_one_epoch(model, optimizer, data_loader, device):
    total_step = len(data_loader)
    per_epoch_loss = 0
    num_correct = 0
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    with torch.enable_grad():
        for x, mask, label in tqdm(data_loader):
            x = x.to(device)
            label = label.to(device)

            logits = model(x)
            loss = loss_function(logits, label)
            per_epoch_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = logits.argmax(dim=1)

            num_correct += torch.eq(pred, label.argmax(dim=1)).sum().float().item()
        train_loss = per_epoch_loss/total_step
        train_acc = num_correct/len(data_loader.dataset)
    return train_loss, train_acc

def evaluate(model, data_loader, device):
    val_num_correct = 0
    model.eval()
    with torch.no_grad():
        for x, mask, label in tqdm(data_loader):
            x = x.to(device)
            label = label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            val_num_correct += torch.eq(pred, label.argmax(dim=1)).sum().float().item()
        val_acc = val_num_correct/len(data_loader.dataset)
    return val_acc

def main(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=args.num_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    data_dir = r'C:\Users\Asus\Desktop\肺腺癌\data\肾结石数据\KdneyStone\202310326结石成分分析龙岗区人民医院李星智'
    train_infos, val_infos = split_data(data_dir)
    train_loader = my_dataloader(data_dir, train_infos, batch_size=1)
    val_loader = my_dataloader(data_dir, val_infos, batch_size=1)
    logger.logger.info('start training......\n')
    summaryWriter = SummaryWriter(log_dir=args.log_dir)

    best_acc = 0
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        print("Train Epoch: {}\t Loss: {:.3f}\t Acc: {:.3f}".format(epoch, train_loss, train_acc))
        summaryWriter.add_scalars('loss', {"loss": train_loss}, epoch)
        summaryWriter.add_scalars('acc', {"acc": train_acc}, epoch)
        logger.logger.info('%d epoch train loss: %.3f \n' % (epoch, train_loss))
        logger.logger.info('%d epoch train acc: %.3f \n' % (epoch, train_acc))

        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        num_params = sum([param.nelement() for param in model.parameters()])
        print("val Epoch: {}\t Acc: {:.3f}\t time: {:.3f}".format(epoch, val_acc, time.time()-start))
        print("total params: {:.3f}".format(num_params))
        summaryWriter.add_scalars('acc', {"val_acc": val_acc}, epoch)
        summaryWriter.add_scalars('time', {"time": (time.time() - start)}, epoch)
        logger.logger.info('%d epoch validation acc: %.3f \n' % (epoch, val_acc))
        logger.logger.info('%d epoch cost time: %.3f \n' % (epoch, time.time()-start))
        logger.logger.info('%d epoch total params: %.3f \n' % (epoch, num_params))

        if epoch % args.save_epoch == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),  # *模型参数
                'optimizer_state_dict': optimizer.state_dict(),  # *优化器参数
                'scheduler_state_dict': scheduler.state_dict(),  # *scheduler
                'best_val_score': val_acc,
                'num_params': num_params,
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint-%d.pt' % epoch))
            logger.logger.info('save model %d successed......\n'%epoch)

        if best_acc < val_acc:
            best_acc = val_acc
            logger.logger.info('best model in %d epoch, train acc: %.3f \n' % (epoch, train_acc))
            logger.logger.info('best model in %d epoch, validation acc: %.3f \n' % (epoch, val_acc))
            checkpoint = {
                'model_state_dict': model.state_dict(),  # *模型参数
                'optimizer_state_dict': optimizer.state_dict(),  # *优化器参数
                'scheduler_state_dict': scheduler.state_dict(),  # *scheduler
                'best_val_score': best_acc,
                'num_params': num_params,
                'epoch': epoch
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_checkpoint.pt'))
            logger.logger.info('save best model  successed......\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_epoch', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--num_workers', type=int, default=0)

    opt = parser.parse_args()

    logger = Logger()
    main(opt, logger)