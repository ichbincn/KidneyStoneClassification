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

from tqdm import tqdm
import torch

from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
from src.dataloader.load_data import split_data, my_dataloader

from torch.nn.parallel import DataParallel




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


class Trainer:
    def __init__(self, model, optimizer, device, train_loader, test_loader, epochs, scheduler, args, summaryWriter):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.epoch = 0
        self.best_acc = 0
        self.args = args
        self.load_model()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.summaryWriter = summaryWriter

    def __call__(self):
        for epoch in tqdm(range(self.epochs)):
            start = time.time()
            self.epoch = epoch+1
            self.train_one_epoch()
            self.num_params = sum([param.nelement() for param in self.model.parameters()])
            self.scheduler.step()

            end = time.time()
            print("Epoch: {}, train time: {}".format(epoch, end - start))
            if epoch % 1 == 0:
                self.evaluate()

    def load_model(self):
        if self.args.MODEL_WEIGHT:
            # self.model =
            self.model.load_state_dict(torch.load(self.args.MODEL_WEIGHT)['model_state_dict'])
            print('load model weight success!')

    def evaluate(self):
        self.model.eval()
        total_step = 0
        per_epoch_loss = 0
        per_epoch_num_correct = 0
        with torch.no_grad():
            for inx, (x, mask, label) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                x = x.to(self.device)
                label = label.to(self.device)
                total_step += x.shape[0]
                logits = self.model(x)
                loss = self.loss_function(logits, label)
                per_epoch_loss += loss.item()
                pred = logits.argmax(dim=1)
                per_epoch_num_correct += torch.eq(pred, label).sum().item()
            test_acc = per_epoch_num_correct / total_step
            print(f'TEST: Epoch:{self.epoch}/{self.epochs}, Loss:{per_epoch_loss/(inx+1)}, acc:{test_acc}')
            self.summaryWriter.add_scalar("Loss/TEST", per_epoch_loss/len(self.test_loader), self.epoch)
            self.summaryWriter.add_scalar("acc/TEST", test_acc, self.epoch)


            if self.epoch % self.args.save_epoch == 0:
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),  # *模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                    'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                    'best_val_score': per_epoch_num_correct / total_step,
                    'num_params': self.num_params,
                    'epoch': self.epoch
                }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'checkpoint-%d.pth' % self.epoch))
                logger.logger.info('save model %d successed......\n'%self.epoch)

            if self.best_acc < test_acc:
                best_acc = test_acc
                # logger.logger.info('best model in %d epoch, train acc: %.3f \n' % (self.epoch, train_acc))
                # logger.logger.info('best model in %d epoch, validation acc: %.3f \n' % (epoch, val_acc))
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),  # *模型参数
                    'optimizer_state_dict': self.optimizer.state_dict(),  # *优化器参数
                    'scheduler_state_dict': self.scheduler.state_dict(),  # *scheduler
                    'best_val_score': self.best_acc,
                    'num_params': self.num_params,
                    'epoch': self.epoch
                }
                torch.save(checkpoint, os.path.join(self.args.save_dir, 'best_checkpoint.pth'))
                logger.logger.info('save best model  successed......\n')

    def train_one_epoch(self):
        per_epoch_loss = 0
        total_step = 0
        num_correct = 0
        self.model.train()
        for inx, (x, mask, label) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            x = x.to(self.device)
            label = label.to(self.device)
            logits = self.model(x)

            loss = self.loss_function(logits, label)
            per_epoch_loss += loss.item()
            total_step += x.shape[0]
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pred = logits.argmax(dim=1)
            # print(f'logits:{logits}, pred:{pred}, label:{label}')
            num_correct += torch.eq(pred, label).sum().item()
            # if inx % 5 == 0:
            # print(f'iters:{inx}/{len(self.train_loader)}, Loss:{loss.item()}, acc:{num_correct/total_step}')
        self.summaryWriter.add_scalar("Loss/TRAIN", per_epoch_loss / len(self.train_loader), self.epoch)
        self.summaryWriter.add_scalar("acc/TRAIN", num_correct/total_step, self.epoch)
        print(f'train epoch:{self.epoch}/{self.epochs}, Loss:{per_epoch_loss / len(self.train_loader)}, acc:{num_correct / total_step}')

def main(args, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("can use {} gpus".format(torch.cuda.device_count()))
    print(device)
    model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=args.num_classes)
    # model = models.resnet18(pretrained=True)
    # num_ftrs = model.fc.in_features  # 获取低级特征维度
    # model.fc = nn.Linear(num_ftrs, args.num_classes)  # 替换新的输出层

    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    # data_dir = r'C:\Users\Asus\Desktop\肺腺癌\data\肾结石数据\KdneyStone\202310326结石成分分析龙岗区人民医院李星智'
    # if not os.path.exists(data_dir):
    #     data_dir = '/home/wangchangmiao/kidney/data/data'
    data_dir = args.input_path
    train_infos, val_infos = split_data(data_dir)
    train_loader = my_dataloader(data_dir, train_infos, batch_size=args.batch_size, input_size=args.input_size)
    val_loader = my_dataloader(data_dir, val_infos, batch_size=args.batch_size, input_size=args.input_size)
    logger.logger.info('start training......\n')
    summaryWriter = SummaryWriter(log_dir=args.log_dir)
    # train_writer = SummaryWriter(os.path.join(summary_dir, 'train'), flush_secs=2)
    # test_writer = SummaryWriter(os.path.join(summary_dir, 'test'), flush_secs=2)
    trainer = Trainer(model,
                      optimizer,
                      device,
                      train_loader,
                      val_loader,
                      args.epochs,
                      scheduler,
                      args,
                      summaryWriter)
    trainer()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save_epoch', type=int, default=5)
    parser.add_argument('--log_dir', type=str, default='./Logs')
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--input_size', type=float, default=(128, 128, 128))
    parser.add_argument('--input_path', type=str, default='/home/wangchangmiao/kidney/data/data')
    parser.add_argument('--MODEL_WEIGHT', type=str, default=None)

    opt = parser.parse_args()

    logger = Logger()
    main(opt, logger)