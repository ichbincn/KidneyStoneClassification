import logging
import time
import os
import argparse
import json

import torch.nn.functional as F
import monai.losses
from monai.bundle import ConfigParser
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sys
from tqdm import tqdm
from src.models.networks.module import ResEncoder, CAL_Net
from src.models.networks.sc_net import SC_Net
import shutil
from utils import AverageMeter, generate_patch_mask, returnCAM, load_npy_data, set_seed, _init_fn
from src.dataloader.load_data import split_data, my_dataloader
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import itertools
import functools
import sklearn.metrics
from sklearn.metrics import accuracy_score


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

def main(args, logger):
    config = ConfigParser()
    config.read_config(args.config_file)
    task = args.task
    train_seg = True if 0 in task else False
    train_cla = True if 1 in task else False
    save_path = os.path.join(args.output_path, "seg-{}_cla-{}".format(train_seg, train_cla))
    start_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_path = os.path.join(save_path, start_time)
    model_dir = os.path.join(save_path, "models")
    summary_dir = os.path.join(save_path, "summarys")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    train_writer = SummaryWriter(os.path.join(summary_dir, 'train'), flush_secs=2)
    test_writer = SummaryWriter(os.path.join(summary_dir, 'test'), flush_secs=2)

    #model
    img_size = tuple([i//8 for i in args.input_size])
    seg_net = SC_Net(in_channels=384, out_channels=None, img_size=img_size)
    backbone_oi = ResEncoder(depth=7, in_channels=1)
    backbone_zoom = ResEncoder(depth=7, in_channels=2)
    backbone_mask = ResEncoder(depth=7, in_channels=1)
    cla_net = CAL_Net(backbone_mask, backbone_zoom, num_classes=args.num_classes)

    #seg_net = DataParallel(seg_net)
    #cla_net = DataParallel(cla_net)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seg_net.to(device)
    cla_net.to(device)
    backbone_oi.to(device)
    ##################

    #load pretrainmodel
    if args.pretrain_seg != "None":
        seg_net.load_state_dict(torch.load(args.pretrain_seg))
        print("Successfully loaded seg_net!")
    if args.pretrain_cla != "None":
        cla_net.load_state_dict(torch.load(args.pretrain_cla))
        print("Successfully loaded cla_net!")
    ######################

    if not train_seg:
        for name, param in seg_net.named_parameters():
            param.requires_grad = False
    if not train_cla:
        for name, param in cla_net.named_parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        params=itertools.chain(seg_net.parameters(), cla_net.parameters()),
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=args.milestones,
        gamma=args.gamma
    )

    #loss
    criterion_seg = monai.losses.DiceLoss()
    criterion_cla = torch.nn.CrossEntropyLoss()
    criterion_cam = monai.losses.DiceLoss()
    criterion_weight = args.loss_weights
    ############

    metrics_seg_list = config.get_parsed_content("VALIDATION#metrics#seg")

    metrics_seg = {k.func.__name__: k for k in metrics_seg_list if type(k) == functools.partial}

    metrics_seg.update({k.__class__.__name__: k for k in metrics_seg_list if type(k) != functools.partial})


    metrics_cla_list = config.get_parsed_content("VALIDATION#metrics#cla")
    metrics_cla = {}
    for k in metrics_cla_list:
        if type(k) == functools.partial:  # partial func
            metrics_cla[k.func.__name__] = k
        elif type(k) == str:  # func
            name = k.split('.')[-1]
            metrics_cla[name] = eval(k)
        else:  # class
            metrics_cla[k.__class__.__name__] = k

    #data
    train_info, val_info = split_data(args.input_path, rate=0.8)
    train_data_loader = my_dataloader(args.input_path,
                               train_info,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               size=args.input_size)
    test_data_loader = my_dataloader(args.input_path,
                             val_info,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             size=args.input_size)
    #########################

    print("Start training")
    logger.logger.info('Start training......\n')

    running_loss = AverageMeter()
    running_loss_seg = AverageMeter()
    running_loss_cla = AverageMeter()
    metrics_seg_values = {k: AverageMeter() for k in metrics_seg}
    best_metric = {k:{"epoch":0, "value":0} for k in config["VALIDATION"]["save_best_metric"]}

    for epoch in range(args.epochs):
        s_t = time.time()

        seg_net.train()
        cla_net.train()
        running_loss.reset()
        running_loss_seg.reset()
        running_loss_cla.reset()
        for k in metrics_seg_values:
            metrics_seg_values[k].reset()

        gt = []
        cla_pred = []

        for batch_idx, (img, seg_label, cla_label) in enumerate(train_data_loader):
            bs, c, h, w, d = img.shape
            img = img.to(device)
            seg_label = seg_label.to(device)
            cla_label = cla_label.to(device)

            seg_loss = 0
            cla_loss = 0
            cam_loss = 0
            if train_seg:
                res_encoder_output = backbone_oi(img)
                pred_mask = seg_net(res_encoder_output)
                #pred_mask = activation(pred_mask)
                seg_loss += criterion_seg(pred_mask, seg_label)
                pred_mask = torch.where(pred_mask > 0.5, 1, 0).byte()

                if train_cla:
                    original_seg, zoom_seg = generate_patch_mask(img, pred_mask)
                    cla_out = cla_net(res_encoder_output, zoom_seg, original_seg)

                    features = cla_net.finalconv
                    fc_weights = cla_net.fc.weight.data
                    cams = returnCAM(features, fc_weights, cla_label, size=args.input_size)
                    cams = cams.to(device)
                    cam_loss += criterion_cam(cams, seg_label)

                    cla_loss += criterion_cla(cla_out, cla_label)

                    gt.append(cla_label.cpu())
                    cla_pred.append(F.softmax(cla_out, dim=-1).argmax(1, keepdim=True).cpu())

                for k in metrics_seg:

                    res = metrics_seg[k](pred_mask, seg_label)
                    print(res, pred_mask, seg_label)

                    if type(res) == torch.Tensor and res.shape[0] > 0:
                        res = torch.mean(res[~torch.isnan(res)])
                    metrics_seg_values[k].update(res, bs)
                break

            else:
                if train_cla:
                    res_encoder_output = backbone_oi(img)
                    original_seg, zoom_seg = generate_patch_mask(img, seg_label)
                    cla_out = cla_net(res_encoder_output, zoom_seg, original_seg)
                    features = cla_net.finalconv
                    fc_weights = cla_net.fc.weight.data
                    cams = returnCAM(features, fc_weights, cla_label)
                    cam_loss += criterion_cam(cams, seg_label)

                    cla_loss += criterion_cla(cla_out, cla_label)

                    gt.append(cla_label.cpu())
                    cla_pred.append(F.softmax(cla_out, dim=-1).argmax(1, keepdim=True).cpu())
                else:
                    raise ValueError("Both of 'train_seg' and 'train_cla' are False")

            w_s, w_cam, w_c = criterion_weight
            loss = w_s * seg_loss + w_cam * cam_loss + w_c * cla_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss.update(loss, bs)
            running_loss_seg.update(seg_loss, bs)
            running_loss_cla.update(cla_loss, bs)

            # if train_cla:
            #     gt.append(cla_label.cpu())
            #     cla_pred.append(F.softmax(cla_out, dim=-1).argmax(1, keepdim=True).cpu())

        if train_cla:
            gt = torch.cat(gt, dim=0)
            cla_pred = torch.cat(cla_pred, dim=0)
            metrics_cla_values = {k: m(gt, cla_pred) for k, m in metrics_cla.items()}
        else:
            metrics_cla_values = {k: 0 for k in metrics_cla}

        scheduler.step()
        print("----------------------------------------------------------------------------------")
        print("Epoch [%d/%d], time: %.2f" % (epoch, args.epochs, time.time() - s_t))
        print("Loss: total-{}_segmentation-{}_classification-{}.".format(
            running_loss.avg, running_loss_seg.avg, running_loss_cla.avg
        ))
        print("Metrics:", {k: metrics_seg_values[k].avg for k in metrics_seg_values}, metrics_cla_values)

        train_writer.add_scalar("Loss/total", running_loss.avg, epoch)
        train_writer.add_scalar("Loss/segmentation", running_loss_seg.avg, epoch)
        train_writer.add_scalar("Loss/classification", running_loss_cla.avg, epoch)
        train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        logger.logger.info('%d epoch train total loss: %.3f \n' % (epoch, running_loss.avg))
        logger.logger.info('%d epoch train segmentation loss: %.3f \n' % (epoch, running_loss_seg.avg))
        logger.logger.info('%d epoch train classification loss: %.3f \n' % (epoch, running_loss_cla.avg))
        for k in metrics_seg_values:
            train_writer.add_scalar("Metrics_seg/" + k, metrics_seg_values[k].avg, epoch)
            logger.logger.info('%d epoch Metrics_seg/{}: %.3f \n'.format(k) % (epoch, metrics_seg_values[k].avg))
        for k in metrics_cla_values:
            train_writer.add_scalar("Metrics_cla/" + k, metrics_cla_values[k], epoch)
            logger.logger.info('%d epoch Metrics_cla/{}: %.3f \n'.format(k) % (epoch, metrics_cla_values[k]))
        logger.logger.info('%d epoch train time: %.3f \n' % (epoch, time.time() - s_t))

        if epoch % args.save_epoch == 0:
            torch.save(seg_net.state_dict(), os.path.join(model_dir, "seg_net_params_epo-{}.pkl".format(epoch)))
            torch.save(cla_net.state_dict(), os.path.join(model_dir, "cla_net_params_epo-{}.pkl".format(epoch)))

        print("Start testing...")
        logger.logger.info('Start testing......\n')
        s_t = time.time()
        seg_net.eval()
        cla_net.eval()
        running_loss.reset()
        running_loss_seg.reset()
        running_loss_cla.reset()
        for k in metrics_seg_values:
            metrics_seg_values[k].reset()

        gt = []
        cla_pred = []

        with torch.no_grad():

            for batch_idx, (img, seg_label, cla_label) in enumerate(test_data_loader):
                bs, c, h, w, d = img.shape
                img = img.to(device)
                seg_label = seg_label.to(device)
                cla_label = cla_label.to(device)

                seg_loss = 0
                cla_loss = 0
                if train_seg:
                    res_encoder_output = backbone_oi(img)
                    pred_mask = seg_net(res_encoder_output)
                    #pred_mask = activation(pred_mask)
                    seg_loss += criterion_seg(pred_mask, seg_label)
                    pred_mask = torch.where(pred_mask > 0.5, 1, 0).byte()

                    if train_cla:
                        original_seg, zoom_seg = generate_patch_mask(img, pred_mask)
                        original_seg = original_seg.to(device)
                        zoom_seg = zoom_seg.to(device)

                        cla_out = cla_net(res_encoder_output, zoom_seg, original_seg)

                        cla_loss += criterion_cla(cla_out, cla_label)

                        gt.append(cla_label.cpu())
                        cla_pred.append(F.softmax(cla_out, dim=-1).argmax(1, keepdim=True).cpu())

                    seg_label = seg_label.byte()
                    for k in metrics_seg:
                        res = metrics_seg[k](pred_mask, seg_label)
                        if type(res) == torch.Tensor and res.shape[0] > 0:
                            res = torch.mean(res[~torch.isnan(res)])
                        metrics_seg_values[k].update(res, bs)

                else:
                    if train_cla:
                        original_seg, zoom_seg = generate_patch_mask(img, seg_label)
                        img = img.to(device)
                        original_seg = original_seg.to(device)
                        zoom_seg = zoom_seg.to(device)
                        res_encoder_output = backbone_oi(img)

                        cla_out = cla_net(res_encoder_output, zoom_seg, original_seg)

                        cla_loss += criterion_cla(cla_out, cla_label)

                        gt.append(cla_label.cpu())
                        cla_pred.append(F.softmax(cla_out, dim=-1).argmax(1, keepdim=True).cpu())
                    else:
                        raise ValueError("Both of 'train_seg' and 'train_cla' are False")

                w_s, w_cam, w_c = criterion_weight
                loss = w_s * seg_loss + w_c * cla_loss

                running_loss.update(loss, bs)
                running_loss_seg.update(seg_loss, bs)
                running_loss_cla.update(cla_loss, bs)

                # if train_cla:
                #     gt.append(cla_label.cpu())
                #     cla_pred.append(F.softmax(cla_out, dim=-1).argmax(1, keepdim=True).cpu())

            if train_cla:
                gt = torch.cat(gt, dim=0)
                cla_pred = torch.cat(cla_pred, dim=0)
                metrics_cla_values = {k: m(gt, cla_pred) for k, m in metrics_cla.items()}
            else:
                metrics_cla_values = {k: 0 for k in metrics_cla}

            print("Test results:")
            print("Loss: total-{}_segmentation-{}_classification-{}.".format(
                running_loss.avg, running_loss_seg.avg, running_loss_cla.avg
            ))
            print("Metrics:", {k: metrics_seg_values[k].avg for k in metrics_seg_values}, metrics_cla_values)
            test_writer.add_scalar("Loss/total", running_loss.avg, epoch)
            test_writer.add_scalar("Loss/segmentation", running_loss_seg.avg, epoch)
            test_writer.add_scalar("Loss/classification", running_loss_cla.avg, epoch)
            logger.logger.info('%d epoch test total loss: %.3f \n' % (epoch, running_loss.avg))
            logger.logger.info('%d epoch test segmentation loss: %.3f \n' % (epoch, running_loss_seg.avg))
            logger.logger.info('%d epoch test classification loss: %.3f \n' % (epoch, running_loss_cla.avg))
            for k in metrics_seg_values:
                test_writer.add_scalar("Metrics_seg/" + k, metrics_seg_values[k].avg, epoch)
                logger.logger.info('%d epoch Metrics_seg/{}: %.3f \n'.format(k) % (epoch, metrics_seg_values[k].avg))
            for k in metrics_cla_values:
                test_writer.add_scalar("Metrics_cla/" + k, metrics_cla_values[k], epoch)
                logger.logger.info('%d epoch Metrics_cla/{}: %.3f \n'.format(k) % (epoch, metrics_cla_values[k]))

            for k in best_metric:
                if k in metrics_seg_values and metrics_seg_values[k].avg > best_metric[k]["value"]:
                    print('{} increased ({:.6f} --> {:.6f}).'.format(k, best_metric[k]["value"],
                                                                     metrics_seg_values[k].avg))
                    torch.save(cla_net.state_dict(), os.path.join(model_dir, "max_{}-cla_net.pkl".format(k)))
                    torch.save(seg_net.state_dict(), os.path.join(model_dir, "max_{}-seg_net.pkl".format(k)))
                    best_metric[k]["value"] = metrics_seg_values[k].avg
                    best_metric[k]["epoch"] = epoch
                if k in metrics_cla_values and metrics_cla_values[k] > best_metric[k]["value"]:
                    print('{} increased ({:.6f} --> {:.6f}).'.format(k, best_metric[k]["value"], metrics_cla_values[k]))
                    torch.save(cla_net.state_dict(), os.path.join(model_dir, "max_{}-cla_net.pkl".format(k)))
                    torch.save(seg_net.state_dict(), os.path.join(model_dir, "max_{}-seg_net.pkl".format(k)))
                    best_metric[k]["value"] = metrics_cla_values[k]
                    best_metric[k]["epoch"] = epoch

        with open(os.path.join(model_dir, "best_metrics.json"), "w") as f:
            json.dump(best_metric, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='configs/config.yaml')
    parser.add_argument('--task', type=list, default=[0, 1])
    parser.add_argument('--pretrain_seg', type=str, default='None')
    parser.add_argument('--pretrain_cla', type=str, default='None')
    parser.add_argument('--input_path', type=str, default='/home/KidneyData/data')
    parser.add_argument('--output_path', type=str, default='./results')
    parser.add_argument('--input_size', type=tuple, default=(256, 256, 256))
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_epoch', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--milestones', type=list, default=[100])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--loss_weights', type=list, default=[0.2, 1, 1])
    parser.add_argument('--log_dir', type=str, default='./Logs')
    parser.add_argument('--save_dir', type=str, default='./Save')
    parser.add_argument('--num_workers', type=int, default=0)

    opt = parser.parse_args()

    logger = Logger()
    main(opt, logger)


