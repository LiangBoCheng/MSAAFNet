import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pdb, os, argparse
from datetime import datetime
from model.MSAAFNet import MSAAFNet
# from model.MSAAFNet_V3 import MSAAFNet
from utils.data import get_loader
from utils.func import AvgMeter, clip_gradient, adjust_lr
import pytorch_iou
import pytorch_fm

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=60, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') 
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--trainset', type=str, default='EORSSD', help='training  dataset')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models
model = MSAAFNet()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = 'E:/Datasets/' + opt.trainset + '/train-images/'
gt_root = 'E:/Datasets/' + opt.trainset + '/train-labels/'
edge_root = 'E:/Datasets/' + opt.trainset + '/train-labels-canny/'

train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)
floss = pytorch_fm.FLoss()
size_rates = [0.75, 1, 1.25]  # multi-scale training
train_losses=[]

def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
           optimizer.zero_grad()
           images, gts, edges = pack
           images = Variable(images).cuda()
           gts = Variable(gts).cuda()
           edges = Variable(edges).cuda()

           # multi-scale training samples
           trainsize = int(round(opt.trainsize * rate / 32) * 32)

           if rate != 1:
               images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
               gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
               edges = F.interpolate(edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

           s1, s1_sig, edge1, s2, s2_sig, edge2, s3, s3_sig, edge3, s4, s4_sig, edge4, s5, s5_sig, edge5 = model(images)
           # bce+iou+fmloss
           loss1 = CE(s1, gts) + IOU(s1_sig, gts) + floss(s1_sig, gts) + CE(edge1, edges)
           loss2 = CE(s2, gts) + IOU(s2_sig, gts) + floss(s2_sig, gts) + CE(edge2, edges)
           loss3 = CE(s3, gts) + IOU(s3_sig, gts) + floss(s3_sig, gts) + CE(edge3, edges)
           loss4 = CE(s4, gts) + IOU(s4_sig, gts) + floss(s4_sig, gts) + CE(edge4, edges)
           loss5 = CE(s5, gts) + IOU(s5_sig, gts) + floss(s5_sig, gts) + CE(edge5, edges)

           loss = loss1 + loss2 + loss3 + loss4 + loss5

           loss.backward()

           clip_gradient(optimizer, opt.clip)
           optimizer.step()

           if rate == 1:
               loss_record1.update(loss1.data, opt.batchsize)
               loss_record2.update(loss2.data, opt.batchsize)
               loss_record3.update(loss3.data, opt.batchsize)
               loss_record4.update(loss4.data, opt.batchsize)
               loss_record5.update(loss5.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                           loss2.data))
        if i == total_step:
            train_losses.append(loss.data)

    save_path = 'models/MSAAFNet/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) >= 40:
        torch.save(model.state_dict(), save_path + 'MSAAFNet_' + opt.trainset + '.pth' + '.%d' % epoch)


print("Let's go!")
if __name__ == '__main__':
 for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
