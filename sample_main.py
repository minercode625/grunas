import numpy as np
import torch
from torch import nn
import os
import time

from dataloaders import get_loaders, get_test_loader
from config import CONFIG_LAYER, CONFIG_SUPERNET
from utils import weights_init, check_tensor_in_list, FileLogger, accuracy
from caunas import SuperNet, SupernetLoss

from torchsummary import summary
import torchvision.transforms as transforms
import torchvision
from pthflops import count_ops
from utils import AverageMeter


def sample_architecture_from_the_supernet():
    test_loader = get_test_loader(CONFIG_SUPERNET['dataloading']['batch_size'],
                                  CONFIG_SUPERNET['dataloading']['path_to_save_data'])
    model = SuperNet(CONFIG_LAYER, CONFIG_SUPERNET['cluster']['max_cluster_size'], cnt_classes=10).cuda()
    # model = nn.DataParallel(model, device_ids=[0])

    model.load_state_dict(torch.load(CONFIG_SUPERNET['train_settings']['path_to_save_model']))
    module_list = []
    module_list.append(model.module.first)
    for layer in model.module.stages_to_search:
        print(str(np.argmax(layer.thetas.detach().cpu().numpy())))
        module_list.append(layer.ops[np.argmax(layer.thetas.detach().cpu().numpy())])

    module_list.append(model.module.last)

    # sampled_model = nn.Sequential(*module_list).cuda()
    sampled_model = nn.Sequential(*module_list)


    # Additional Traning 50 epochs
    train_loader = get_loaders(1, 
                            CONFIG_SUPERNET['dataloading']['batch_size'], 
                            CONFIG_SUPERNET['dataloading']['path_to_save_data'])
    
    # sampled_model = nn.DataParallel(sampled_model, device_ids=[0])
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(sampled_model.parameters(), 
                                lr=CONFIG_SUPERNET['optimizer']['w_lr'],
                                momentum=CONFIG_SUPERNET['optimizer']['w_momentum'],
                                weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
    last_epoch = -1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=50,
                                                            last_epoch=last_epoch)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # sampled_model.train()

    # for epoch in range(1, 51):

    #     for step, (X, y) in enumerate(train_loader):
    #         X, y = X.cuda(), y.cuda()
    #         N = X.shape[0]

    #         optimizer.zero_grad()
    #         outs = sampled_model(X)
    #         loss = criterion(outs, y)

    #         prec1, prec3 = accuracy(outs, y, topk=(1, 5))
    #         losses.update(loss.item(), X.size(0))
    #         top1.update(prec1.item(), X.size(0))
    #         top5.update(prec3.item(), X.size(0))

    #         loss.backward()
    #         optimizer.step()

    #     print('Epoch:[{epoch}/50] | Loss: {loss:.4f} | top1: {top1:.4f} | top5: {top5:.4f}'.format(
    #         epoch=epoch,
    #         loss=losses.avg,
    #         top1=top1.avg,
    #         top5=top5.avg))
    #     scheduler.step()

    sampled_model.load_state_dict(torch.load('logs/sampled_model_50.pth'))

    # prec1_list = []
    # prec3_list = []

    # with torch.no_grad():
    #     sampled_model.eval()
    #     summary(sampled_model, input_size=(3, 32, 32), batch_size=1, device='cuda')

    #     for step, (X, y) in enumerate(test_loader):
    #         X, y = X.cuda(), y.cuda()
    #         N = X.shape[0]

    #         outs = sampled_model(X)
    #         prec1, prec3 = accuracy(outs, y, topk=(1, 5))
    #         prec1_list.append(prec1)
    #         prec3_list.append(prec3)

    # prec1_avg = sum(prec1_list) / len(prec1_list)
    # prec3_avg = sum(prec3_list) / len(prec3_list)
    # print(prec1_avg)
    # print(prec3_avg)

    # Measure model Flops
    # print('Count Operations in random tensor')
    # inp = torch.rand(1, 3, 32, 32).to('cuda')
    # estimated, estimation_dict = count_ops(sampled_model, inp)

    # Data load
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
    dataset = torchvision.datasets.CIFAR10('./cifar10_data', train=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Gpu Inference Time
    # inference_time = AverageMeter()

    # for i_step, (batch, _) in enumerate(loader):
    #     batch = batch.to('cuda')
        
    #     start = time.time()
    #     sampled_model(batch)
    #     inference_time.update(time.time() - start)
        
    #     if i_step == 999 : 
    #         print('Average {device} inference time : {time:.3f}'.format(device=batch.device.type, time=inference_time.avg))
    #         break
    # print('Average {device} inference time : {time:.3f}'.format(device=batch.device.type, time=inference_time.avg))
            

    # Cpu Inference Time
    inference_time = AverageMeter()
    sampled_model = sampled_model.to('cpu')
    for i_step, (batch, _) in enumerate(loader):
        start = time.time()
        sampled_model(batch)
        inference_time.update(time.time() - start)
        
        if i_step == 999 : 
            print('Average {device} inference time : {time:.3f}'.format(device=batch.device.type, time=inference_time.avg))
            break
    print('Average {device} inference time : {time:.3f}'.format(device=batch.device.type, time=inference_time.avg))

    # torch.save(sampled_model.state_dict(), 'logs/sampled_model_50.pth')

if __name__ == "__main__":
    # f = FileLogger()
    # f.file_reset()
    sample_architecture_from_the_supernet()