import numpy as np
import torch
from torch import nn
import sys
import os
import time
import logging

from dataloaders_imagenet import get_loaders, get_test_loader
from config_imagenet import CONFIG_LAYER, CONFIG_SUPERNET
from utils import weights_init, check_tensor_in_list, FileLogger, accuracy
from caunas import SuperNet, SupernetLoss

from torchsummary import summary
import torchvision.transforms as transforms
import torchvision
from pthflops import count_ops
from utils import AverageMeter


def sample_architecture_from_the_supernet(num_epoch=None):
    supernet_param = {
        'config_layer' : CONFIG_LAYER,
        'max_cluster_size' : CONFIG_SUPERNET['cluster']['max_cluster_size'],
        'first_inchannel' : CONFIG_SUPERNET['train_settings']['first_inchannel'],
        'last_feature_size' : CONFIG_SUPERNET['train_settings']['last_feature_size'], 
        'cnt_classes' : CONFIG_SUPERNET['train_settings']['cnt_classes']
    }
    # --- logging --- #
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    if not os.path.exists('./logs/imagenet/sampled_{num_epoch}.pth'.format(num_epoch=num_epoch)[:-4]):
        os.mkdir('./logs/imagenet/sampled_{num_epoch}.pth'.format(num_epoch=num_epoch)[:-4])
    fh = logging.FileHandler(os.path.join('./logs/imagenet/sampled_{num_epoch}.pth'.format(num_epoch=num_epoch)[:-4], 
                            'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    # ----------------- #
    model = SuperNet(supernet_param).cuda()

    from collections import OrderedDict

    state_dict = torch.load(CONFIG_SUPERNET['train_settings']['path_to_save_model'])
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    module_list = []
    module_list.append(model.first)
    for layer in model.stages_to_search:
        logging.info(str(np.argmax(layer.thetas.detach().cpu().numpy())))
        module_list.append(layer.ops[np.argmax(layer.thetas.detach().cpu().numpy())])

    module_list.append(model.last)

    # Create Sampled model
    sampled_model = nn.Sequential(*module_list)
    if torch.cuda.device_count() > 1:
        print('Multi-GPUs : ', torch.cuda.device_count())
        sampled_model = nn.DataParallel(sampled_model)
    sampled_model.cuda()

    test_loader = get_test_loader(512,
                                CONFIG_SUPERNET['dataloading']['path_to_save_data'])


    # Additional Traning N epochs
    if num_epoch is not None:
        train_loader = get_loaders(1, 
                                512, 
                                CONFIG_SUPERNET['dataloading']['path_to_save_data'])

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(sampled_model.parameters(), 
                                    lr=CONFIG_SUPERNET['optimizer']['w_lr'],
                                    momentum=CONFIG_SUPERNET['optimizer']['w_momentum'],
                                    weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])
        last_epoch = -1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=num_epoch,
                                                                last_epoch=last_epoch)
        best_top1 = 0.0                                                        

        for epoch in range(1, num_epoch+1):
            # train
            train_step(epoch, sampled_model, train_loader, criterion, optimizer, scheduler, num_epoch=num_epoch)
            # test
            top1_avg = test_step(epoch, sampled_model, test_loader)
            scheduler.step()

            if best_top1 < top1_avg :
                best_top1 = top1_avg
                torch.save(model.state_dict(), './logs/imagenet/sampled_{num_epoch}.pth'.format(num_epoch=num_epoch))
        logging.info("Best Top1 : {}".format(best_top1))

    # Measure model size & params size
    summary(sampled_model, input_size=(3, 256, 256), batch_size=1, device='cuda')

    # # Measure model Flops
    # print('Count Operations in random tensor')
    # inp = torch.rand(1, 3, 256, 256).to('cuda')
    # estimated, estimation_dict = count_ops(sampled_model, inp)

    # Data load for measuring inference time
    loader = get_test_loader(1,
                                CONFIG_SUPERNET['dataloading']['path_to_save_data'])
    
    # Gpu Inference Time
    inference_time = AverageMeter()

    for i_step, (batch, _) in enumerate(loader):
        batch = batch.to('cuda')
        
        start = time.time()
        sampled_model(batch)
        inference_time.update(time.time() - start)
        
        if i_step == 999 : 
            print('Average {device} inference time : {time:.3f}'.format(device=batch.device.type, time=inference_time.avg))
            break
    print('Average {device} inference time : {time:.3f}'.format(device=batch.device.type, time=inference_time.avg))
            

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

def train_step(epoch, model, loader, criterion, optimizer, scheduler, num_epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.train()

    for step, (X, y) in enumerate(loader):
        X, y = X.cuda(), y.cuda()
        N = X.shape[0]

        outs = model(X)
        loss = criterion(outs, y)

        prec1, prec3 = accuracy(outs, y, topk=(1, 5))
        losses.update(loss.item(), X.size(0))
        top1.update(prec1.item(), X.size(0))
        top5.update(prec3.item(), X.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logging.info('Epoch:[{epoch}/{num_epoch}] | Train Loss: {loss:.4f} | top1: {top1:.4f} | top5: {top5:.4f}'.format(
        epoch=epoch,
        num_epoch=num_epoch,
        loss=losses.avg,
        top1=top1.avg,
        top5=top5.avg))

def test_step(epoch, model, loader):
        # Measure Test Accuracy
        prec1_list = []

        with torch.no_grad():
            model.eval()

            for step, (X, y) in enumerate(loader):
                X, y = X.cuda(), y.cuda()
                N = X.shape[0]

                outs = model(X)
                prec1, _ = accuracy(outs, y, topk=(1, 5))
                prec1_list.append(prec1)

        prec1_avg = sum(prec1_list) / len(prec1_list)
        logging.info("Top1 Accuracy is : {top1}".format(top1=prec1_avg))
        return prec1_avg


if __name__ == "__main__":
    sample_architecture_from_the_supernet(num_epoch=50)