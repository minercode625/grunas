import numpy as np
import torch
from torch import nn
import os
import time

from utils import weights_init, check_tensor_in_list, FileLogger, accuracy
from caunas import SuperNet, SupernetLoss

from torchsummary import summary
import torchvision.transforms as transforms
import torchvision
from pthflops import count_ops
from utils import AverageMeter

DATA = 'MNIST'

if DATA == 'CIFAR10':
    from config import CONFIG_SUPERNET, CONFIG_LAYER
    from dataloaders import get_loaders, get_test_loader
elif DATA == 'IMAGENET':
    from config_imagenet import CONFIG_LAYER, CONFIG_SUPERNET
    from dataloaders_imagenet import get_loaders, get_test_loader
elif DATA == 'MNIST':
    from config_mnist import CONFIG_LAYER, CONFIG_SUPERNET
    from dataloaders_mnist import get_loaders, get_test_loader

def sample_architecture_from_the_supernet(num_epoch=None):
    supernet_param = {
        'config_layer' : CONFIG_LAYER,
        'max_cluster_size' : CONFIG_SUPERNET['cluster']['max_cluster_size'],
        'first_inchannel' : CONFIG_SUPERNET['train_settings']['first_inchannel'],
        'last_feature_size' : CONFIG_SUPERNET['train_settings']['last_feature_size'], 
        'cnt_classes' : CONFIG_SUPERNET['train_settings']['cnt_classes']
    }
    model = SuperNet(supernet_param).cuda()

    from collections import OrderedDict

    state_dict = torch.load(CONFIG_SUPERNET['train_settings']['path_to_save_model'])
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    # model.load_state_dict(torch.load(CONFIG_SUPERNET['train_settings']['path_to_save_model']))

    module_list = []
    module_list.append(model.first)
    for layer in model.stages_to_search:
        print(str(np.argmax(layer.thetas.detach().cpu().numpy())))
        module_list.append(layer.ops[np.argmax(layer.thetas.detach().cpu().numpy())])

    module_list.append(model.last)

    # Create Sampled model
    sampled_model = nn.Sequential(*module_list).cuda()

    test_loader = get_test_loader(CONFIG_SUPERNET['dataloading']['batch_size'],
                                CONFIG_SUPERNET['dataloading']['path_to_save_data'])

    # Additional Traning N epochs
    if num_epoch is not None:
        train_loader = get_loaders(1, 
                                CONFIG_SUPERNET['dataloading']['batch_size'], 
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
                torch.save(model.state_dict(), './logs/mnist/sampled_{num_epoch}.pth'.format(num_epoch=num_epoch))

    # Measure model size & params size
    summary(sampled_model, input_size=(1, 28, 28), batch_size=1, device='cuda')
    
    # Measure model Flops
    print('Count Operations in random tensor')
    inp = torch.rand(1, 1, 28, 28).to('cuda')
    estimated, estimation_dict = count_ops(sampled_model, inp)

    # Data load for measuring inference time
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
    dataset = torchvision.datasets.MNIST('./mnist_data', train=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

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

    print('Epoch:[{epoch}/{num_epoch}] | Loss: {loss:.4f} | top1: {top1:.4f} | top5: {top5:.4f}'.format(
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
        print("Top1 Accuracy is : {top1}".format(top1=prec1_avg))
        return prec1_avg


if __name__ == "__main__":
    # If you want additional training loop for sampled model, you choose num_epoch
    sample_architecture_from_the_supernet(num_epoch=10)