import numpy as np
import torch
from torch import nn
import os
from utils import weights_init, check_tensor_in_list, FileLogger, accuracy
from caunas import SuperNet, SupernetLoss
from train_supernet import TrainerSupernet

manual_seed = 1
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
torch.backends.cudnn.benchmark = True

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

def train_supernet():
    
    #### DataLoading
    train_w_loader, train_thetas_loader = get_loaders(CONFIG_SUPERNET['dataloading']['w_share_in_train'],
                                                      CONFIG_SUPERNET['dataloading']['batch_size'],
                                                      CONFIG_SUPERNET['dataloading']['path_to_save_data'],
                                                      )
    test_loader = get_test_loader(CONFIG_SUPERNET['dataloading']['batch_size'],
                                  CONFIG_SUPERNET['dataloading']['path_to_save_data'])

    #### Model
    supernet_param = {
        'config_layer' : CONFIG_LAYER,
        'max_cluster_size' : CONFIG_SUPERNET['cluster']['max_cluster_size'],
        'first_inchannel' : CONFIG_SUPERNET['train_settings']['first_inchannel'],
        'last_feature_size' : CONFIG_SUPERNET['train_settings']['last_feature_size'], 
        'cnt_classes' : CONFIG_SUPERNET['train_settings']['cnt_classes']
    }
    model = SuperNet(supernet_param).cuda()
    model = model.apply(weights_init)
    model = nn.DataParallel(model, device_ids=[0])

    #### Loss, Optimizer and Scheduler
    supernetloss_param = {
        'alpha' : CONFIG_SUPERNET['loss']['alpha'], 
        'beta' : CONFIG_SUPERNET['loss']['beta'], 
        'min_param_value' : CONFIG_SUPERNET['loss']['min_param_value'], 
        'max_param_size' : CONFIG_SUPERNET['loss']['max_param_size']
    }
    criterion = SupernetLoss(supernetloss_param).cuda()


    thetas_params = [param for name, param in model.named_parameters() if 'thetas' in name]
    params_except_thetas = [param for param in model.parameters() if not check_tensor_in_list(param, thetas_params)]

    w_optimizer = torch.optim.SGD(params=params_except_thetas,
                                  lr=CONFIG_SUPERNET['optimizer']['w_lr'],
                                  momentum=CONFIG_SUPERNET['optimizer']['w_momentum'],
                                  weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])

    theta_optimizer = torch.optim.Adam(params=thetas_params,
                                       lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                       weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])

    last_epoch = -1
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer,
                                                             T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'],
                                                             last_epoch=last_epoch)

    train_param = {
        'temperature' : CONFIG_SUPERNET['train_settings']['init_temperature'],
        'exp_anneal_rate' : CONFIG_SUPERNET['train_settings']['exp_anneal_rate'],  # apply it every epoch
        'cnt_epochs' : CONFIG_SUPERNET['train_settings']['cnt_epochs'],
        'train_thetas_from_the_epoch' : CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch'],
        'path_to_save_model' : CONFIG_SUPERNET['train_settings']['path_to_save_model'] 
    }
    trainer = TrainerSupernet(criterion, w_optimizer, theta_optimizer, w_scheduler, train_param)
    trainer.train_loop(train_w_loader, train_thetas_loader, test_loader, model)


if __name__ == "__main__":
    train_supernet()