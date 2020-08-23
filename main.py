import numpy as np
import torch
from torch import nn
import os

from dataloaders import get_loaders, get_test_loader
from config import CONFIG_LAYER, CONFIG_SUPERNET
from utils import weights_init, check_tensor_in_list, FileLogger, accuracy
from caunas import SuperNet, SupernetLoss
from train_supernet import TrainerSupernet


def train_supernet():
    manual_seed = 1
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.benchmark = True

    #### DataLoading
    train_w_loader, train_thetas_loader = get_loaders(CONFIG_SUPERNET['dataloading']['w_share_in_train'],
                                                      CONFIG_SUPERNET['dataloading']['batch_size'],
                                                      CONFIG_SUPERNET['dataloading']['path_to_save_data'],
                                                      )
    test_loader = get_test_loader(CONFIG_SUPERNET['dataloading']['batch_size'],
                                  CONFIG_SUPERNET['dataloading']['path_to_save_data'])

    #### Model
    model = SuperNet(CONFIG_LAYER, CONFIG_SUPERNET['cluster']['max_cluster_size'], cnt_classes=10).cuda()
    model = model.apply(weights_init)
    model = nn.DataParallel(model, device_ids=[0])

    #### Loss, Optimizer and Scheduler
    criterion = SupernetLoss().cuda()


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

    #### Training Loop
    trainer = TrainerSupernet(criterion, w_optimizer, theta_optimizer, w_scheduler)
    trainer.train_loop(train_w_loader, train_thetas_loader, test_loader, model)

def sample_architecture_from_the_supernet():
    test_loader = get_test_loader(CONFIG_SUPERNET['dataloading']['batch_size'],
                                  CONFIG_SUPERNET['dataloading']['path_to_save_data'])
    model = SuperNet(CONFIG_LAYER, CONFIG_SUPERNET['cluster']['max_cluster_size'], cnt_classes=10).cuda()
    model = nn.DataParallel(model, device_ids=[0])

    model.load_state_dict(torch.load(CONFIG_SUPERNET['train_settings']['path_to_save_model']))
    module_list = []
    module_list.append(model.module.first)
    for layer in model.module.stages_to_search:
        print(str(np.argmax(layer.thetas.detach().cpu().numpy())))
        module_list.append(layer.ops[np.argmax(layer.thetas.detach().cpu().numpy())])
    
    module_list.append(model.module.last)

    sampled_model = nn.Sequential(*module_list).cuda()

    prec1_list = []
    prec3_list = []

    with torch.no_grad():
        for step, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            N = X.shape[0]

            outs = model(X, 0 ,0)
            prec1, prec3 = accuracy(outs, y, topk=(1, 5))
            prec1_list.append(prec1)
            prec3_list.append(prec3)

    prec1_avg = sum(prec1_list) / len(prec1_list)
    prec3_avg = sum(prec3_list) / len(prec3_list)



if __name__ == "__main__":
    train_supernet()