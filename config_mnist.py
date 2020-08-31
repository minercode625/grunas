import numpy as np

CONFIG_LAYER = [(24, 24, 1, 1),
                (48, 48, 1, 1),
                (96, 96, 1, 1),
                (192, 192, 1, 1)]

CONFIG_SUPERNET = {
    'gpu_settings' : {
        'gpu_ids' : [0]
    },
    'dataloading' : {
        'batch_size' : 50,
        'w_share_in_train' : 0.8,
        'path_to_save_data' : './mnist_data'
    },
    'optimizer' : {
        # SGD parameters for w
        'w_lr' : 0.1,
        'w_momentum' : 0.9,
        'w_weight_decay' : 1e-4,
        # Adam parameters for thetas
        'thetas_lr' : 0.01,
        'thetas_weight_decay' : 5 * 1e-4
    },
    'loss' : {
        'alpha' : 1,
        'beta' : 0.8,
        'gamma' : 10,
        'min_param_value' : 0.001,
        'max_param_size' : (10 ** 4) * 8
    },
    'train_settings' : {
        'cnt_epochs' : 200, # 90
        'train_thetas_from_the_epoch' : 20,
        'path_to_save_model' : 'logs/mnist/best_model.pth',
        # for Gumbel Softmax
        'init_temperature' : 5.0,
        'exp_anneal_rate' : np.exp(-0.045),
        # first, last config
        'last_feature_size' : 256,
        'first_inchannel' : 1,
        'first_outchannel' : CONFIG_LAYER[0][0],
        'last_inchannel' : CONFIG_LAYER[-1][0],
        'cnt_classes' : 10
    },
    'cluster' : {
        'max_cluster_size' : 4
    }
}




