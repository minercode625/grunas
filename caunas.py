import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import CONFIG_SUPERNET
from utils import FileLogger

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class ConvUnit(nn.Module):
    def __init__(self, in_planes, out_planes, stride, pad, cluster, stage):
        super(ConvUnit, self).__init__()
        self.cluster = cluster
        self.stride = stride
        self.pad = pad
        self.stage = stage
        self.conv1x1 = nn.Conv2d(in_planes, int(out_planes/2), kernel_size=1, stride=stride, padding=pad, bias=False, groups=self.cluster)
        self.norm1 = nn.BatchNorm2d(int(out_planes/2))
        self.relu = nn.ReLU()
        self.depth_conv = nn.Conv2d(int(out_planes/2), int(out_planes/2), kernel_size=3, stride=stride, bias=False, groups=int(out_planes/2))
        self.group_conv = nn.Conv2d(int(out_planes/2), out_planes, kernel_size=1, stride=stride, bias=False, groups=self.cluster)
        self.norm2 = nn.BatchNorm2d(out_planes)
        self.flatten = Flatten()

    def forward(self, x):

        shortcut = x
        out = self.conv1x1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.depth_conv(out)
        out = self.group_conv(out)
        out = self.norm2(out)
        out = torch.cat((shortcut, out), 1)
        out = self.relu(out)
        out = channel_shuffle(out, self.cluster)
        return out


class FirstUnit(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FirstUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=3, stride=2, padding=3)
        self.norm = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.relu(out)
        return out


class LastUnit(nn.Module):
    def __init__(self, in_planes, out_features, cnt_classes):
        super(LastUnit, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_features, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.linear = nn.Linear(out_features, out_features=cnt_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.linear(out)
        return out


class MixedOp(nn.Module):
    def __init__(self, layer_parameters, max_cluster_size, stage):
        super(MixedOp, self).__init__()
        self.stage = stage
        #self.filelog = FileLogger()
        self.ops = nn.ModuleList([ConvUnit(*layer_parameters, i+1, stage)
                                  for i in range(max_cluster_size)])
        self.num_of_parameters = []
        for i in range(len(self.ops)):
            self.num_of_parameters.append(sum([p.numel() for p in self.ops[i].parameters() if p.requires_grad]))
        self.thetas = nn.Parameter(torch.Tensor([1.0 / float(max_cluster_size) for i in range(max_cluster_size)]))

    def forward(self, x, temperature, parameters_to_accumulate):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)
        #sl = soft_mask_variables.tolist()
        #self.filelog.write_loss(self.stage, sl)
        output = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        num_of_parameters = sum(m * lat for m, lat in zip(soft_mask_variables, self.num_of_parameters))
        parameters_to_accumulate += num_of_parameters
        return output, parameters_to_accumulate


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class SuperNet(nn.Module):
    def __init__(self, layer_table, max_cluster_size, cnt_classes=1000):
        super(SuperNet, self).__init__()

        self.first = FirstUnit(3, layer_table[0][0])

        self.first = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=3),
                                   nn.BatchNorm2d(24), nn.ReLU())
        self.first_param = sum([p.numel() for p in self.first.parameters() if p.requires_grad])
        self.stages_to_search = nn.ModuleList([MixedOp(
            layer_table[layer_id],
            max_cluster_size, layer_id)
            for layer_id in range(len(layer_table))])

        self.last = LastUnit(layer_table[-1][1] * 2, CONFIG_SUPERNET['train_settings']['last_feature_size'], cnt_classes)
        self.last_param = sum([p.numel() for p in self.last.parameters() if p.requires_grad])

    def forward(self, x, temperature, parameters_to_accumulate):
        out = self.first(x)
        for mixed_op in self.stages_to_search:
            out, parameters_to_accumulate = mixed_op(out, temperature, parameters_to_accumulate)
        out = self.last(out)
        parameters_to_accumulate = self.first_param + self.last_param + parameters_to_accumulate
        return out, parameters_to_accumulate


class SupernetLoss(nn.Module):
    def __init__(self):
        super(SupernetLoss, self).__init__()
        self.alpha = CONFIG_SUPERNET['loss']['alpha']
        self.beta = CONFIG_SUPERNET['loss']['beta']
        self.gamma = CONFIG_SUPERNET['loss']['gamma']

        self.weight_criterion = nn.CrossEntropyLoss()
        self.min_param_value = CONFIG_SUPERNET['loss']['min_param_value']
        #self.max_param_size = torch.log(torch.tensor(CONFIG_SUPERNET['loss']['max_param_size'], dtype=torch.float32))
        self.max_param_size = torch.tensor(CONFIG_SUPERNET['loss']['max_param_size'], dtype=torch.float32).cuda()
    def forward(self, outs, targets, parameters):
        ce = self.weight_criterion(outs, targets)
        # parameters = torch.log(parameters)
        # parameters = self.max_param_size * torch.tensor(math.exp(self.gamma * self.max_param_size)) \
        #              + parameters * torch.exp(self.gamma * parameters)
        # divider = torch.tensor(math.exp(self.gamma * self.max_param_size)) + torch.exp(self.gamma * parameters)
        # divider *= self.max_param_size

        # parameters = parameters / divider + self.min_param_value
        parameters = torch.max(parameters, self.max_param_size) / self.max_param_size + self.min_param_value

        loss = self.alpha * ce * (parameters ** self.beta)
        return loss