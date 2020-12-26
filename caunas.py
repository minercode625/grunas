import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import FileLogger
from torch.autograd import Variable
import numpy as np

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

def group_channels(channel_size, cluster):
    channel_div = int(channel_size / cluster)
    channel_mod = int(channel_size % cluster)

    channel_list = [channel_div for i in range(cluster)]
    channel_list[cluster - 1] += channel_mod

    return channel_list


class GroupConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride, cluster):
        super(GroupConv, self).__init__()


        self.in_planes = group_channels(in_planes, cluster)

        self.out_planes = group_channels(out_planes, cluster)

        self.convlist = nn.ModuleList([nn.Conv2d(self.in_planes[i], self.out_planes[i],
                                                 kernel_size=1, stride=stride, bias=False)
                                       for i in range(cluster)])
        self.indices = [0]
        for i in range(cluster):
            self.indices.append(self.indices[i] + self.in_planes[i])


    def forward(self, x):
        out_list = []
        for i in range(len(self.in_planes)):
            out_list.append(self.convlist[i](x[:, self.indices[i]:self.indices[i+1], :, :]))

        out = out_list[0]

        for i in range(1, len(out_list)):
            out = torch.cat((out, out_list[i]), dim=1)

        return out



def add(a, b):
    return a+b

def cat(a, b):
    return torch.cat((a, b), dim=1)



class ConvUnit(nn.Module):
    def __init__(self, in_planes, out_planes, stride, pad, cluster, stage):
        super(ConvUnit, self).__init__()
        self.cluster = cluster
        self.stride = stride
        self.pad = pad
        self.stage = stage

        expansion_rate = 2
        expansion_channels = int(in_planes * expansion_rate)
        # self.conv1x1 = GroupConv(in_planes, expansion_channels, stride=stride, cluster=cluster)
        self.conv1x1 = nn.Conv2d(in_planes, expansion_channels, kernel_size=1, stride=1,
                                    bias=False)
        # self.conv1x1 = GroupConv(in_planes, expansion_channels, stride=1, cluster=cluster)
        self.norm1 = nn.BatchNorm2d(expansion_channels)
        self.relu = nn.ReLU(inplace=True)
        self.depth_conv = nn.Conv2d(expansion_channels, expansion_channels, kernel_size=3, stride=stride, padding=pad,
                                    bias=False, groups=expansion_channels)

        if in_planes == out_planes:
            self.add_func = add
            self.group_conv = GroupConv(expansion_channels, out_planes, stride=1, cluster=cluster)
            self.norm2 = nn.BatchNorm2d(out_planes)
        else:
            self.add_func = cat
            self.group_conv = GroupConv(expansion_channels, int(out_planes / 2), stride=1, cluster=cluster)
            self.norm2 = nn.BatchNorm2d(int(out_planes / 2))

        self.flatten = Flatten()

    def forward(self, x):
        shortcut = x
        if self.stride == 2:
            shortcut = F.avg_pool2d(shortcut, kernel_size=3, stride=2, padding=1)
        out = self.conv1x1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.depth_conv(out)
        out = self.group_conv(out)
        out = self.norm2(out)
        out = self.add_func(shortcut, out)
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
    def __init__(self, in_planes, last_feature_size, cnt_classes):
        super(LastUnit, self).__init__()
        self.lastconv = nn.Conv2d(in_channels=in_planes,
                              out_channels=last_feature_size,
                              kernel_size=1, stride=1)
        self.flatten = Flatten()
        self.fc = nn.Linear(last_feature_size, out_features=cnt_classes)

    def forward(self, x):
        out = self.lastconv(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.flatten(out)
        out = self.fc(out)
        return out


class MixedOp(nn.Module):
    def __init__(self, layer_parameters, max_cluster_size, stage):
        super(MixedOp, self).__init__()
        self.stage = stage
        self.ops = nn.ModuleList([ConvUnit(*layer_parameters, i+1, stage)
                                  for i in range(max_cluster_size)])
        self.num_of_parameters = []
        for i in range(len(self.ops)):
            self.num_of_parameters.append(sum([p.numel() for p in self.ops[i].parameters() if p.requires_grad]))
        self.thetas = nn.Parameter(torch.Tensor([1.0 / float(max_cluster_size) for i in range(max_cluster_size)]))

    def forward(self, x, temperature, parameters_to_accumulate):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)

        output = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        num_of_parameters = sum(m * lat for m, lat in zip(soft_mask_variables, self.num_of_parameters))
        parameters_to_accumulate += num_of_parameters
        return output, parameters_to_accumulate

    def _get_max_group(self):
        return np.argmax(self.thetas.detach().cpu().numpy())

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class SuperNet(nn.Module):
    def __init__(self, supernet_param):
        super(SuperNet, self).__init__()
        
        self.layer_table = supernet_param['config_layer']
        self.max_cluster_size = supernet_param['max_cluster_size']
        self.first_inchannel = supernet_param['first_inchannel']
        self.last_feature_size = supernet_param['last_feature_size']
        self.cnt_classes = supernet_param['cnt_classes']
        self.first = FirstUnit(self.first_inchannel, self.layer_table[0][0])

        self.first_param = sum([p.numel() for p in self.first.parameters() if p.requires_grad])
        self.stages_to_search = nn.ModuleList([MixedOp(
            self.layer_table[layer_id],
            self.max_cluster_size, layer_id)
            for layer_id in range(len(self.layer_table))])

        self.last = LastUnit(self.layer_table[-1][1], self.last_feature_size, self.cnt_classes)

        self.last_param = sum([p.numel() for p in self.last.parameters() if p.requires_grad])

    def forward(self, x, temperature):
        parameters_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
        out = self.first(x)
        for mixed_op in self.stages_to_search:
            out, parameters_to_accumulate = mixed_op(out, temperature, parameters_to_accumulate)
        out = self.last(out)
        parameters_to_accumulate = self.first_param + self.last_param + parameters_to_accumulate
        return out, parameters_to_accumulate

    def get_max_group(self):
        group_max_str = []
        for mixed_op in self.stages_to_search:
           group_max_str.append(np.argmax(mixed_op.thetas.detach().cpu().numpy()))
        
        return " ".join(map(str, group_max_str))

class SupernetLoss(nn.Module):
    def __init__(self, supernetloss_param):
        super(SupernetLoss, self).__init__()
        self.alpha = supernetloss_param['alpha']
        self.beta = supernetloss_param['beta']

        self.weight_criterion = nn.CrossEntropyLoss()
        self.min_param_value = supernetloss_param['min_param_value']
        self.max_param_size = torch.tensor(supernetloss_param['max_param_size'], dtype=torch.float32).cuda()

    def forward(self, outs, targets, parameters):
        ce = self.weight_criterion(outs, targets)
        parameters = torch.max(parameters, self.max_param_size) / self.max_param_size
        loss = self.alpha * ce * (parameters ** self.beta)
        return loss, ce, parameters
