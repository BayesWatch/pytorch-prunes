import random
import numpy as np
import torchvision.transforms as transforms
import torchvision
import time
from functools import reduce
from models import *
import random
import time
import operator
import torchvision
import torchvision.transforms as transforms

from models import *


class Pruner:
    def __init__(self, module_name='MaskBlock'):
        # First get vector of masks
        self.module_name = module_name
        self.masks = []
        self.prune_history = []

    def fisher_prune(self, model, prune_every):

        self._get_fisher(model)
        tot_loss = self.fisher.div(prune_every) + 1e6 * (1 - self.masks)  # dummy value for off masks
        print(len(tot_loss))
        min, argmin = torch.min(tot_loss, 0)
        self.prune(model, argmin.item())
        self.prune_history.append(argmin.item())

    def fixed_prune(self, model, ID):
        self.prune(model, ID)
        self.prune_history.append(ID)

    def random_prune(self, model):

        self._get_fisher(model)
        # Do this to update costs.
        masks = []
        for m in model.modules():
            if m._get_name() == self.module_name:
                masks.append(m.mask.detach())

        masks = self.concat(masks)
        masks_on = [i for i, v in enumerate(masks) if v == 1]
        random_pick = random.choice(masks_on)
        self.prune(model, random_pick)
        self.prune_history.append(random_pick)

    def l1_prune(self, model, prune_every):
        masks = []
        l1_norms = []

        for m in model.modules():
            if m._get_name() == 'MaskBlock':
                l1_norm = torch.sum(m.conv1.weight, (1, 2, 3)).detach().cpu().numpy()
                masks.append(m.mask.detach())
                l1_norms.append(l1_norm)

        masks = self.concat(masks)
        self.masks = masks
        l1_norms = np.concatenate(l1_norms)

        l1_norms_on = []
        for m, l in zip(masks, l1_norms):
            if m == 1:
                l1_norms_on.append(l)
            else:
                l1_norms_on.append(9999.)  # dummy value

        smallest_norm = min(l1_norms_on)
        pick = np.where(l1_norms == smallest_norm)[0][0]

        self.prune(model, pick)
        self.prune_history.append(pick)

    def prune(self, model, feat_index):
        print('Pruned %d out of %d channels so far' % (len(self.prune_history), len(self.masks)))
        if len(self.prune_history) > len(self.masks):
            raise Exception('Time to stop')
        """feat_index refers to the index of a feature map. This function modifies the mask to turn it off."""
        safe = 0
        running_index = 0
        for m in model.modules():
            if m._get_name() == self.module_name:
                mask_indices = range(running_index, running_index + len(m.mask))

                if feat_index in mask_indices:
                    print('Pruning channel %d' % feat_index)
                    local_index = mask_indices.index(feat_index)
                    m.mask[local_index] = 0
                    safe = 1
                    break
                else:
                    running_index += len(m.mask)
                    # print(running_index)
        if not safe:
            raise Exception('The provided index doesn''t correspond to any feature maps. This is bad.')

    def compress(self, model):
        for m in model.modules():
            if m._get_name() == 'MaskBlock':
                m.compress_weights()

    def _get_fisher(self, model):
        masks = []
        fisher = []

        self._update_cost(model)

        for m in model.modules():
            if m._get_name() == self.module_name:
                masks.append(m.mask.detach())
                fisher.append(m.running_fisher.detach())

                # Now clear the fisher cache
                m.reset_fisher()

        self.masks = self.concat(masks)
        self.fisher = self.concat(fisher)

    def _get_masks(self, model):
        masks = []

        for m in model.modules():
            if m._get_name() == self.module_name:
                masks.append(m.mask.detach())

        self.masks = self.concat(masks)

    def _update_cost(self, model):
        for m in model.modules():
            if m._get_name() == self.module_name:
                m.cost()

    def get_cost(self, model):
        params = 0
        for m in model.modules():
            if m._get_name() == self.module_name:
                m.cost()
                params += m.params
        return params

    @staticmethod
    def concat(input):
        return torch.cat([item for item in input])


def find(input):
    # Find as in MATLAB to find indices in a binary vector
    return [i for i, j in enumerate(input) if j]


def concat(input):
    return torch.cat([item for item in input])


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def get_error(output, target, topk=(1,)):
    """Computes the error@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_inf_params(net, verbose=True, sd=False):
    if sd:
        params = net
    else:
        params = net.state_dict()
    tot = 0
    conv_tot = 0
    for p in params:
        no = params[p].view(-1).__len__()

        if ('num_batches_tracked' not in p) and ('running' not in p) and ('mask' not in p):
            tot += no

            if verbose:
                print('%s has %d params' % (p, no))
        if 'conv' in p:
            conv_tot += no

    if verbose:
        print('Net has %d conv params' % conv_tot)
        print('Net has %d params in total' % tot)

    return tot


count_ops = 0
count_params = 0


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, x):
    global count_ops, count_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_learned_conv
    elif type_name in ['LearnedGroupConv']:
        measure_layer(layer.relu, x)
        measure_layer(layer.norm, x)
        conv = layer.conv
        out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
                    conv.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
                    conv.stride[1] + 1)
        delta_ops = conv.in_channels * conv.out_channels * conv.kernel_size[0] * \
                    conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
        delta_params = get_layer_param(conv) / layer.condense_factor

    ### ops_nonlinearity
    elif type_name in ['ReLU']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        print(delta_ops)
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    ### ops_nothing
    elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout']:
        delta_params = get_layer_param(layer)

    ### unknown layer type
    else:
        None
        # raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    return


def measure_model(model, H, W):
    global count_ops, count_params
    count_ops = 0
    count_params = 0
    data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(x):
                        measure_layer(m, x)
                        return m.old_forward(x)

                    return lambda_forward

                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops, count_params
