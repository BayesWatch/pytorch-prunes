"""This script just trains models from scratch, to later be pruned"""

import argparse
import json
import os
import time
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.model_zoo as model_zoo

from models import *

from funcs import *


parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--GPU', default='0', type=str, help='GPU to use')
parser.add_argument('--save_file', default='saveto', type=str, help='save file for checkpoints')
parser.add_argument('--base_file', default='bbb', type=str, help='base file for checkpoints')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--data_loc', default='/disk/scratch/datasets/cifar')

# Learning specific arguments
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-lr', '--learning_rate', default=.1, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-epochs', '--no_epochs', default=200, type=int, metavar='epochs', help='no. epochs')
parser.add_argument('--epoch_step', default='[60,120,160]', type=str, help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay factor')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay')
parser.add_argument('--eval', '-e', action='store_true', help='resume from checkpoint')
parser.add_argument('--mask', '-m', type=int, help='mask mode', default=0)
parser.add_argument('--deploy', '-de', action='store_true', help='prune and deploy model')
parser.add_argument('--params_left', '-pl', default=0, type=int, help='prune til...')
parser.add_argument('--net', choices=['res', 'dense'], default='res')

# Net specific
parser.add_argument('--depth', '-d', default=40, type=int, metavar='D', help='depth of wideresnet/densenet')
parser.add_argument('--width', '-w', default=2.0, type=float, metavar='W', help='width of wideresnet')
parser.add_argument('--growth', default=12, type=int, help='growth rate of densenet')
parser.add_argument('--transition_rate', default=0.5, type=float, help='transition rate of densenet')


# Uniform bottlenecks
parser.add_argument('--bottle', action='store_true', help='Linearly scale bottlenecks')
parser.add_argument('--bottle_mult', default=0.5, type=float, help='bottleneck multiplier')


if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.net == 'res':
    if not args.bottle:
        model = WideResNet(args.depth, args.width, mask=args.mask)
    else:
        model = WideResNetBottle(args.depth, args.width, bottle_mult=args.bottle_mult)
elif args.net == 'dense':
    if not args.bottle:
        model = DenseNet(args.growth, args.depth, args.transition_rate, 10, True, mask=args.mask)
    else:
        model = DenseNet(args.growth, args.depth, args.transition_rate, 10, True, width=args.bottle_mult)

else:
    raise ValueError('pick a valid net')

pruner = Pruner()

if args.deploy:
    # Feed example to activate masks
    model(torch.rand(1, 3, 32, 32))
    SD = torch.load('checkpoints/%s.t7' % args.base_file)

    if not args.eval:

        pruner = Pruner()
        pruner._get_masks(model)

        for ii in SD['prune_history']:
            pruner.fixed_prune(model, ii)

    else:
        model.load_state_dict(SD['state_dict'])

pruner.compress(model)

get_inf_params(model)
time.sleep(1)
model.to(device)

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

print('==> Preparing data..')
num_classes = 10

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                      (4, 4, 4, 4), mode='reflect').squeeze()),
    transforms.ToPILImage(),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    normalize,

])

trainset = torchvision.datasets.CIFAR10(root=args.data_loc,
                                        train=True, download=True, transform=transform_train)
valset = torchvision.datasets.CIFAR10(root=args.data_loc,
                                      train=False, download=True, transform=transform_val)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=args.workers,
                                          pin_memory=False)
valloader = torch.utils.data.DataLoader(valset, batch_size=50, shuffle=False,
                                        num_workers=args.workers,
                                        pin_memory=False)

error_history = []
epoch_step = json.loads(args.epoch_step)


def train():
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))




def validate():
    global error_history

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(valloader):

        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = get_error(output.detach(), target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(valloader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Error@1 {top1.avg:.3f} Error@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))


    # Record Top 1 for CIFAR
    error_history.append(top1.avg)


if __name__ == '__main__':

    filename = 'checkpoints/%s.t7' % args.save_file
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio)

    if not args.eval:

        for epoch in range(args.no_epochs):
            scheduler.step()

            print('Epoch %d:' % epoch)
            print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])
            # train for one epoch
            train()
            # # evaluate on validation set
            validate()

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'error_history': error_history,
            }, filename=filename)

    else:
        if not args.deploy:
            model.load_state_dict(torch.load(filename)['state_dict'])
        epoch = 0
        validate()
