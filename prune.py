"""Pruning script"""

import argparse
import os

import torch.utils.model_zoo as model_zoo

from funcs import *
from models import *


parser = argparse.ArgumentParser(description='Pruning')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--GPU', default='0', type=str, help='GPU to use')
parser.add_argument('--save_file', default='wrn16_2_p', type=str, help='save file for checkpoints')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--resume_ckpt', default='checkpoint', type=str,
                    help='save file for resumed checkpoint')
parser.add_argument('--data_loc', default='/disk/scratch/datasets/cifar', type=str, help='where is the dataset')

# Learning specific arguments
parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd', type=str, help='optimizer')
parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-lr', '--learning_rate', default=8e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-epochs', '--no_epochs', default=1300, type=int, metavar='epochs', help='no. epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, metavar='W', help='weight decay')
parser.add_argument('--prune_every', default=100, type=int, help='prune every X steps')
parser.add_argument('--save_every', default=100, type=int, help='save model every X EPOCHS')
parser.add_argument('--random', default=False, type=bool, help='Prune at random')
parser.add_argument('--base_model', default='base_model', type=str, help='basemodel')
parser.add_argument('--val_every', default=1, type=int, help='val model every X EPOCHS')
parser.add_argument('--mask', default=1, type=int, help='Mask type')
parser.add_argument('--l1_prune', default=False, type=bool, help='Prune via l1 norm')
parser.add_argument('--net', default='dense', type=str, help='dense, res')
parser.add_argument('--width', default=2.0, type=float, metavar='D')
parser.add_argument('--depth', default=40, type=int, metavar='W')
parser.add_argument('--growth', default=12, type=int, help='growth rate of densenet')
parser.add_argument('--transition_rate', default=0.5, type=float, help='transition rate of densenet')

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

device = torch.device("cuda:%s" % '0' if torch.cuda.is_available() else "cpu")


if args.net == 'res':
    model = WideResNet(args.depth, args.width, mask=args.mask)
elif args.net =='dense':
    model = DenseNet(args.growth, args.depth, args.transition_rate, 10, True, mask=args.mask)

model.load_state_dict(torch.load('checkpoints/%s.t7' % args.base_model, map_location='cpu')['state_dict'], strict=True)

if args.resume:
    state = torch.load('checkpoints/%s.t7' % args.resume_ckpt, map_location='cpu')

    model = resume_from(state, model_type=args.net)
    error_history = state['error_history']
    prune_history = state['prune_history']
    flop_history = state['flop_history']
    param_history = state['param_history']
    start_epoch = state['epoch']

else:

    error_history = []
    prune_history = []
    param_history = []
    start_epoch = 0

model.to(device)

normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)

print('==> Preparing data..')
num_classes = 10

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normTransform
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    normTransform

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

prune_count = 0
pruner = Pruner()
pruner.prune_history = prune_history

NO_STEPS = args.prune_every


def finetune():
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    dataiter = iter(trainloader)

    for i in range(0, NO_STEPS):

        try:
            input, target = dataiter.next()
        except StopIteration:
            dataiter = iter(trainloader)
            input, target = dataiter.next()

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
            print('Prunepoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Error@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, NO_STEPS, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))




def prune():
    print('Pruning')
    if args.random is False:
        if args.l1_prune is False:
            print('fisher pruning')
            pruner.fisher_prune(model, prune_every=args.prune_every)
        else:
            print('l1 pruning')
            pruner.l1_prune(model, prune_every=args.prune_every)
    else:
        print('random pruning')
        pruner.random_prune(model, )


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

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(start_epoch, args.no_epochs):

        print('Epoch %d:' % epoch)
        print('Learning rate is %s' % [v['lr'] for v in optimizer.param_groups][0])

        # finetune for one epoch
        finetune()
        # # evaluate on validation set
        if epoch != 0 and ((epoch % args.val_every == 0) or (epoch + 1 == args.no_epochs)):  # Save at last epoch!
            validate()

            # Error history is recorded in validate(). Record params here
            no_params = pruner.get_cost(model)
            param_history.append(no_params)

        # Save before pruning
        if epoch != 0 and ((epoch % args.save_every == 0) or (epoch + 1 == args.no_epochs)):  #
            filename = 'checkpoints/%s_%d_prunes.t7' % (args.save_file, epoch)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'error_history': error_history,
                'param_history': param_history,
                'prune_history': pruner.prune_history,
            }, filename=filename)

        ## Prune
        prune()

