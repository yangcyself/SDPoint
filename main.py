import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
# import torch.utils.data
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import models
import utils.flops as flops
from predictor import Predictor

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training - Stochastic Downsampling')
parser.add_argument('--arch', '-a', metavar='ARCH', default='preresnet101',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: preresnet101)')

parser.add_argument("--train_path", type=str, default="~/dataset/vgg100")
parser.add_argument("--test_path", type=str, default="~/dataset/vgg100")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=115, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-vb', '--val-batch-size', default=1024, type=int,
                    metavar='N', help='validation mini-batch size (default: 1024)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-o', '--nodownsample', dest='nodownsample', action='store_true',
                    help='turn off the down sampling function of the model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('-m','--message', default='', type=str,
                    help='the message used for naming the outputfile')
parser.add_argument('--val-results-path', default='val_results.txt', type=str,
                    help='filename of the file for writing validation results')
parser.add_argument("--torch_version", dest="torch_version", action="store", type=float, default=0.4)

best_prec1 = 0

args = parser.parse_args()

class DataSet:
    def __init__(self, torch_v=0.4):
        self.torch_v = torch_v

    def loader(self, path, batch_size=32, num_workers=4, pin_memory=True,valid_size=0.1):
        '''normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'''
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if self.torch_v == 0.3:
            resize = transforms.RandomSizedCrop(224)
        else:
            resize = transforms.RandomResizedCrop(224)

        traindata_transforms = transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        return data.DataLoader(
                dataset=datasets.CIFAR100(root=path, train=True, download=True, transform=traindata_transforms),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory)
            

    def test_loader(self, path, batch_size=32, num_workers=4, pin_memory=True):
        '''normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])'''
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if self.torch_v == 0.3:
            resize = transforms.Scale(256)
        else:
            resize = transforms.Resize(256)
        testdata_transforms = transforms.Compose([
            resize,
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        return data.DataLoader(
            dataset=datasets.CIFAR100(root=path, train=False, download=True, transform=testdata_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory)




def main():
    global args, best_prec1
    # args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=100,nodownsample = args.nodownsample)

    if not args.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    if args.distributed:
        train_sampler = data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.evaluate:
        args.batch_size = args.val_batch_size

    dataset=DataSet(torch_v=args.torch_version)
    train_loader = dataset.loader(args.train_path)
    val_loader = dataset.test_loader(args.test_path)

    trainPredictor(val_loader,model,criterion,10,0.75)
    return 

    if args.evaluate:
        model.eval()
        val_results_file = open(args.val_results_path, 'w')
        val_results_file.write('blockID\tratio\tflops\ttop1-acc\ttop5-acc\t\n')
        if(not args.nodownsample):
            for i in [-1] + [model.module.blockID] + list(range(model.module.blockID)):
                for r in [0.5, 0.75]:
                    model_flops = flops.calculate(model, i, r)
                    top1, top5 = validate(train_loader, val_loader, model, criterion, i, r)
                    val_results_file.write('{0}\t{1}\t{2}\t{top1:.3f}\t{top5:.3f}\n'.format(
                                            i if i>-1 else 'nil', r if i>-1 else 'nil',
                                            model_flops, top1=top1, top5=top5))
                    if i == -1:
                        break
        else:
            i = -1
            model_flops = flops.calculate(model, i, 1)
            top1, top5 = validate(train_loader, val_loader, model, criterion, i, 1)
            val_results_file.write('{0}\t{1}\t{2}\t{top1:.3f}\t{top5:.3f}\n'.format(
                                    i if i>-1 else 'nil', r if i>-1 else 'nil',
                                    model_flops, top1=top1, top5=top5))

        val_results_file.close()
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        })


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

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
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(train_loader, val_loader, model, criterion, blockID, ratio):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    with torch.no_grad():
        end = time.time()
        for i, (input, _) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.cuda()

            # compute output
            output = model(input, blockID=blockID, ratio=ratio)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Iteration: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                       i, len(train_loader), batch_time=batch_time,
                       data_time=data_time))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input, blockID=blockID, ratio=ratio)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def trainPredictor(val_loader, model,criterion,blockID, ratio):
    predictor = Predictor(model.module.allblocks[blockID].inplanes)
    predictor = predictor.cuda()
    optimizer = torch.optim.SGD(predictor.parameters(),
                                0.1, momentum=0.9,
                                weight_decay=1e-4)

    stor = []
    losses = AverageMeter()

    def hook(m,inp,outp,stor):
        stor.append(inp)
    hookHandle = model.module.allblocks[blockID].register_forward_hook(lambda m,i,o: hook(m,i,o,stor))
    model.eval()
    predictor.train()
    for i, (feainput, featarget) in enumerate(val_loader):
        feainput = feainput.cuda()
        featarget = featarget.cuda(non_blocking=True)

        # compute output
        dsoutput = model(feainput, blockID=blockID, ratio=ratio)
        dsloss = nn.functional.cross_entropy(dsoutput, featarget,reduction = 'none')

        orioutput = model(feainput, blockID=None, ratio=None,downSample = False)#.detach()
        oriloss = nn.functional.cross_entropy(orioutput, featarget,reduction = 'none')

        target = oriloss/dsloss
        target = target.cuda()
        input = stor.pop()[0]
        input = input.cuda()
        pred = predictor(input)
        loss = nn.functional.mse_loss(pred,target)
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(i,loss)
        # print("oneEPOCH")
        if i % args.print_freq == 0:
            print('Epoch: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   i, len(val_loader),
                   loss=losses))
    hookHandle.remove()



def save_checkpoint(state, filename='{}_{}.checkpoint.pth.tar'.format(args.arch,args.message)):
    torch.save(state, filename)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after the 40th, 75th, and 105th epochs"""
    lr = args.lr
    for e in [40,75,105]:
        if epoch >= e:
            lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
