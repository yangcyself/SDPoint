# -^- coding:utf-8 -^- 
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
from agent import Agent


class dsENV:
    """
    The API contains the init and step and validation, which gives a total reward
    """
    def __init__(self,model,val_loader,criterion):
        self.dsrate = []
        self.model = model
        self.currentShape=100
        self.val_loader = val_loader
        self.criterion = criterion
        self.layers = self.getshape()

    def flops(self):
        return flops.calculate(self.model,stochastic = False)

    def getshape(self):
        shape = []
        for b in self.model.module.allblocks:
            shape.append(b.shape)
        return shape


    def getState(self):
        """
        state contains the shapes of the network layers
            the current layer
            the current featuremap shape
        """
        cl = len(dsrate)
        return np.array([cl,*(self.layers(cl)),self.currentShape])

    def reset(self):
        """
        reset the downsampling rates
        """
        self.dsrate = []
        self.currentShape=100
        return getState()
    def step(self,action):
        """
        returns state, done
        """
        self.dsrate.append(action)
        self.currentShape *= action
        done = len(self.dsrate) ==len(self.layers)
        return self.etState(),done
        
    def validation(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.model.eval()
        self.model.setDSRate(self.dsrate)
        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.val_loader):
                input = input.cuda()
                target = target.cuda(non_blocking=True)

                # compute output
                output = model(input, blockID=blockID, ratio=ratio)
                loss = self.scriterion(output, target)

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

    def final_score(self):
        """
        calculate the final reward score
        """
        top1,top5 = self.validation()
        return top1-(1e-9)*self.flops


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




def train(
    seed=None,
    total_timesteps=100000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    
    batch_size=32,
    print_freq=100,
    checkpoint_freq=10000,
    
    learning_starts=1000,
    gamma=1.0,
    prioritized_replay=False,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta0=0.4,
    prioritized_replay_beta_iters=None,
    prioritized_replay_eps=1e-6,
    param_noise=False,
    **network_kwargs
        ):
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                    initial_p=1.0,
                                    final_p=exploration_final_eps)

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    tempbuff = []
    for t in range(total_timesteps):
        # Take action and update exploration to the newest value
        kwargs = {}
        if not param_noise:
            update_eps = exploration.value(t)
            update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        action = agent.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
        env_action = action
        reset = False
        new_obs,  done= env.step(env_action)
        tempbuff.append(obs, action, new_obs, done)
        obs = new_obs
        
        if done:
            obs = env.reset()
            final_r = env.final_score()
            episode_rewards[-1] = final_r
            episode_rewards.append(0.0)
            reset = True
            agent.learn_step(tempbuff,final_r,t)
            tempbuff = []

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)

        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            logger.record_tabular("steps", t)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
            logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
            logger.dump_tabular()

        if (checkpoint_freq is not None and t > learning_starts and
                num_episodes > 100 and t % checkpoint_freq == 0):
            if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                if print_freq is not None:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                saved_mean_reward, mean_100ep_reward))
                agent.save_variables()
                model_saved = True
                saved_mean_reward = mean_100ep_reward

    # if model_saved:
    #     if print_freq is not None:
    #     logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
    # load_variables(model_file)



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training - Stochastic Downsampling')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnext101',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: preresnet101)')

parser.add_argument("--train_path", type=str, default="~/dataset/vgg100")
parser.add_argument("--test_path", type=str, default="~/dataset/vgg100")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-vb', '--val-batch-size', default=1024, type=int,
                    metavar='N', help='validation mini-batch size (default: 1024)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
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


if __name__ == "__main__":

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=100)

    model = torch.nn.DataParallel(model).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    criterion = nn.CrossEntropyLoss().cuda()
    dataset=DataSet(torch_v=args.torch_version)
    val_loader = dataset.test_loader(args.test_path)

    env = dsENV(model,val_loader,criterion)
    agent = Agent(
        2,#action_space
        )