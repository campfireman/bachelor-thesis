
import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from alpha_zero_general.Game import Game
from alpha_zero_general.NeuralNet import NeuralNet
from alpha_zero_general.utils import *
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm

from src.settings import CoachArguments


class NNetWrapper(NeuralNet):
    def __init__(self, game: Game, args: CoachArguments):
        self.nnet = AbaloneNNetTorchMini(game, args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        if self.args.cuda:
            self.nnet.cuda()

    def train(self, examples: List[Tuple[npt.NDArray, npt.NDArray, float]]):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(
                    len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(
                    ), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board: npt.NDArray):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if self.args.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets: List[npt.NDArray], outputs: List[npt.NDArray]):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets: List[float], outputs: List[float]):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar', full_path: str = None):
        filepath = full_path
        if full_path is None:
            filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder: str = 'checkpoint', filename: str = 'checkpoint.pth.tar', full_path: str = None):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = full_path
        if full_path is None:
            filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])


class TorchNNet(nn.Module):
    def show_info(self):
        from prettytable import PrettyTable

        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")


class ConvBlock(nn.Module):
    def __init__(self, board_x: int, board_y: int, action_size: int):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.board_x = board_x
        self.board_y = board_y
        self.action_size = action_size

    def forward(self, s: torch.Tensor):
        # batch_size x channels x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes: int = 256, planes: int = 256, stride: int = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self, board_x: int, board_y: int, action_size: int):
        super(OutBlock, self).__init__()
        self.board_x = board_x
        self.board_y = board_y
        self.action_size = action_size
        self.conv = nn.Conv2d(256, 3, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(1*self.board_x*self.board_y, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(256, 2, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(self.board_x*self.board_y*2, self.action_size)

    def forward(self, s: torch.Tensor):
        v = F.relu(self.bn(self.conv(s)))  # value head
        # batch_size X channel X height X width
        v = v.view(-1, 1*self.board_x*self.board_y)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, self.board_x*self.board_y*2)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class AbaloneNNetTorch(TorchNNet):
    def __init__(self, game: Game, args: CoachArguments):
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        super(AbaloneNNetTorch, self).__init__()
        self.conv = ConvBlock(self.board_x, self.board_y, self.action_size)
        for block in range(self.args.residual_tower_size):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock(self.board_x, self.board_y, self.action_size)

    def forward(self, s: torch.Tensor):
        s = self.conv(s)
        for block in range(self.args.residual_tower_size):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s


class AbaloneNNetTorchMini(TorchNNet):
    def __init__(self, game: Game, args: CoachArguments):
        # game params
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()
        self.args = args

        super(AbaloneNNetTorchMini, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(
            args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s: torch.Tensor):
        #                                                           s: batch_size x board_x x board_y
        # batch_size x 1 x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))
        # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn3(self.conv3(s)))
        # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels *
                   (self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 512

        # batch_size x action_size
        pi = self.fc3(s)
        # batch_size x 1
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
