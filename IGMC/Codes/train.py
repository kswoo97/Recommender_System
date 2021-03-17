import time
import os
import math
import multiprocessing as mp
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from tqdm import tqdm
import pdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)

def train (model, optimizer, loader, device, regression = True, ARR = 0, show_progress = False, epoch = None) :
    model.train()
    train_loss = 0
    total_loss = 0
    if show_progress :
        pbar = tqdm(loader)
    else :
        pbar = loader
    for data in pbar :
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        if regression : # 만약 일반적인 선형 회귀라면 MSE를 loss로 사용한다.
            loss = F.mse_loss(out, data.y.view(-1))
        else :
            loss = F.nll_loss(out, data.y.view(-1))
        if show_progress :
            pbar.set_description('Epoch {}, batch loss: {}'.format(epoch, loss.item()))
        if ARR != 0 :
            for gconv in model.convs :
                w = torch.matmul(
                    gconv.comp,
                    gconv.weight.view(gconv.num_bases, -1)
                ).view(gconv.num_relations, gconv.in_channels, gconv.out_channels)
                reg_loss = torch.sum((w[1:, :, :] - w[:-1, :, :])**2)
                loss += ARR * reg_loss
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        torch.cuda.empty_cache()
    return total_loss / len(loader.dataset)

def eval_loss(model, loader, device, regression=False, show_progress=False) :
    model.eval()
    loss = 0
    if show_progress:
        print('Testing begins...')
        pbar = tqdm(loader)
    else:
        pbar = loader
    for data in pbar:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        if regression:
            loss += F.mse_loss(out, data.y.view(-1), reduction='sum').item()
        else:
            loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        torch.cuda.empty_cache()
    return loss / len(loader.dataset)

def train_multiple_epochs(train_dataset,
                          test_dataset,
                          model, epochs, batch_size, lr,
                          lr_decay_factor, lr_decay_step_size,
                          weight_decay, ARR = 0,
                          test_freq = 1, logger = None,
                          continue_from = None, res_dir = None,
                          multiple_gpu = False, save_interval = 5) :

    rmses = []

    if train_dataset.__class__.__name__ == 'MyDynamicDataset' :
        num_workers = mp.cpu_count()
    else :
        num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size, shuffle = True,
                              num_workers = num_workers)
    if test_dataset.__class__.__name__ == 'MyDynamicDataset':
        num_workers = mp.cpu_count()
    else:
        num_workers = 2
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False,
                             num_workers=num_workers)

    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    start_epoch = 1
    if  continue_from is not None :
        # 학습을 중간부터 다시 재개한다.
        model.load_state_dict(
            torch.load(os.path.join(res_dir, 'model_checkpoint{}.pth'.format(continue_from)))
        )
        optimizer.load_state_dict(
            torch.load(os.path.join(res_dir, 'optimizer_checkpoint{}.pth'.format(continue_from)))
        )
        start_epoch = continue_from + 1
        epochs -= continue_from

    # 재용이형 피드백 받아서, GPU가 여러개가 아니라면 아래 인자는 실행되지 않음
    # 본 논문에서는 이 인지가 수행됨

    if multiple_gpu :
        torch.cuda.synchronize()

    batch_pbar = len(train_dataset) >= 100000
    t_start = time.perf_counter()
    if not batch_pbar:
        pbar = tqdm(range(start_epoch, epochs + start_epoch))
    else:
        pbar = range(start_epoch, epochs + start_epoch)

    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader, device, regression=True, ARR=ARR,
                           show_progress=batch_pbar, epoch=epoch)
        if epoch % test_freq == 0:
            rmses.append(eval_rmse(model, test_loader, device, show_progress=batch_pbar))
        else:
            rmses.append(np.nan)
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'test_rmse': rmses[-1],
        }
        if not batch_pbar:
            pbar.set_description(
                'Epoch {}, train loss {:.6f}, test rmse {:.6f}'.format(*eval_info.values())
            )
        else:
            print('Epoch {}, train loss {:.6f}, test rmse {:.6f}'.format(*eval_info.values()))

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        if logger is not None:
            logger(eval_info, model, optimizer, res_dir, save_interval)

    if multiple_gpu :
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    duration = t_end - t_start

    print('Final Test RMSE: {:.6f}, Duration: {:.6f}'.
          format(rmses[-1],
                 duration))

    return rmses[-1]

def test_once(test_dataset,
              model,
              batch_size,
              logger=None,
              ensemble=False,
              checkpoints=None):

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    model.to(device)
    t_start = time.perf_counter()
    if ensemble and checkpoints:
        rmse = eval_rmse_ensemble(model, checkpoints, test_loader, device, show_progress=True)
    else:
        rmse = eval_rmse(model, test_loader, device, show_progress=True)
    t_end = time.perf_counter()
    duration = t_end - t_start
    print('Test Once RMSE: {:.6f}, Duration: {:.6f}'.format(rmse, duration))
    epoch_info = 'test_once' if not ensemble else 'ensemble'
    eval_info = {
        'epoch': epoch_info,
        'train_loss': 0,
        'test_rmse': rmse,
        }
    if logger is not None:
        logger(eval_info, None, None)
    return rmse

def eval_loss_ensemble(model, checkpoints, loader, device, regression=False, show_progress=False):
    loss = 0
    Outs = []
    for i, checkpoint in enumerate(checkpoints):
        if show_progress:
            print('Testing begins...')
            pbar = tqdm(loader)
        else:
            pbar = loader
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        outs = []
        if i == 0:
            ys = []
        for data in pbar:
            data = data.to(device)
            if i == 0:
                ys.append(data.y.view(-1))
            with torch.no_grad():
                out = model(data)
                outs.append(out)
        if i == 0:
            ys = torch.cat(ys, 0)
        outs = torch.cat(outs, 0).view(-1, 1)
        Outs.append(outs)
    Outs = torch.cat(Outs, 1).mean(1)
    if regression:
        loss += F.mse_loss(Outs, ys, reduction='sum').item()
    else:
        loss += F.nll_loss(Outs, ys, reduction='sum').item()
    torch.cuda.empty_cache()
    return loss / len(loader.dataset)


def eval_rmse_ensemble(model, checkpoints, loader, device, show_progress=False):
    mse_loss = eval_loss_ensemble(model, checkpoints, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse

def eval_rmse(model, loader, device, show_progress=False):
    mse_loss = eval_loss(model, loader, device, True, show_progress)
    rmse = math.sqrt(mse_loss)
    return rmse




