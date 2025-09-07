import os
import sys
import argparse
import numpy as np
from datetime import datetime, timedelta
# from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
sys.path.append('../../')
from stgcn_traffic_prediction.dataloader.milano_crop2 import load_data
from stgcn_traffic_prediction.models.model2 import T_STGCN
from stgcn_traffic_prediction.utils.lr_scheduler import LR_Scheduler
from stgcn_traffic_prediction.utils.parser2 import getparse
from stgcn_traffic_prediction.utils.metrics import getmetrics
from stgcn_traffic_prediction.utils.show import plot
import time
import csv

torch.manual_seed(22)
opt = getparse()
print(opt)
opt.save_dir = '{}/{}'.format(opt.save_dir, opt.traffic)
if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)
opt.model_filename = '{}/flow={}-close={}-period={}-trend={}-spatial={}-mode={}-c={}-s={}-FS={}-model_N={}-scptmodel_d={}-{}-{}-{}'.format(
                    opt.save_dir, opt.flow, opt.close_size,opt.period_size,opt.trend_size,opt.spatial,opt.mode,opt.c,opt.s,opt.FS,opt.model_N,
                    opt.s_model_d,opt.c_model_d,opt.p_model_d,opt.t_model_d)



print('Saving to ' + opt.model_filename)


se = 0

best_model = opt.model_filename+'.model'
print(best_model)
if os.path.exists(best_model):
    saved = torch.load(best_model, weights_only=False)
    se = saved['epoch']+1
    opt.best_valid_loss = saved['valid_loss'][-1]
    lr = saved['lr']

lr = 0.001
if opt.lr is not None:
    lr = opt.lr
if opt.se is not None:
    se = opt.se
total_epochs = se + opt.epoch_size


def log(fname, s):
    if not os.path.isdir(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()

def train_epoch(data_type,epoch):
    total_loss = 0
    if data_type == 'train':
        model.train()
        data = train_loader
    if data_type == 'valid':
        model.eval()
        data = valid_loader
    i = 0
    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
        for idx, (c, p, t, target) in enumerate(data):
            #print(idx,c,p,t,target)
            if data_type == 'train':
                scheduler(optimizer,i,epoch)
            optimizer.zero_grad()
            model.zero_grad()
            pred = model(c.float(),opt.mode,opt.c,opt.s,opt.c_t,opt.FS,opt.s_t,opt.flow,p.float(),t.float())
            loss = criterion(pred.float(),target.cuda().float()[:,:,opt.flow])
           
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            i += 1
    elif (opt.period_size > 0) & (opt.close_size > 0):
        for idx, (c, p, target) in enumerate(data):
            #print(idx,c,p,t,target)
            if data_type == 'train':
                scheduler(optimizer,i,epoch)
            optimizer.zero_grad()
            model.zero_grad()
            pred = model(c.float(),opt.mode,opt.c,opt.s,opt.FS,opt.c_t,opt.s_t,opt.flow,p.float())
            loss = criterion(pred.float(), target.cuda().float()[:,:,opt.flow])
            #print(loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            i += 1
    return total_loss/len(data)


def train():
    best_valid_loss = opt.best_valid_loss
    train_loss, valid_loss = [], []
    train_loss_last_values = []
    for i in range(se,total_epochs):
        train_loss.append(train_epoch('train',i))
        valid_loss.append(train_epoch('valid',i))
        train_loss_last_values.append(train_loss[-1])

        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]

            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss,'lr':optimizer.param_groups[0]['lr']}, opt.model_filename + '.model')
            #torch.save(optimizer, opt.model_filename + '.optim')
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.8f}, valid_loss: {:0.8f}, '
                      'best_valid_loss: {:0.8f}, lr: {:0.8f}').format((i + 1), total_epochs,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      best_valid_loss,
                                                                      optimizer.param_groups[0]['lr'])
        for name, parms in model.named_parameters():
            if parms.grad is not None:
                print('-->name:', name, '-->grad_requirs:', parms.requires_grad)
                print('--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
        print(log_string)
        log(opt.model_filename + '.log', log_string)
    plt.figure(figsize=(10, 6))
    plt.plot(range(se + 1, total_epochs + 1), train_loss_last_values, label='Train Loss (-1)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss (-1) vs Epoch')
    plt.legend()
    plt.savefig(opt.model_filename + '_train_loss_last_plot.png')
    plt.show()
def predict(test_type='train'):
    predictions = []
    ground_truth = []
    test = []
    loss = []
    best_model = torch.load(opt.model_filename + '.model', weights_only=False).get('model')
    # best_model = model

    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader
    i=0
    if (opt.period_size > 0) & (opt.close_size > 0) & (opt.trend_size > 0):
        for idx, (c, p, t, target) in enumerate(data):
            # pred = best_model(c.float(),opt.mode,opt.c,opt.s,opt.c_t,opt.s_t,opt.flow,p.float(),t.float())
            pred = torch.relu(best_model(c.float(),opt.mode,opt.c,opt.s,opt.FS,opt.c_t,opt.s_t,opt.flow,p.float(),t.float()))
            predictions.append(pred.float().data.cpu().numpy())
            ground_truth.append(target.float().numpy()[:,:,opt.flow])
            loss.append(criterion(pred.float(), target.cuda()[:,:,opt.flow]).float().item())
    elif (opt.close_size > 0) & (opt.period_size > 0):
        t = 0
        for idx, (c, p, target) in enumerate(data):
            start = time.time()
            pred = torch.relu(best_model(c.float(),opt.mode,opt.c,opt.s,opt.FS,opt.c_t,opt.s_t,opt.flow,p.float()))
            end = time.time()
            t += (end-start)
            predictions.append(pred.float().data.cpu().numpy())
            ground_truth.append(target.float().numpy()[:,:,opt.flow])
            loss.append(criterion(pred.float(), target.cuda()[:,:,opt.flow]).float().item())
            i += 1

    final_predict = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)
    ground_truth_true = mmn.inverse_transform(ground_truth)
    final_predict_true = mmn.inverse_transform(final_predict)
    print("Shape of ground_truth_true[-1, -1, :]:", ground_truth_true[-1, -1, :].shape)
    print("Shape of final_predict_true[-1, -1, :]:", final_predict_true[-1, -1, :].shape)
    filename1 = f'{opt.model_filename}_final_predict_true.csv'
    with open(filename1, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入数据
        writer.writerow(final_predict_true[-1, -1, :])
    filename2 = f'{opt.model_filename}_ground_truth_true.csv'
    with open(filename2, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入数据
        writer.writerow(ground_truth_true[-1, -1, :])
    filename3 = f'{opt.model_filename}_ground_truth_true_all_ponts.csv'
    with open(filename3, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入数据
        writer.writerows(ground_truth_true[-1, :, :])
    filename4 = f'{opt.model_filename}_final_predict_true_all_ponts.csv'
    with open(filename4, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入数据
        writer.writerows(ground_truth_true[-1, :, :])
    font1 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 14
    }
    font2 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 9
    }
    font3 = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 9
    }
    last_date = datetime.strptime('2019/12/13', '%Y/%m/%d')
    date_interval = timedelta(days=11)
    for i in range(final_predict_true.shape[2]):
        plt.figure(figsize=(10, 5))
        line1, = plt.plot(final_predict_true[-1, :, i], label='ST-Trans Prediction', color='red', linewidth=2)
        line2, = plt.plot(ground_truth_true[-1, :, i], label='InSAR', color='black', linewidth=2)

        # 计算日期刻度
        num_dates = final_predict_true.shape[1]
        date_list = [last_date - (num_dates - int(tick) - 1) * date_interval for tick in
                     np.linspace(0, final_predict_true.shape[1] - 1, num=5)]
        date_labels = [date.strftime('%Y/%m/%d') for date in date_list]

        y_min = np.min(ground_truth_true[-1, :, i])
        y_max = np.max(ground_truth_true[-1, :, i])
        y_ticks = np.linspace(y_min, y_max, num=5)  # 生成 5 个均匀分布的 y 轴刻度
        y_tick_labels = ["{:.1f}".format(y) for y in y_ticks]  # 格式化y轴刻度标签，保留一位小数
        plt.yticks(y_ticks, y_tick_labels)
        for label in plt.gca().get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')
            label.set_fontsize(14)

        ax = plt.gca()
        ax.spines['left'].set_position(('outward', 0))  # 将 y 轴设置在最左边
        ax.spines['bottom'].set_position(('outward', 0))

        plt.xticks(np.linspace(0, final_predict_true.shape[1] - 1, num=5), date_labels)
        for label in plt.gca().get_xticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontweight('bold')
            label.set_fontsize(14)

        # 设置 x 轴和 y 轴标签，并同时获取文本对象
        xlabel = plt.xlabel('Time')
        ylabel = plt.ylabel('InSAR time series deformation')

        # 修改文本对象的字体属性
        xlabel.set_fontname('Times New Roman')
        xlabel.set_fontweight('bold')
        xlabel.set_fontsize(14)

        ylabel.set_fontname('Times New Roman')
        ylabel.set_fontweight('bold')
        ylabel.set_fontsize(14)

        # plt.suptitle(f'P{i}')
        # title = plt.gca().get_title()
        # title.set_fontname('Times New Roman')
        # title.set_fontweight('bold')
        # title.set_fontsize(9)
        plt.suptitle(f'P{i}', fontdict={'fontname': 'Times New Roman', 'fontweight': 'bold', 'fontsize': 14})
        plt.legend(handles=[line1, line2], fontsize=14,
                   prop={'family': 'Times New Roman', 'weight': 'bold'})
        plt.savefig(f'{opt.model_filename}_sample_{i}.png')
        plt.close()
    mrt = t/i
    # final_predict = np.concatenate(predictions)
    # ground_truth = np.concatenate(ground_truth)
    plot(final_predict_true[:,:,10],ground_truth[:,:,10],opt.model_filename)

    factor = mmn.max - mmn.min
    sklearn_mae,sklearn_mse,sklearn_rmse,sklearn_nrmse,sklearn_r2 = getmetrics(final_predict.ravel(),ground_truth.ravel())
    a = mmn.inverse_transform(ground_truth)
    b = mmn.inverse_transform(final_predict)
    mae,mse,rmse,nrmse,r2 = getmetrics(b.ravel(),a.ravel())
    sklearn_mse = sklearn_mse.item() if isinstance(sklearn_mse, torch.Tensor) else sklearn_mse
    sklearn_rmse = sklearn_rmse.item() if isinstance(sklearn_rmse, torch.Tensor) else sklearn_rmse
    log_string = ' [MSE]:{:0.5f}, [RMSE]:{:0.5f}, [NRMSE]: {:0.5f}, [MAE]:{:0.5f}, [R2]: {:0.5f},[mrt]:{:0.5f}\n'.format(sklearn_mse,sklearn_rmse,sklearn_nrmse,sklearn_mae,sklearn_r2,mrt)+' [Real MSE]:{:0.5f}, [Real RMSE]:{:0.5f}, [Real NRMSE]: {:0.5f}, [Real MAE]:{:0.5f}, [Real R2]: {:0.5f}'.format(mse,rmse,nrmse,mae,r2)
    plot(b[:,:,224],a[:,:,224],opt.model_filename+'real')
    print(log_string)
    print('mean runtime:',mrt)

    log(opt.model_filename + '.log', log_string)


def train_valid_split(dataloader, test_size=0.2, shuffle=True, random_seed=0):
    length = len(list(dataloader))
    indices = list(range(0, length))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if type(test_size) is float:
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]


if __name__ == '__main__':
    path = '../all_data_sliced2.h5'

    x_train, y_train, x_test, y_test, mmn = load_data(path, opt.traffic, opt.close_size, opt.period_size,
                                                       opt.trend_size,opt.test_size, opt.nb_flow)
    x_train.append(y_train)
    x_test.append(y_test)
    train_data = list(zip(*x_train))
    test_data = list(zip(*x_test))

    # split the training data into train and validation
    train_idx, valid_idx = train_valid_split(train_data,0.1)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
 
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=train_sampler,#num_workers=8,
                               pin_memory=True,drop_last=True)
    valid_loader = DataLoader(train_data, batch_size=opt.batch_size, sampler=valid_sampler,#num_workers=8,
                               pin_memory=True,drop_last=True)

    test_loader = DataLoader(test_data, batch_size=opt.test_batch_size, shuffle=False,drop_last=True)

    external_size = 6

    if opt.g is not None:
        GPU = opt.g
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    print("preparing gpu...")
    if torch.cuda.is_available():
        print('using Cuda devices, num:',torch.cuda.device_count())
        print('using GPU:',torch.cuda.current_device())

    if os.path.isfile(best_model):
        #print(best_model)
        model = torch.load(best_model, weights_only=False)['model'].cuda()
    else:
        model = T_STGCN(opt.close_size, external_size, opt.model_N, opt.k, opt.spatial,
                        opt.c_model_d,opt.s_model_d,opt.p_model_d,opt.t_model_d,
                        temporal_arch=opt.temporal).cuda()
    scheduler = LR_Scheduler(opt.lr_scheduler, lr, total_epochs, len(train_loader),warmup_epochs=opt.warmup)
    optimizer = optim.Adam(model.parameters(),lr,betas=(0.9, 0.98), eps=1e-9)

    if not os.path.isdir(opt.save_dir):
        raise Exception('%s is not a dir' % opt.save_dir)

    if opt.loss == 'l1':
        criterion = nn.L1Loss().cuda()
    elif opt.loss == 'l2':
        criterion = nn.MSELoss().cuda()

    print('Training...')
    log(opt.model_filename + '.log', '[training]')
    if opt.train:
        train()
    #predict('train')
    predict('test')
