import os
import argparse
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from models import DLinear, PatchTST
from numpy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import pandas as pd
import logging
from torch import optim
import torch
import torch.nn as nn
import gc  # 引入垃圾回收模块
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger('loss')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 将handler添加到logger
logger.addHandler(console_handler)


def vali(args, vali_data, vali_loader, criterion):
    total_loss = []
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for i, batch_data in enumerate(vali_loader):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            preds.append(pred)
            trues.append(true)

            loss = criterion(pred, true)

            total_loss.append(loss)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    total_loss = np.mean((preds - trues) ** 2)
    model.train()
    return total_loss


def save_results(channelscores, scores, data_idxs, output_dir):
    # Convert lists to numpy arrays if they aren't already
    channelscores_array = np.array(channelscores)
    scores_array = np.array(scores)
    data_idxs_array = np.array(data_idxs)
    
    # Save arrays to .npy files
    np.save(f'{output_dir}/channelscores.npy', channelscores_array)
    np.save(f'{output_dir}/scores.npy', scores_array)  # Save average scores
    np.save(f'{output_dir}/data_idxs.npy', data_idxs_array)
    print("Files saved successfully.")


parser = argparse.ArgumentParser(description='TSCurator')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='Autoformer',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')
# our method
parser.add_argument('--threshold', type=float, default=0.2, help='status')
parser.add_argument('--filter_ratio', type=float, default=1.0, help='status')
parser.add_argument('--score', type=str, default='random', help='random, seasonal')
parser.add_argument('--individual', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False, help='channel independence')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--revin', action='store_true', help='inverse output data', default=False)


# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--patch_len', type=int, default=16, help='patch len for patch_embedding')
parser.add_argument('--stride', type=int, default=8, help='stride for patch_embedding')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=48,
                    help='the length of segmen-wise iteration of SegRNN')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# metrics (dtw)
parser.add_argument('--use_dtw', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=False, 
                    help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

# Augmentation
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")


arg_mapping = {
    "ETTh1": {
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        'individual': False
    },
    "ETTh2": {
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        'individual': False
    },
    "ETTm1": {
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        'individual': False
    },
    "ETTm2": {
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        'individual': False
    },
    "weather": {
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
        'individual': True
    },
    "traffic": {
        "enc_in": 862,
        "dec_in": 862,
        "c_out": 862,
        'individual': False
    },
    "electricity": {
        "enc_in": 321,
        "dec_in": 321,
        "c_out": 321,
        'individual': False
    },
    "exchange_rate": {
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        'individual': False
    },
    "national_illness": {
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        'individual': False
    },
}
args = parser.parse_args()
dataset = args.data
        
if arg_mapping[dataset]['individual'] == True:
    args.individual = True
# if dataset == 'national_illness':
#     args.seq_len = 104
args.enc_in = arg_mapping[dataset]['enc_in']
args.dec_in = arg_mapping[dataset]['dec_in']
args.c_out = arg_mapping[dataset]['c_out']
settings = '{}_{}_ft{}_sl{}_ll{}_pl{}_eb{}'.format(
    args.task_name,
    args.data,
    args.features,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.embed,)

path = os.path.join(args.checkpoints, settings)
if not os.path.exists(path):
    os.makedirs(path)
output_directory = f'./data_curator/results/{settings}_loss_based'
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data, train_loader = data_provider(args, 'train', logger)
train_data, train_no_drop_loader = data_provider(args, 'train_no_drop', logger)
vali_data, vali_loader = data_provider(args, 'val', logger)
test_data, test_loader = data_provider(args, 'test', logger)

model = PatchTST.Model(args).float().to(device)
model_optim = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.MSELoss()
early_stopping = EarlyStopping(patience=args.patience, verbose=True)

iter_count = 0
time_now = time.time()
train_steps = len(train_loader)
if args.is_training == 1:
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
    
        model.train()
        epoch_time = time.time()
        for i, batch_data in enumerate(train_loader):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
    
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
    
            # encoder - decoder
            if args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
    
            loss.backward()
            model_optim.step()
        logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss = vali(args, vali_data, vali_loader, criterion)
        test_loss = vali(args, test_data, test_loader, criterion)
    
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            break
    
        adjust_learning_rate(model_optim, epoch + 1, args)
logger.info('processing results...')

# Load the best model
best_model_path = os.path.join(path, 'checkpoint.pth')
model.load_state_dict(torch.load(best_model_path))

data_idxs = []
all_seq_data = []
channelscores = []
scores = []  # To store average scores
with torch.no_grad():
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, data_idx) in enumerate(train_no_drop_loader):
        # B, L, C = batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
    
        # Get model outputs
        outputs = model(batch_x, batch_x_mark, None, batch_y_mark)
        
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
        
        # Compute per-channel MSE loss
        loss = torch.nn.MSELoss(reduction='none')(outputs, batch_y).mean(dim=(1))  # Shape: [batch_size, num_channels]
        
        # Store per-channel scores
        channelscores.extend(loss.detach().cpu().numpy().tolist())
        all_seq_data.append(batch_x.detach().cpu().numpy())
        
        # Compute average score across channels for each data point
        avg_loss = loss.mean(dim=1).cpu().numpy()  # Shape: [batch_size]
        scores.extend(avg_loss.tolist())
        
        for idx in data_idx:
            data_idxs.extend([int(idx)])  # Repeat each index C times for channelscores

# Save both channelscores and average scores
save_results(channelscores, scores, data_idxs, output_directory)

# Now, handle filtering based on both channelscores and average scores
result_dir = output_directory.replace("_loss_based", "")
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
asc_names = ['asc', 'desc', 'median']
channel_score_name = "channelscores"
average_score_name = "scores"
threshold = 0.0

# Filtering based on per-channel scores
for asc_name in asc_names:
    channel_score_file = f"{channel_score_name}.npy"
    channelscores = np.load(f'{output_directory}/{channel_score_file}')
    data_idxs_array = np.load(f'{output_directory}/data_idxs.npy')
    num_channels  = channelscores.shape[-1]
    
    # 确定每个通道的筛选结果后，合并到一个大数组中
    for filter_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        all_selected_indices = np.array([], dtype=int)
        for channel in range(num_channels):
            channel_scores = channelscores[:, channel]
            if asc_name == 'median':
                median_value = np.median(channel_scores)
                sorted_indices = np.argsort(np.abs(channel_scores - median_value))
            elif asc_name == "desc":
                sorted_indices = np.argsort(channel_scores)[::-1]
            else:
                sorted_indices = np.argsort(channel_scores)

            selected_indices = sorted_indices[:int(filter_ratio * len(channel_scores))]
            all_selected_indices = np.concatenate((all_selected_indices, selected_indices))
        all_selected_indices = all_selected_indices.reshape(num_channels, -1).T
        # 保存所有通道的选定索引到一个文件中
        print(f'{channel_score_name}_{asc_name}_ts{threshold}_fr{filter_ratio}', len(all_selected_indices))
        np.save(f'{result_dir}/{channel_score_name}_{asc_name}_ts{threshold}_fr{filter_ratio}.npy', data_idxs_array[all_selected_indices])
        gc.collect()  # 在每次大循环后调用垃圾回收

# Filtering based on average scores
for asc_name in asc_names:
    average_score_file = f"{average_score_name}.npy"
    average_scores = np.load(f'{output_directory}/{average_score_file}')
    data_idxs_array = np.load(f'{output_directory}/data_idxs.npy')
    
    num_samples = average_scores.shape[0]
    
    # Define filter ratios for average scores
    for filter_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if asc_name == 'median':
            median_value = np.median(average_scores)
            sorted_indices = np.argsort(np.abs(average_scores - median_value))
        elif asc_name == "desc":
            sorted_indices = np.argsort(average_scores)[::-1]
        else:
            sorted_indices = np.argsort(average_scores)

        selected_indices = sorted_indices[:int(filter_ratio * num_samples)]
        
        print(f'{average_score_name}_{asc_name}_ts{threshold}_fr{filter_ratio}', len(selected_indices))
        np.save(f'{result_dir}/{average_score_name}_{asc_name}_ts{threshold}_fr{filter_ratio}.npy', data_idxs_array[selected_indices])
        gc.collect()  # 在每次大循环后调用垃圾回收


# Now proceed to compute PCA
all_seq_data_np = np.concatenate(all_seq_data)  # Shape: (num_samples, seq_len, num_channels)
num_samples, seq_len, num_channels = all_seq_data_np.shape
all_seq_data_reshaped = all_seq_data_np.reshape(num_samples, -1)  # Shape: (num_samples, seq_len * num_channels)
# Perform PCA and compute reconstruction loss
n_components = 10  # Choose the number of principal components
pca = PCA(n_components=n_components)
pca.fit(all_seq_data_reshaped)

# Transform the data to lower-dimensional space and reconstruct
X_pca = pca.transform(all_seq_data_reshaped)
X_reconstructed = pca.inverse_transform(X_pca)

# Compute per-sample reconstruction error (mean squared error)
reconstruction_errors = np.mean((all_seq_data_reshaped - X_reconstructed) ** 2, axis=1)  # Shape: (num_samples,)

# Save pcascores
np.save(f'{output_directory}/pcascores.npy', reconstruction_errors)

# Reshape reconstructed data to original shape
X_reconstructed_reshaped = X_reconstructed.reshape(num_samples, seq_len, num_channels)

# Compute per-sample per-channel reconstruction error
channel_reconstruction_errors = np.mean((all_seq_data_np - X_reconstructed_reshaped) ** 2, axis=1)  # Shape: (num_samples, num_channels)

# Save channelpcascores
np.save(f'{output_directory}/channelpcascores.npy', channel_reconstruction_errors)

# Now, you can proceed to filter based on the new 'pcascores' and 'channelpcascores' similar to how you did for 'scores' and 'channelscores'

asc_names = ['asc', 'desc', 'median']
threshold = 0.0
result_dir = output_directory.replace("_loss_based", "")
# Filtering based on per-channel PCA scores
channel_score_name = "channelpcascores"
for asc_name in asc_names:
    channel_score_file = f"{channel_score_name}.npy"
    channelscores = np.load(f'{output_directory}/{channel_score_file}')
    data_idxs_array = np.load(f'{output_directory}/data_idxs.npy')
    num_channels = channelscores.shape[-1]

    for filter_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        all_selected_indices = np.array([], dtype=int)
        for channel in range(num_channels):
            channel_scores = channelscores[:, channel]
            if asc_name == 'median':
                median_value = np.median(channel_scores)
                sorted_indices = np.argsort(np.abs(channel_scores - median_value))
            elif asc_name == "desc":
                sorted_indices = np.argsort(channel_scores)[::-1]
            else:
                sorted_indices = np.argsort(channel_scores)

            selected_indices = sorted_indices[:int(filter_ratio * len(channel_scores))]
            all_selected_indices = np.concatenate((all_selected_indices, selected_indices))
        all_selected_indices = all_selected_indices.reshape(num_channels, -1).T

        print(f'{channel_score_name}_{asc_name}_ts{threshold}_fr{filter_ratio}', len(all_selected_indices))
        np.save(f'{result_dir}/{channel_score_name}_{asc_name}_ts{threshold}_fr{filter_ratio}.npy', data_idxs_array[all_selected_indices])
        gc.collect()

# Filtering based on PCA average scores
average_score_name = "pcascores"
for asc_name in asc_names:
    average_score_file = f"{average_score_name}.npy"
    average_scores = np.load(f'{output_directory}/{average_score_file}')
    data_idxs_array = np.load(f'{output_directory}/data_idxs.npy')

    num_samples = average_scores.shape[0]

    for filter_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if asc_name == 'median':
            median_value = np.median(average_scores)
            sorted_indices = np.argsort(np.abs(average_scores - median_value))
        elif asc_name == "desc":
            sorted_indices = np.argsort(average_scores)[::-1]
        else:
            sorted_indices = np.argsort(average_scores)

        selected_indices = sorted_indices[:int(filter_ratio * num_samples)]

        print(f'{average_score_name}_{asc_name}_ts{threshold}_fr{filter_ratio}', len(selected_indices))
        np.save(f'{result_dir}/{average_score_name}_{asc_name}_ts{threshold}_fr{filter_ratio}.npy', data_idxs_array[selected_indices])
        gc.collect()
