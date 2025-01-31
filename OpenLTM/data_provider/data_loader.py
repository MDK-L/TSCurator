import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class UnivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1

    def __getitem__(self, index):
        feat_id = index // self.n_timepoint
        s_begin = index % self.n_timepoint
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_var * self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return int(self.n_var * self.n_timepoint)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class MultivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1
        print(self.n_var, self.n_timepoint)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return self.n_timepoint

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class PretrainMultivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0, fixed_length=4320, base_length=96):
        """
        初始化数据集。

        参数:
        - root_path: 数据根路径
        - flag: 数据集类型 ['train', 'test', 'val']
        - size: 序列长度 [seq_len, input_token_len, output_token_len]
        - data_path: 数据文件路径
        - scale: 是否进行标准化
        - nonautoregressive: 是否为非自回归模式
        - subset_rand_ratio: 子集随机比例
        - fixed_length: 拼接后的固定长度
        - base_length: 基础片段长度（倍数）
        """
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        self.fixed_length = fixed_length
        self.base_length = base_length  # 基础长度，如96
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        # 根据数据类型划分训练、验证、测试集
        if self.data_type in ['ETTh', 'ETTh1', 'ETTh2']:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type in ['ETTm', 'ETTm1', 'ETTm2']:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 处理数据格式
        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        # 标准化
        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint = len(self.data_x) - self.seq_len - self.output_token_len + 1
        print(f"Number of variables: {self.n_var}, Number of timepoints: {self.n_timepoint}")

    def __getitem__(self, index):
        """
        获取一个样本。

        每个样本由若干个96的倍数长度的片段随机拼接而成，拼接后的长度为fixed_length。
        """
        # 随机选择多个base_length的倍数
        # 确保总长度不超过fixed_length
        remaining_length = self.fixed_length
        x_segments = []
        y_segments = []
        current_index = index

        while remaining_length >= self.base_length:
            multiplier = np.random.randint(1, len(self.data_x) // self.base_length)
            segment_length = self.base_length * multiplier

            # 随机选择一个起始索引，确保不会越界
            current_index = np.random.randint(0, self.n_timepoint)
            
            # 确保不会越界
            if current_index + segment_length > len(self.data_x):
                segment_length = len(self.data_x) - current_index
                multiplier = segment_length // self.base_length
                segment_length = self.base_length * multiplier

            x_segment = self.data_x[current_index:current_index + segment_length]
            x_segments.append(x_segment)
            if not self.nonautoregressive:
                y_segment = self.data_y[current_index + self.input_token_len:current_index + segment_length + self.output_token_len]
                y_segment = torch.tensor(y_segment)
                y_segment = y_segment.unfold(dimension=0, size=self.output_token_len,
                                     step=self.input_token_len).permute(0, 2, 1)
                y_segment = y_segment.reshape(y_segment.shape[0] * y_segment.shape[1], -1)
            else:
                y_segment = self.data_y[current_index + segment_length:current_index + segment_length + self.output_token_len]
            y_segments.append(y_segment)
            current_index += segment_length
            remaining_length -= segment_length

        # 拼接所有片段
        x_concatenated = np.concatenate(x_segments, axis=0)
        y_concatenated = np.concatenate(y_segments, axis=0)
        # 如果超过fixed_length，截断
        x_concatenated = x_concatenated[:self.fixed_length]
        y_concatenated = y_concatenated[:self.fixed_length]
        seq_x = torch.tensor(x_concatenated, dtype=torch.float)
        seq_y = torch.tensor(y_concatenated, dtype=torch.float)


        # 标记（可以根据需要调整）
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        返回数据集的长度。

        对于训练集，使用subset_rand_ratio控制随机子集的大小。
        """
        if self.set_type == 0:
            return max(int(self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return self.n_timepoint

    def inverse_transform(self, data):
        """
        反标准化数据。
        """
        return self.scaler.inverse_transform(data)


class SelectedMultivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0, selected_idxs=None):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.selected_idxs = selected_idxs
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1
        print(self.n_var, self.n_timepoint)

    def __getitem__(self, index):

        if self.set_type == 0:
            s_begin = self.selected_idxs[index, :]
            s_end = s_begin + self.seq_len
            if not self.nonautoregressive:
                r_begin = s_begin + self.input_token_len
                r_end = s_end + self.output_token_len
                seq_x, seq_y = [], []
                for i in range(len(s_begin)):
                    seq_x.append(self.data_x[s_begin[i]:s_end[i], i])
                    seq_y.append(self.data_y[r_begin[i]:r_end[i], i])
                seq_x, seq_y = np.array(seq_x).T, np.array(seq_y).T
                seq_y = torch.tensor(seq_y)
                seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                     step=self.input_token_len).permute(0, 2, 1)
                seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
            else:
                r_begin = s_end
                r_end = r_begin + self.output_token_len
                seq_x, seq_y = [], []
                for i in range(len(s_begin)):
                    seq_x.append(self.data_x[s_begin[i]:s_end[i], i])
                    seq_y.append(self.data_y[r_begin[i]:r_end[i], i])
                seq_x, seq_y = np.array(seq_x).T, np.array(seq_y).T
        else:
            s_begin = index
            s_end = s_begin + self.seq_len
            if not self.nonautoregressive:
                r_begin = s_begin + self.input_token_len
                r_end = s_end + self.output_token_len
                seq_x = self.data_x[s_begin:s_end]
                seq_y = self.data_y[r_begin:r_end]
                seq_y = torch.tensor(seq_y)
                seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                     step=self.input_token_len).permute(0, 2, 1)
                seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
            else:
                r_begin = s_end
                r_end = r_begin + self.output_token_len
                seq_x = self.data_x[s_begin:s_end]
                seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return len(self.selected_idxs)
        else:
            return self.n_timepoint

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class BalancedMultivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [4380 - self.seq_len - self.output_token_len, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [4380 - self.seq_len - self.output_token_len, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [4380 - self.seq_len - self.output_token_len, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1
        print(self.n_var, self.n_timepoint)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return self.n_timepoint

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class NearselectedMultivariateDatasetBenchmark(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.data_type = data_path.split('.')[0]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.subset_rand_ratio = subset_rand_ratio
        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset_file_path = os.path.join(self.root_path, self.data_path)
        if dataset_file_path.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file_path)
        elif dataset_file_path.endswith('.txt'):
            df_raw = []
            with open(dataset_file_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif dataset_file_path.endswith('.npz'):
            data = np.load(dataset_file_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif dataset_file_path.endswith('.npy'):
            data = np.load(dataset_file_path)
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(dataset_file_path))

        if self.data_type == 'ETTh' or self.data_type == 'ETTh1' or self.data_type == 'ETTh2':
            border1s = [12 * 30 * 24 - 1000 - self.seq_len, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_type == 'ETTm' or self.data_type == 'ETTm1' or self.data_type == 'ETTm2':
            border1s = [12 * 30 * 24 - 1000 - self.seq_len, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [num_train - 1000 - self.seq_len, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        self.n_var = self.data_x.shape[-1]
        self.n_timepoint =  len(self.data_x) - self.seq_len - self.output_token_len + 1
        print(self.n_var, self.n_timepoint)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.set_type == 0:
            return max(int(self.n_timepoint * self.subset_rand_ratio), 1)
        else:
            return self.n_timepoint

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Global_Temp(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = np.load(os.path.join(self.root_path,
                                             "temp_global_hourly_" + self.flag + ".npy"),
                                allow_pickle=True)
        raw_data = self.raw_data
        data_len, station, feat = raw_data.shape
        raw_data = raw_data.reshape(data_len, station * feat)
        data = raw_data.astype(np.float)

        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.output_token_len + 1


class Global_Wind(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = np.load(os.path.join(self.root_path,
                                             "wind_global_hourly_" + self.flag + ".npy"),
                                allow_pickle=True)
        raw_data = self.raw_data
        data_len, station, feat = raw_data.shape
        raw_data = raw_data.reshape(data_len, station * feat)
        data = raw_data.astype(np.float)

        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.output_token_len + 1


class Dataset_ERA5_Pretrain(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - \
            self.output_token_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path))
        # split only the train set
        L, S = df_raw.shape
        Train_S = int(S * 0.8)
        df_raw = df_raw[:, :Train_S]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len,
                    len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.output_token_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ERA5_Pretrain_Test(Dataset):
    def __init__(self, root_path, flag='test', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.test_flag = test_flag
        assert test_flag in ['T', 'V', 'TandV']
        type_map = {'T': 0, 'V': 1, 'TandV': 2}
        self.test_type = type_map[flag]
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - \
            self.output_token_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = np.load(os.path.join(self.root_path, self.data_path))
        # split only the train set
        L, S = df_raw.shape
        if self.test_type == 0:
            Train_S = int(S * 0.8)
            df_raw = df_raw[:, :Train_S]
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len,
                        len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            data = df_raw
            border1 = border1s[-1]
            border2 = border2s[-1]

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
        else:
            Train_S = int(S * 0.8)
            df_raw = df_raw[:, Train_S:]
            num_train = int(len(df_raw) * 0.8)
            num_test = len(df_raw) - num_train
            border1s = [0, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, len(df_raw)]
            data = df_raw
            if self.test_type == 1:
                border1 = border1s[0]
                border2 = border2s[0]
            else:
                border1 = border1s[1]
                border2 = border2s[1]

            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        s_end = s_begin + self.seq_len

        if not self.nonautoregressive:
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len

            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_y = torch.tensor(seq_y)
            seq_y = seq_y.unfold(dimension=0, size=self.output_token_len,
                                 step=self.input_token_len).permute(0, 2, 1)
            seq_y = seq_y.reshape(seq_y.shape[0] * seq_y.shape[1], -1)
        else:
            r_begin = s_end
            r_end = r_begin + self.output_token_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.output_token_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# Download link: https://huggingface.co/datasets/thuml/UTSD
class UTSD(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, stride=1, split=0.9, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.context_len = self.seq_len + self.output_token_len
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.nonautoregressive = nonautoregressive
        self.split = split
        self.stride = stride
        self.data_list = []
        self.n_window_list = []
        self.root_path = root_path
        self.__confirm_data__()

    def __confirm_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.csv'):
                    dataset_path = os.path.join(root, file)

                    self.scaler = StandardScaler()
                    df_raw = pd.read_csv(dataset_path)

                    if isinstance(df_raw[df_raw.columns[0]][0], str):
                        data = df_raw[df_raw.columns[1:]].values
                    else:
                        data = df_raw.values

                    num_train = int(len(data) * self.split)
                    num_test = int(len(data) * (1 - self.split) / 2)
                    num_vali = len(data) - num_train - num_test
                    if num_train < self.context_len:
                        continue
                    border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(data)]

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    if self.scale:
                        train_data = data[border1s[0]:border2s[0]]
                        self.scaler.fit(train_data)
                        data = self.scaler.transform(data)
                    else:
                        data = data

                    data = data[border1:border2]
                    n_timepoint = (
                        len(data) - self.context_len) // self.stride + 1
                    n_var = data.shape[1]
                    self.data_list.append(data)
                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
        print("Total number of windows in merged dataset: ",
              self.n_window_list[-1])

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - \
            self.n_window_list[dataset_index -
                               1] if dataset_index > 0 else index
        n_timepoint = (
            len(self.data_list[dataset_index]) - self.context_len) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = self.data_list[dataset_index][s_begin:s_end,
                                              c_begin:c_begin + 1]
        seq_y = self.data_list[dataset_index][r_begin:r_end,
                                              c_begin:c_begin + 1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]


# Download link: https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/
class UTSD_Npy(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv', scale=True, nonautoregressive=False, stride=1, split=0.9, test_flag='T', subset_rand_ratio=1.0):
        self.seq_len = size[0]
        self.input_token_len = size[1]
        self.output_token_len = size[2]
        self.context_len = self.seq_len + self.output_token_len
        self.flag = flag
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.scale = scale
        self.root_path = root_path
        self.nonautoregressive = nonautoregressive
        self.split = split
        self.stride = stride
        self.data_list = []
        self.n_window_list = []
        self.__confirm_data__()

    def __confirm_data__(self):
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith('.npy'):
                    dataset_path = os.path.join(root, file)

                    self.scaler = StandardScaler()
                    data = np.load(dataset_path)

                    num_train = int(len(data) * self.split)
                    num_test = int(len(data) * (1 - self.split) / 2)
                    num_vali = len(data) - num_train - num_test
                    if num_train < self.context_len:
                        continue
                    border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
                    border2s = [num_train, num_train + num_vali, len(data)]

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    if self.scale:
                        train_data = data[border1s[0]:border2s[0]]
                        self.scaler.fit(train_data)
                        data = self.scaler.transform(data)
                    else:
                        data = data

                    data = data[border1:border2]
                    n_timepoint = (
                        len(data) - self.context_len) // self.stride + 1
                    n_var = data.shape[1]
                    self.data_list.append(data)

                    n_window = n_timepoint * n_var
                    self.n_window_list.append(n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + n_window)
        print("Total number of windows in merged dataset: ",
              self.n_window_list[-1])

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - \
            self.n_window_list[dataset_index -
                               1] if dataset_index > 0 else index
        n_timepoint = (
            len(self.data_list[dataset_index]) - self.context_len) // self.stride + 1

        c_begin = index // n_timepoint  # select variable
        s_begin = index % n_timepoint  # select start timestamp
        s_begin = self.stride * s_begin
        s_end = s_begin + self.seq_len
        r_begin = s_begin + self.input_token_len
        r_end = s_end + self.output_token_len

        seq_x = self.data_list[dataset_index][s_begin:s_end,
                                              c_begin:c_begin + 1]
        seq_y = self.data_list[dataset_index][r_begin:r_end,
                                              c_begin:c_begin + 1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.n_window_list[-1]
