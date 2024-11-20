from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_hour_ours, Dataset_ETT_hour_shuffle_valid, Dataset_ETT_minute, Dataset_ETT_minute_ours, Dataset_Custom, Dataset_Custom_ours
from torch.utils.data import DataLoader
import numpy as np

data_dict = {
    'ETTh1_shuffle_valid': Dataset_ETT_hour_shuffle_valid,
    'ETTh1_ours': Dataset_ETT_hour_ours,
    'ETTh2_ours': Dataset_ETT_hour_ours,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ETTm1_ours': Dataset_ETT_minute_ours,
    'ETTm2_ours': Dataset_ETT_minute_ours,
    'traffic_ours': Dataset_Custom_ours,
    'traffic': Dataset_Custom,
    'weather_ours': Dataset_Custom_ours,
    'weather': Dataset_Custom,
    'weather_ours': Dataset_Custom_ours,
    'weather': Dataset_Custom,
    'electricity_ours': Dataset_Custom_ours,
    'electricity': Dataset_Custom,
    'exchange_rate_ours': Dataset_Custom_ours,
    'exchange_rate': Dataset_Custom,
    'national_illness_ours': Dataset_Custom_ours,
    'national_illness': Dataset_Custom,
    'custom_ours': Dataset_Custom_ours,
    'custom': Dataset_Custom,
}


def data_provider(args, flag, logger):
    Data = data_dict[args.data]
    drop_last = True if flag == 'train' else False
    flag = flag.replace("_no_drop", "")
    filter_ratio = args.filter_ratio if flag == 'train' else 1
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if flag == 'test' else True
    batch_size = args.batch_size
    freq = args.freq

    selected_idxs = None
    if flag == 'train' and "ours" in args.data:
        try:
            settings = '{}_{}_ft{}_sl{}_ll{}_pl{}_eb{}'.format(
                args.task_name,
                args.data.replace("_ours", ""),
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.embed,)
            selected_idxs = np.load(f'./data_curator/results/{settings}/{args.score}_ts{args.threshold}_fr{args.filter_ratio}.npy')
        except:
            logger.info(f'./data_curator/results/{settings}/{args.score}_ts{args.threshold}_fr{args.filter_ratio}.npy  not found!')
    if "ours" in args.data:
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            selected_idxs = selected_idxs,
            filter_ratio = filter_ratio
        )
    else:
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
        )
    logger.info(f"{flag}: {len(data_set)}")
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
