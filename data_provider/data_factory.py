from data_provider.data_loader import Dataset_ETT_hour, Dataset_Berkley_sensor, Dataset_Simulate_ar, UEAloader, \
    Dataset_Real_Doppler_Kaggle
from data_provider.uea import UEA_Collate_Fn
from data_provider.berkely_sensor_data import collate_fn_for_None_type

from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'moiteid_3_temp_volt': Dataset_Berkley_sensor,
    'simulate_ar': Dataset_Simulate_ar,
    'real_doppler_kaggle': Dataset_Real_Doppler_Kaggle,
    'UEA': UEAloader  # classification task 的数据集全是用这个 UEAloader
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq  # data point 的 sample freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        if args.data == 'real_doppler_kaggle':
            data_set = Data(flag=flag)
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=False,
                collate_fn=UEA_Collate_Fn(max_len=None))
            return data_set, data_loader
        else:
            data_set = Data(
                args=args,
                root_path=args.root_path,
                flag=flag,
            )
            print(flag, len(data_set))
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=UEA_Collate_Fn(max_len=args.seq_len)
            )
            return data_set, data_loader
    else:  # Berkely Sensor Data, for single variable predicts single variable, 这里的 collate_fn 是自己写的，会降低训练速度。没有 default 的快。
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)  # 使用自定义的 collate_fn 来处理包含 None 的情况
        return data_set, data_loader