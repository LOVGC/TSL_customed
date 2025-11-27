我现在有一个数据集存储在 dataset\real_doppler_RAD_DAR_database\processed, input_data.npy with shape (N, seq_len, features), where N is the total number of samples; 另外一个 data 是  targets.npy with shape (N, 3), 是 input_data.npy 的 label, 用的是 one-hot encoding for 3 classes。

现在，你需要把这个 input_data.npy 和其 label targets.npy 这两个 numpy array 给 split 成一个 train, val, test dataset： 分别命名为 input_data_train.npy targets_train.npy,  input_data_val.npy targets_val.npy, input_data_test.npy targets_test.npy, 按照比例为 

train: 0.6, val: 0.2, test: 0.2 来做。在 split 的时候要确保是 shuffle 过的，这样就能保证 train, val, test 的数据分布是差不多的。把 split 后的数据存在 dataset\real_doppler_RAD_DAR_database\processed 