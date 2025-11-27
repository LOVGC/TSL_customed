在 data_provider\data_loader.py 中有一个 class Dataset_Real_Doppler_Kaggle(Dataset)。这个 Dataset class 是用来对 dataset\real_doppler_RAD_DAR_database\processed 这里面的数据做抽象的。

你的任务是 implement 这个 Dataset class, 要求是
- user 可以根据 flag = "train" or "val" or "test" 来 load 相应的 data. e.g. flag == "train", 然后就 load dataset\real_doppler_RAD_DAR_database\processed\input_data_train.npy 以及其对应的 label dataset\real_doppler_RAD_DAR_database\processed\targets_train.npy 

以下以 flag == "train" 来举例，对于 flag == "test" or "val", 是一样的：
- 对于 __getitem__() 这个 method, 它的输入是一个 index, 然后输出就是 input_data_train 的对应 index 的那个  sample, 以及这个 sample 对应的 target. i.e. return input_data_sample, target_sample

这里 input_data shape is (N, seq_len, features), target shape is (N, 3) which uses one-hot encoding for a 3-class problem.

要求：不要改变 data_provider\data_loader.py 中其他 class 的 implementation。如果需要 import 其他额外 packages，可以 import。也可以 implement helper functions, 使得代码更加 readable and elegant。

