export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet 

# to use your customed data, need to change 
# 1. data_loader.py (define your dataset)
# 2. data_factory.py (import your dataset and add to data_dict)
# 3. add your data to the folder ./dataset/YOUR_DATASET/YOUR_DATA.csv
# 4. change the parameters below:
#    --root_path (the path of your dataset)
#    --data_path (the name of your data file)
#    --data (the name of your dataset defined in data_factory.py)   
#    --model_id
#    --enc_in, --dec_in, --c_out 
#    --target (the target feature you want to predict)
#    --features (S for single feature, M for multi-features)
#    --seq_len, --label_len, --pred_len (set according to your needs)
#    ....
#    the key is that 基本上每一个参数你都要看一下。

python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/simulate_data/ \
  --data_path ar_data.csv \
  --model_id ar_data_96_96 \
  --model $model_name \
  --data simulate_ar \
  --features S \
  --target AR_Value \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 