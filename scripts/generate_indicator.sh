export CUDA_VISIBLE_DEVICES=0


dataset=ETTh2
seq_len=336
label_len=0
pred_len=720
model_name=PatchTST
batch_size=128
train_epochs=20
patience=5
itr=1
enc_in=7

python -u generate_indicator.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path $dataset.csv \
    --model_id $dataset'_'$seq_len'_'$pred_len \
    --model $model_name \
    --features M \
    --data $dataset \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in $enc_in \
    --dec_in $enc_in \
    --c_out $enc_in \
    --lradj 'type3' \
    --train_epochs $train_epochs \
    --batch_size $batch_size \
    --threshold 0 \
    --des 'Exp' \
    --itr $itr 
