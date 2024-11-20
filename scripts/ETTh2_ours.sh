
export CUDA_VISIBLE_DEVICES=0



pred_lens=(720)
datasets=("ETTh2")
seq_len=336
batch_size=128
label_len=0
train_epochs=20
patience=5
itr=1
enc_in=7
score_name="channelscores"
asc_names=("asc")
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.005
d_model=16
d_ff=32


filter_ratios=(0.3)
model_name=DLinear

for dataset in "${datasets[@]}"
do
    for pred_len in "${pred_lens[@]}"
    do
        echo "Running full dataset baseline: $pred_len"
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path $dataset'.csv' \
            --model_id $dataset'_'$seq_len'_'$pred_len \
            --model $model_name \
            --data $dataset \
            --features M \
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
            --patience $patience \
            --batch_size $batch_size \
            --threshold 0 \
            --des 'Exp' \
            --itr $itr 
        for filter_ratio in "${filter_ratios[@]}"
        do
            for asc_name in "${asc_names[@]}"
            do
                score="${score_name}_${asc_name}"
                echo "Running test: $pred_len $filter_ratio asc_name=$asc_name"
                python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path ./dataset/ \
                    --data_path $dataset'.csv' \
                    --model_id $dataset'_'$seq_len'_'$pred_len \
                    --model $model_name \
                    --data $dataset'_ours' \
                    --features M \
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
                    --patience $patience \
                    --batch_size $batch_size \
                    --threshold 0 \
                    --des 'Exp' \
                    --itr $itr \
                    --score $score \
                    --filter_ratio $filter_ratio 
            done
        done 
    done
done



filter_ratios=(0.8)
model_name=TimeMixer

for dataset in "${datasets[@]}"
do
    for pred_len in "${pred_lens[@]}"
    do
        echo "Running full dataset baseline: $pred_len"
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path $dataset'.csv' \
            --model_id $dataset'_'$seq_len'_'$pred_len \
            --model $model_name \
            --data $dataset \
            --features M \
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
            --patience $patience \
            --batch_size $batch_size \
            --threshold 0 \
            --des 'Exp' \
            --down_sampling_layers $down_sampling_layers \
            --down_sampling_method avg \
            --down_sampling_window $down_sampling_window \
            --d_model $d_model \
            --d_ff $d_ff \
            --itr $itr 
        for filter_ratio in "${filter_ratios[@]}"
        do
            for asc_name in "${asc_names[@]}"
            do
                score="${score_name}_${asc_name}"
                echo "Running test: $pred_len $filter_ratio asc_name=$asc_name"
                python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path ./dataset/ \
                    --data_path $dataset'.csv' \
                    --model_id $dataset'_'$seq_len'_'$pred_len \
                    --model $model_name \
                    --data $dataset'_ours' \
                    --features M \
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
                    --patience $patience \
                    --batch_size $batch_size \
                    --threshold 0 \
                    --des 'Exp' \
                    --down_sampling_layers $down_sampling_layers \
                    --down_sampling_method avg \
                    --down_sampling_window $down_sampling_window \
                    --d_model $d_model \
                    --d_ff $d_ff \
                    --itr $itr \
                    --score $score \
                    --filter_ratio $filter_ratio 
            done
        done 
    done
done
