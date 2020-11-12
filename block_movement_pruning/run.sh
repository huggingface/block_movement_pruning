

python masked_run_squad.py --output_dir block_movement_pruning/output --overwrite_output_dir \
                   --data_dir squad_data \
                   --train_file train-v1.1.json --predict_file dev-v1.1.json \
                   --do_train --do_eval \
		               --save_steps 5000 \
		               --logging_steps 200 \
                   --eval_all_checkpoints \
                   --do_lower_case \
                   --model_type masked_bert \
                   --model_name_or_path bert-base-uncased \
                   --per_gpu_train_batch_size 16 --gradient_accumulation_steps 1 \
		               --per_gpu_eval_batch_size 16 \
                   --num_train_epochs 10 \
                   --learning_rate 3e-5 \
                   --initial_threshold 1 \
                   --final_threshold 0.1 \
                   --warmup_steps 5400 \
                   --initial_warmup 1 \
                   --final_warmup 2 \
                   --pruning_method topK \
                   --ampere_pruning_method disabled \
                   --mask_scores_learning_rate 1e-2 \
                   --mask_init constant \
                   --mask_scale 0. \
                   --mask_block_rows 1 \
                   --mask_block_cols 1 \
                    --threads 8 \
                   --shuffling_method mask_annealing \
                   --initial_shuffling_temperature 0.1 \
                   --final_shuffling_temperature 20 \
                   --shuffling_learning_rate 1e-2 \
                   --in_shuffling_group 2 \
                   --out_shuffling_group 2
                   #--truncate_train_examples 100
#                   --initial_ampere_temperature 0.1 \
#                   --final_ampere_temperature 10 \
