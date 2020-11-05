python masked_run_squad.py --output_dir block_movement_pruning/output --overwrite_output_dir \
                   --data_dir squad_data \
                   --train_file train-v1.1.json --predict_file dev-v1.1.json \
                   --do_train --do_eval \
                   --evaluate_during_training \
                   --do_lower_case \
                   --model_type masked_bert \
                   --model_name_or_path bert-base-uncased \
                   --per_gpu_train_batch_size 1 \
                   --gradient_accumulation_steps 16 \
                   --num_train_epochs 10 \
                   --learning_rate 3e-5 \
                   --initial_threshold 1 \
                   --final_threshold 1.0 \
                   --warmup_steps 5400 \
                   --initial_warmup 1 \
                   --final_warmup 2 \
                   --pruning_method topK \
                   --ampere_pruning_method annealing \
                   --final_ampere_temperature 10 \
                   --initial_ampere_temperature 0.1 \
                   --mask_scores_learning_rate 1e-2 \
                   --mask_init constant \
                   --mask_scale 0. \
                   --mask_block_rows 32 \
                   --mask_block_cols 32 \
                   --threads 8

#		                --truncate_train_examples 100 \
