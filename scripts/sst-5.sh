export CUDA_VISIBLE_DEVICES=2 &&

python3 cli.py \
--data_dir data/k-shot/sst-5/16-13 \
--model_type roberta \
--model_name_or_path roberta-large \
--cache_dir pretrain/roberta-large \
--task_name sst-5 \
--output_dir output/sst-5 \
--do_eval \
--do_train \
--pet_per_gpu_eval_batch_size 8 \
--pet_per_gpu_train_batch_size 16 \
--pet_gradient_accumulation_steps 1 \
--pet_max_seq_length 128 \
--pet_max_steps 250 \
--learning_rate 1e-4 \
--eval_set "test" \
--prompt_encoder_type "none"


# Albert-xxlarge-v2

# LSTM encoder (2 words)

# None encoder

# Inner encoder (2 words)

# Inner encoder (2 stage)

# RoBERTa-large

