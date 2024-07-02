master_addr=localhost
nnodes=1
node_rank=0
nproc_per_node=1
pretrained_path=/hf_cache/modelscope/qwen/Qwen2-7B-Instruct/
data_path=/workspace/Parameter-Efficient-MoE/data/example.json
output_path=/workspace/Parameter-Efficient-MoE/output

# v100 cannot supports
# --bf16 True \
# --tf32 True

torchrun --nproc_per_node=${nproc_per_node} --nnodes=${nnodes} --master_addr ${master_addr} --master_port=4741 --node_rank ${node_rank} pesc_moe.py \
    --model_name_or_path $pretrained_path \
    --data_path $data_path \
    --model_max_length 512 \
    --output_dir $output_path \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --logging_strategy "steps" \
    --eval_steps 1000 \
    --save_steps 1000 \
    --logging_steps 10 \
    --learning_rate 2e-4 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --weight_decay 0.0 \
    --warmup_steps 200 \
    --config_class "qwen2idae.configuration_qwen2idae.Qwen2idaeConfig" \
    --modeling_class "qwen2idae.modeling_qwen2idae.Qwen2ForCausalLM" \
    --merge \
    --test \
    --skip_train \
    --ckpt_path /workspace/Parameter-Efficient-MoE/output/checkpoint-31