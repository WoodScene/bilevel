#!/bin/bash


begin_id=14

for data_id in 1
do
    # 循环从 begin_id 到 15
    for ((ORDERR=$begin_id; ORDERR<15; ORDERR++))
    do
        # 执行 Python 文件，传递参数 $i
        WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun \
            --nproc_per_node=4 \
            --master_port=1238 \
            finetune_multitask_llama.py \
            --base_model '/data_8T2/yujie/Backbones/Llama-2-7b-chat' \
            --num_epochs=10 \
            --cutoff_len=512 \
            --group_by_length \
            --lora_target_modules='[q_proj,v_proj]' \
            --micro_batch_size=4 \
            --batch_size=32 \
            --dataset_id=${data_id} \
            --task_id=${ORDERR} 


    done
done

wait

begin_id=3

for data_id in 4
do
    # 循环从 begin_id 到 15
    for ((ORDERR=$begin_id; ORDERR<4; ORDERR++))
    do
        # 执行 Python 文件，传递参数 $i
        WORLD_SIZE=4 CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun \
            --nproc_per_node=4 \
            --master_port=1238 \
            finetune_multitask_llama.py \
            --base_model '/data_8T2/yujie/Backbones/Llama-2-7b-chat' \
            --num_epochs=10 \
            --cutoff_len=512 \
            --group_by_length \
            --lora_target_modules='[q_proj,v_proj]' \
            --micro_batch_size=4 \
            --batch_size=32 \
            --dataset_id=${data_id} \
            --task_id=${ORDERR} 


    done
done