#!/bin/bash

# 设置起始变量
begin_id=14


# 循环从 begin_id 到 15
for data_id in 7
do
    # 执行 Python 文件，传递参数 $i

    CUDA_VISIBLE_DEVICES=4 python finetune_multitask_t5lora.py \
        --base_model '/data_8T2/yujie/Backbones/t5large' \
        --data_dir './data_superni' \
        --num_epochs=10 \
        --dataset_id=${data_id} \
        --task_id=${begin_id} \

done

wait

for data_id in 7
do
    CUDA_VISIBLE_DEVICES=5 python generate_avgPerf_t5lora.py \
        --base_model '/data_8T2/yujie/Backbones/t5large' \
        --dataset_id=${data_id} \
        --method_name='multitask' \
        --model_type='' \

done