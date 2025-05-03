#!/bin/bash

# 设置起始变量
begin_id=3


# 循环从 begin_id 到 15
for data_id in 4
do
    # 执行 Python 文件，传递参数 $i

    CUDA_VISIBLE_DEVICES=4 python finetune_multitask_t5lora.py \
        --base_model '/data_8T2/yujie/Backbones/t5large' \
        --num_epochs=10 \
        --dataset_id=${data_id} \
        --task_id=${begin_id} \

done
