#!/bin/bash

# 设置起始变量
begin_id=0

for data_id in 1 2 3
do
    # 循环从 begin_id 到 15
    for ((ORDER=$begin_id; ORDER<15; ORDER++))
    do
        # 执行 Python 文件，传递参数 $i

        CUDA_VISIBLE_DEVICES=4 python finetune_vanilla_t5lora.py \
            --base_model '/data_8T2/yujie/Backbones/t5large' \
            --num_epochs=10 \
            --dataset_id=${data_id} \
            --task_id=${ORDER} \

    done
done