# 设置起始变量

for data_id in 1
do
    CUDA_VISIBLE_DEVICES=4 python generate_avgPerf_t5lora.py \
        --base_model '/data_8T2/yujie/Backbones/t5large' \
        --dataset_id=${data_id} \
        --method_name='vanilla' \

done