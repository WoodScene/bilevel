# 设置起始变量

for data_id in 2 3 4 5 6
do
    CUDA_VISIBLE_DEVICES=5 python generate_avgPerf.py \
        --base_model '/data_8T2/yujie/Backbones/Llama-2-7b-chat' \
        --dataset_id=${data_id} \
        --method_name='vanilla' \
        --model_type='' \

done