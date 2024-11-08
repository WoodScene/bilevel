# 生成计算fwt指标所需要的预测结果
# 用i-1的权重去测试i的结果，共T-1次
# 代码里就不循环了，在外面的sh中做循环。

import os
import sys
import json
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


from utils.dataset_order import get_dataset_order


def main(
    load_8bit: bool = True,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    testfile_name: str = "",
    testfile_idx: str = "",
    output_file: str = "",
    dataset_id: int = 1, # 1 - 5  5次实验
    max_new_tokens: int = 128, # 1 - 5  5次实验
    service_begin_id: int = 1 ,
    with_replay: bool = False, # 这个表示是否用有memory的数据集进行训练
    lora_type: str = "TaSL",
    hyper_para: int = 1, # 超参数的组合
    hyper_para2: str = "", # 超参数的组合2
    hyper_para3: str = "", # 超参数的组合3,二阶梯度用的
):
    print(f"base_model: {base_model}")
    print(f"max_new_tokens: {max_new_tokens}")
    print(f"dataset_id: {dataset_id}")
    print(f"with_replay: {with_replay}")
    print(f"lora_type: {lora_type}")
    print(f"hyper_para: {hyper_para}")
    print(f"hyper_para2: {hyper_para2}")
    print(f"hyper_para3: {hyper_para3}")
    dataset_order = get_dataset_order(dataset_id)
    last_service_name = dataset_order[service_begin_id-1]
    if hyper_para3 != "":
        model_name = str(hyper_para3) 
    elif hyper_para2 != "":
        model_name = base_model.split("/")[-1] + "lora" + str(hyper_para2)  
    else:
        model_name = base_model.split("/")[-1] + "lora" + str(hyper_para)
    print(f"model_name: {model_name}")
    
    if with_replay:
        if lora_type == "vanilla":
            lora_weights = os.path.join("./checkpoint_files", model_name + "_dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_begin_id-1)+"-"+last_service_name)
        else:
            lora_weights = os.path.join("./checkpoint_files", model_name + "_TaSL_dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_begin_id-1)+"-"+last_service_name+ "-averaging")
    else:
        if lora_type == "vanilla":
            lora_weights = os.path.join("./checkpoint_files", model_name + "_dataset_id_"+str(dataset_id), str(service_begin_id-1)+"-"+last_service_name)
        else:
            lora_weights = os.path.join("./checkpoint_files", model_name + "_TaSL_dataset_id_"+str(dataset_id), str(service_begin_id-1)+"-"+last_service_name+ "-averaging")

    if not os.path.exists(lora_weights):
        print(f"lora dir {lora_weights} not find!")
        sys.exit(1)   
    assert (
        lora_weights
    ), "Please specify a --lora_weights, e.g. --lora_weights='xxx'"


    if with_replay:
        if lora_type == "vanilla":
            output_dir = os.path.join("./output", model_name + "_dataset_id_"+str(dataset_id)+"_fwt_with_memoryreplay")
        else:
            output_dir = os.path.join("./output", model_name + "_TaSL_dataset_id_"+str(dataset_id)+"_fwt_with_memoryreplay")
    else:
        if lora_type == "vanilla":
            output_dir = os.path.join("./output", model_name + "_dataset_id_"+str(dataset_id)+"_fwt")
        else:
            output_dir = os.path.join("./output", model_name + "_TaSL_dataset_id_"+str(dataset_id)+"_fwt")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"lora_weights: {lora_weights}")
    print(f"output_dir: {output_dir}")
    
    
    
    prompter = Prompter(prompt_template)
    #tokenizer = LlamaTokenizer.from_pretrained(base_model)
    #tokenizer = LlamaTokenizer.from_pretrained('/data_8T1/liubo/llama_weights_hf')
    tokenizer = AutoTokenizer.from_pretrained(base_model, device_map="auto")
    #print(torch.cuda.is_available())
    #sys.exit(1)
    if device == "cuda":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model, 
            load_in_8bit=load_8bit,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        print("model cuda error")
        sys.eixt(1)
    #print(model.config.use_cache)
    #sys.exit(1)
    # unwind broken decapoda-research config
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    service_id = service_begin_id
    print(f"current service name: {dataset_order[service_id]}... begin generating!")
    
    output_file = os.path.join(output_dir, str(service_id)+"-"+dataset_order[service_id] +"_result.txt")
    print(f"output filename: {output_file}")
    
    testfile_name = "./data2/test/" + dataset_order[service_id] + "_T5.json"
    
    print(f"test filename: {testfile_name}")
    

    if not os.path.isfile(output_file): # 如果文件不存在，就新建文件，从头开始写入
        result_out = open(output_file, "w", encoding='utf-8')
        begin_id = 0 # 相当于从头开始获取结果
        print("——————————————————————————————从头开始写入——————————————————————————————")
    else: # 如果文件已经存在了，我们看看已经写了多少行了，需要继续从哪里开始获取结果
        with open(output_file, "r") as f:
            lines = f.readlines()
            begin_id = len(lines)
            f.close()
        print(f"——————————————————————————————从第{begin_id}行开始写入——————————————————————————————")
        result_out = open(output_file, "a", encoding='utf-8')
    
    data = json.load(open(testfile_name)) 
    for idx_ in range(begin_id, len(data)):
        sample = data[idx_]
        
        Response_list = []

        input_ids = tokenizer(sample['input'], return_tensors='pt').input_ids.cuda()
        output = model.generate(input_ids=input_ids, max_length=max_new_tokens)
        answer = tokenizer.decode(output[0])

        #print(answer)
        #sys.exit(1)
        if "</s>" in answer:
            answer = answer.replace("</s>","")
        if "<pad>" in answer:
            answer = answer.replace("<pad>","")
                
        Response_list.append(answer.strip())

        #print("Input:", input2)
        print("Input:", sample['input'])
        print("Response list:", Response_list)
        print("Ground truth:", sample['output'])
        print()
        # if "NONE" not in Response:
        #     break
        # if sample['output'] != "NONE":
        #     break
        result_out.write(sample['id'] + "|||" + sample['output'] + "|||" + str(Response_list))
            
        #result_out.write(idx_line + "|||" + str(Response_list))
        result_out.write("\n")

        #break
    result_out.close()
    print(f"current service name: {dataset_order[service_id]}... Generate End!")

if __name__ == "__main__":
    fire.Fire(main)
