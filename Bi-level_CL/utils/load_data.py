
from datasets import concatenate_datasets, load_dataset
from utils.dataset_order import get_dataset_order
import os
import sys
import random
from datasets import Dataset

def load_current_task_data(dataset_id, task_id, data_dir, cache_dir=None):
    """
    加载当前任务的数据

    参数:
    - task_id (int): 当前任务的 ID，用于指定要加载的文件。
    - data_dir (str): 数据的根目录，存放每个任务的数据文件。
    - cache_dir (str, 可选): 数据缓存目录，可避免重复加载。

    返回:
    - Dataset: 返回 Hugging Face 格式的 Dataset 对象。
    """
    # 构建当前任务数据文件路径
    dataset_order = get_dataset_order(dataset_id)
    
    data_path = os.path.join(data_dir, "train", dataset_order[task_id] + "_T5.json")
    print(f"current data path: {data_path}")
    assert os.path.exists(data_path), "data_path not find!"


    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=data_path,cache_dir=cache_dir)['train']
    else:
        dataset = load_dataset(data_path,cache_dir=cache_dir)['train']
    print(f"总样本数：{len(dataset)}")

    return dataset

def load_memory_buffer(dataset_id, task_id, data_dir, sampling_ratio, cache_dir, random_seed: int = 42):
    """
    从历史任务中加载数据作为 memory buffer。

    Args:
        task_id (int): 当前任务的 ID memory buffer 只包括从 task 1 到 task (task_id-1) 的数据。
        sampling_ratio (int): 每个任务数据的采样比例，取值范围在 0 和 100 之间。
        random_seed (int): 随机种子，保证采样的一致性。

    Returns:
        MemoryBufferDataset: 包含采样数据的 memory buffer 数据集。
    """
    random.seed(random_seed)
    dataset_order = get_dataset_order(dataset_id)
    buffer_data = []

    for i in range(task_id):
        # 加载每个历史任务的数据文件
        data_path = os.path.join(data_dir, "train", dataset_order[i] + "_T5.json")
        print(f"task id: {i}, history data path: {data_path}")
        assert os.path.exists(data_path), "data_path not find!"

        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            task_data = load_dataset("json", data_files=data_path,cache_dir=cache_dir)["train"]
        else:
            task_data = load_dataset(data_path,cache_dir=cache_dir)["train"]

        # 根据采样比例进行采样
        num_samples = int(len(task_data) * sampling_ratio / 100)
        sampled_data = task_data.shuffle(seed=random_seed).select(range(num_samples))

        # 将采样数据添加到 memory buffer 中
        buffer_data.append(sampled_data)
        print(f"总样本数：{len(task_data)}, 选择的样本数量：{num_samples}")

    # 将所有采样数据转为 Hugging Face 的 Dataset 格式
    memory_buffer = buffer_data[0]  # 先使用第一个数据集
    for dataset in buffer_data[1:]:
        memory_buffer = concatenate_datasets([memory_buffer, dataset])
    return memory_buffer
