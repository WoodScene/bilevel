from transformers import Trainer
import torch

class BiLevelTrainer(Trainer):
    def __init__(self, 
                 *args, 
                 train_dataset_memory=None,   # memory buffer 数据
                 inner_lr=3e-4,      # 内层学习率
                 outer_lr=1e-3,      # 外层学习率
                 inner_iterations=5, # 内层循环迭代次数
                 **kwargs):
        """
        初始化 BiLevelTrainer，支持 bi-level 优化的训练。

        :param current_data: 当前任务的数据集，用于内层训练
        :param memory_data: memory buffer 数据，用于外层训练
        :param inner_lr: 内层循环的学习率
        :param outer_lr: 外层循环的学习率
        :param inner_iterations: 内层训练的迭代次数
        """
        super().__init__(*args, **kwargs)
        
        # 设置memory buffer数据集
        self.train_dataset_memory = train_dataset_memory
        
        # 设置内外层学习率和内层训练的迭代次数
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_iterations = inner_iterations
        self.inner_bs = self.batch_size / self.inner_iterations

        
        # 内外层优化器初始化为空，后续可以根据需要进行创建
        self.inner_optimizer = None
        self.outer_optimizer = None

        # 其它可能需要的初始化操作


    def training_step(self, model, inputs):
        """
        在每个训练步骤中执行 bi-level 优化，包括内层和外层更新。

        :param model: 当前的模型
        :param inputs: 当前的输入 batch
        :return: 外层 loss
        """
        # Step 1: 内层循环优化 - 使用 current_dataset
        self.model.train()
        for _ in range(self.inner_iterations):
            current_data = self._prepare_inputs(inputs)  # 准备当前任务的数据
            
            # 计算内层 loss 并进行反向传播
            inner_loss = self.compute_loss(model, current_data)
            inner_loss.backward()
            self.inner_optimizer.step()
            self.inner_optimizer.zero_grad()

        # Step 2: 外层循环优化 - 使用 memory buffer
        memory_data = next(iter(self.train_dataset_memory))  # 获取 memory buffer 数据
        memory_data = self._prepare_inputs(memory_data)

        # 计算外层 loss 并进行反向传播
        outer_loss = self.compute_loss(model, memory_data)
        outer_loss.backward()
        self.outer_optimizer.step()
        self.outer_optimizer.zero_grad()

        return outer_loss  # 返回外层 loss 作为该步的优化结果

    def train(self):
        """
        重写 train 方法，以支持交替进行内外循环的优化。
        """
        for epoch in range(int(self.args.num_train_epochs)):
            for step, batch in enumerate(self.get_train_dataloader()):
                # 训练步骤会被劫持到 `training_step`，进行 bi-level 优化
                loss = self.training_step(self.model, batch)
                self.log({'train_loss': loss.item()})