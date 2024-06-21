import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# import torch.optim.adamw
import model_args as args

class MyOptimizer:
    def __init__(self, model_parameters, steps):
        """
        初始化MyOptimizer类。

        参数:
        model_parameters (dict): 一个包含模型参数的字典，键是参数名称，值是参数张量。
        steps (int): 训练步骤的总数，用于学习率调度器。
        """
        self.optims = []  # 用于存储不同的优化器
        self.schedulers = []  # 用于存储不同的学习率调度器

        for name, parameters in model_parameters.items():
            if name.startswith("bert"):
                # 对于名称以"bert"开头的参数，使用AdamW优化器和线性预热调度器
                optim_bert = AdamW(parameters, args.bert_training_lr, eps=1e-8)
                self.optims.append(optim_bert)

                scheduler_bert = get_linear_schedule_with_warmup(optim_bert, 0, steps)
                self.schedulers.append(scheduler_bert)

            elif name.startswith("basic"):
                # 对于名称以"basic"开头的参数，使用Adam优化器和线性预热调度器
                optim = torch.optim.Adam(parameters, lr=args.training_learning_rate, weight_decay=1e-2)
                self.optims.append(optim)

                scheduler = get_linear_schedule_with_warmup(optim, 0.1, steps)
                self.schedulers.append(scheduler)
            else:
                # 如果参数名称不符合以上两种情况，抛出异常
                raise Exception("not found parameter dict", name)

        self.num = len(self.optims)  # 记录优化器的数量

    def step(self):
        """
        执行所有优化器和调度器的step操作，并将梯度清零。
        """
        for optim, scheduler in zip(self.optims, self.schedulers):
            optim.step()  # 更新模型参数
            scheduler.step()  # 更新学习率
            optim.zero_grad()  # 清零梯度

    def zero_grad(self):
        """
        清零所有优化器的梯度。
        """
        for optim in self.optims:
            optim.zero_grad()

    def get_lr(self):
        """
        获取当前学习率。

        返回:
        str: 当前学习率的字符串表示。
        """
        lrs = tuple(map(lambda x: x.get_lr()[-1], self.schedulers))  # 获取每个调度器的当前学习率
        lr = ' %.5f' * self.num  # 构造格式化字符串
        res = lr % lrs  # 格式化学习率
        return res

