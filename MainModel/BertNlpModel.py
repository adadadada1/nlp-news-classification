import torch
import torch.nn as nn
import model_args as args


class BertNlpModel(torch.nn.Module):
    def __init__(self, preTrain):
        """
        初始化BertNlpModel类。

        参数:
        preTrain (transformers.PreTrainedModel): 预训练的BERT模型。
        """
        super(BertNlpModel, self).__init__()
        self.preTrain = preTrain
        hidden_size = 512  # 隐藏层大小
        class_num = 14  # 分类数量

        # 定义网络层
        self.dense_1 = nn.Linear(hidden_size, hidden_size)  # 全连接层1
        self.layerNorm = nn.LayerNorm(hidden_size)  # 层归一化
        self.gelu = nn.GELU()  # GELU激活函数
        self.dense_2 = nn.Linear(hidden_size, class_num)  # 全连接层2

        # 定义多个Dropout层
        self.dropout_ops = nn.ModuleList(
            [nn.Dropout() for _ in range(args.dropout_num)]
        )

        self.dropout = nn.Dropout(0.1)  # 单个Dropout层

        # 收集所有需要训练的参数
        self.all_parameters = {}
        parameters = []
        parameters.extend(list(filter(lambda p: p.requires_grad, self.dense_1.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.layerNorm.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.dense_2.parameters())))
        if len(parameters) > 0:
            self.all_parameters["basic_parameters"] = parameters

        self.all_parameters["bert_parameters"] = self.preTrain.parameters()

    def train(self):
        """
        设置模型为训练模式。
        """
        super(BertNlpModel, self).train()
        self.preTrain.train()

    def eval(self):
        """
        设置模型为评估模式。
        """
        super(BertNlpModel, self).eval()
        self.preTrain.eval()

    def forward(self, input_ids, segment_ids, input_mask):
        """
        前向传播函数。

        参数:
        input_ids (torch.Tensor): 输入的token ID。
        segment_ids (torch.Tensor): 输入的segment ID。
        input_mask (torch.Tensor): 输入的mask。

        返回:
        torch.Tensor: 模型的输出。
        """
        # 通过预训练的BERT模型获取隐藏状态
        hidden_states, _ = self.preTrain.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)

        # 通过自定义的全连接层和激活函数
        hidden_states = self.gelu(self.dense_1(hidden_states))
        hidden_states = self.layerNorm(hidden_states)

        # 对隐藏状态求平均
        hidden_states = torch.mean(hidden_states, dim=1, keepdim=False)

        # 根据条件选择是否使用多个Dropout层
        if args.multi_dropout:
            for i, dropout_op in enumerate(self.dropout_ops):
                if i == 0:
                    out = dropout_op(hidden_states)
                else:
                    temp_out = dropout_op(hidden_states)
                    out += temp_out
            hidden_states = out / args.dropout_num
        else:
            hidden_states = self.dropout(hidden_states)

        # 通过最终的全连接层获取输出
        output = self.dense_2(hidden_states)

        return output
