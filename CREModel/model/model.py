import argparse
from typing import Set, List

import torch
from transformers import AutoConfig
from transformers import AutoModel
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    """编码器类，用于通过预训练模型（如 BERT）生成文本特征"""

    def __init__(self, config: argparse.Namespace):
        """初始化 Encoder 类"""
        super(Encoder, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.bert_path, attn_implementation="sdpa")
        self.bert_config = AutoConfig.from_pretrained(config.bert_path)

        self.output_size = self.bert_config.hidden_size
        self.drop = nn.Dropout(config.drop_out)

        # 检查模式是否有效
        if config.pattern not in ['standard', 'entity_marker']:
            raise ValueError("Invalid pattern, must be 'standard' or 'entity_marker'")
        self.pattern = config.pattern

        # 根据模式选择不同的模型设置
        if self.pattern == 'entity_marker':
            original_vocab_size = self.encoder.config.vocab_size
            self.encoder.resize_token_embeddings(original_vocab_size + 4, mean_resizing=False)

            self.linear_transform = nn.Sequential(
                nn.Linear(self.bert_config.hidden_size * 2, self.bert_config.hidden_size, bias=True),
                nn.GELU(),
                nn.LayerNorm([self.bert_config.hidden_size])
            )
        else:
            self.linear_transform = nn.Linear(
                self.bert_config.hidden_size,
                self.output_size,
                bias=True
            )

    def get_output_size(self) -> int:
        """返回编码器输出的特征大小"""
        return self.output_size

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """前向传播，接受输入的 input_ids 和 attention_mask，返回编码后的特征"""

        if self.pattern == 'standard':
            outputs = self.encoder(inputs, attention_mask=attention_mask, output_hidden_states=True)
            pooled_output = torch.stack(outputs.pooler_outputs[-4:]).mean(dim=0)
            feature = self.drop(pooled_output)
        else:
            e11_pos, e21_pos = [], []
            for i in range(inputs.size(0)):
                tokens = inputs[i].cpu().numpy()

                e11_pos.append(np.argwhere(tokens == self.config.e11_idx)[0][0])
                e21_pos.append(np.argwhere(tokens == self.config.e21_idx)[0][0])

            outputs = self.encoder(inputs, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state

            e11_features = hidden_states[torch.arange(inputs.size(0)), e11_pos]
            e21_features = hidden_states[torch.arange(inputs.size(0)), e21_pos]

            feature = torch.cat([e11_features, e21_features], dim=1)
            feature = self.linear_transform(self.drop(feature))

        return feature


class RelationClassifier(nn.Module):
    """关系分类器模型，用于关系抽取任务"""

    def __init__(self, config: argparse.Namespace, encoder: nn.Module, num_relations: int):
        """初始化 RelationClassifier 类"""

        super(RelationClassifier, self).__init__()

        self.config = config
        self.encoder = encoder

        self.classifier = torch.nn.Linear(encoder.get_output_size(), num_relations)

        self.learned_relations: Set[str] = set()
        self.relation_masks: dict = {}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """前向传播，输入文本的 input_ids 和 attention_mask，输出分类结果和特征"""

        features = self.encoder(input_ids, attention_mask)
        return self.classifier(features), features

    def add_relations(self, new_relations: List[str]):
        """动态地添加新的关系类别，并扩展分类器的输出大小"""

        old_size = self.classifier.out_features
        new_size = old_size + len(new_relations)

        new_classifier = torch.nn.Linear(
            self.classifier.in_features,
            new_size
        ).to(self.classifier.weight.device)

        with torch.no_grad():
            new_classifier.weight[:old_size] = self.classifier.weight
            new_classifier.bias[:old_size] = self.classifier.bias

        self.classifier = new_classifier
        self.learned_relations.update(new_relations)
