from torch.utils.data import Dataset
import torch

import json
from typing import List, Tuple, Dict


class TACREDProcessor:
    """处理 TACRED 数据集的类，包括数据加载、清洗以及实体标记的添加"""

    @staticmethod
    def load_data(file_path: str) -> List[Dict]:
        """从文件加载数据"""
        with open(file_path) as f:
            return json.load(f)

    @staticmethod
    def clean_data(data: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """清洗数据，去除无效样本和不需要的关系类型。"""
        cleaned = []
        valid_relations = set()

        for item in data:
            if not all([item['subj_start'], item['obj_start'], item['relation']]):
                continue

            relation = item['relation'].split('.')[0]

            # if relation == 'no_relation' or relation == 'per:country_of_death':
            #     continue

            cleaned.append({
                'tokens': item['token'],
                'subj_start': item['subj_start'],
                'subj_end': item['subj_end'],
                'obj_start': item['obj_start'],
                'obj_end': item['obj_end'],
                'relation': relation
            })
            valid_relations.add(relation)
        return cleaned, list(valid_relations)

    @staticmethod
    def add_special_tokens(tokens: List[str], subj_start: int, subj_end: int, obj_start: int, obj_end: int) -> List[
        str]:
        """在输入的 token 列表中添加实体标记，标记主语和宾语"""
        new_tokens = []

        # 遍历所有的 token，并根据位置添加实体标记
        for i, token in enumerate(tokens):
            if i == subj_start:
                new_tokens.extend(['[E11]', token])
            elif i == subj_end and subj_start != subj_end:
                new_tokens.extend([token, '[E12]'])
            elif i == obj_start:
                new_tokens.extend(['[E21]', token])
            elif i == obj_end and obj_start != obj_end:
                new_tokens.extend([token, '[E22]'])
            else:
                new_tokens.append(token)

        if subj_start == subj_end:
            new_tokens.insert(subj_end + 1, '[E12]')
        if obj_start == obj_end:
            new_tokens.insert(obj_end + 1, '[E22]')

        return new_tokens


class TACREDDataset(Dataset):
    """TACRED 数据集类，用于处理 TACRED 数据集，基于 PyTorch 的 Dataset 类进行封装"""

    def __init__(self, data: List[Dict], tokenizer, relation_to_id: Dict[str, int]):
        """初始化 TACREDDataset 类"""
        self.data = data
        self.tokenizer = tokenizer
        self.relation_to_id = relation_to_id

    def __len__(self) -> int:
        """返回数据集的大小，即数据样本的数量"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """获取数据集中指定索引的数据样本，进行处理并返回"""
        item = self.data[idx]
        tokens = TACREDProcessor.add_special_tokens(
            item['tokens'],
            item['subj_start'],
            item['subj_end'],
            item['obj_start'],
            item['obj_end']
        )

        encoding = self.tokenizer(
            ' '.join(tokens),
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.relation_to_id[item['relation']])
        }
