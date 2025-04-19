import os
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, pairwise_distances_argmin_min
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from model import Encoder, RelationClassifier
from sampler import TACREDProcessor, TACREDDataset

# ANSI颜色代码
COLORS = {
    "default": "\033[0m",
    "blue": "\033[1;34;40m",
    "yellow": "\033[1;33;40m",
    "green": "\033[1;32;40m",
}


class BaseTrainer(ABC):
    """训练器基类，包含公共方法和抽象接口"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_components()

    def _init_components(self):
        """初始化基础组件"""

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.bert_path)
        self._add_special_tokens()
        self._prepare_datasets()
        self.model = self._build_model().to(self.device)
        self.optimizer = self._configure_optimization()

    def _add_special_tokens(self):
        """添加特殊token"""

        new_tokens = ['[E11]', '[E12]', '[E21]', '[E22]']
        self.tokenizer.add_tokens(new_tokens)
        self.config.e11_idx = self.tokenizer.convert_tokens_to_ids('[E11]')
        self.config.e12_idx = self.tokenizer.convert_tokens_to_ids('[E12]')
        self.config.e21_idx = self.tokenizer.convert_tokens_to_ids('[E21]')
        self.config.e22_idx = self.tokenizer.convert_tokens_to_ids('[E22]')

    def _prepare_datasets(self):
        """准备所有数据集"""

        self.raw_datasets = {
            split: TACREDProcessor.load_data(
                os.path.join(self.config.data_path, f'{split}.json')
            ) for split in ['train', 'dev', 'test']
        }

        # 数据清洗
        relations = list()
        self.datasets = {}
        for split, data in self.raw_datasets.items():
            clean_data, relations = TACREDProcessor.clean_data(data)
            self.datasets[split] = clean_data
        self.all_relations = relations

    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        """构建模型，子类必须实现"""

        pass

    def _configure_optimization(self) -> AdamW:
        """配置优化器和学习率调度器"""

        # 分层学习率
        optimizer_params = [
            {"params": self.model.encoder.parameters(), "lr": self.config.encoder_lr},
            {"params": self.model.classifier.parameters(), "lr": self.config.classifier_lr},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)

        return optimizer

    def create_dataloader(
            self,
            data: List[Dict],
            relations: List[str],
            batch_size: int = 16,
            shuffle: bool = True,
            pin_memory: bool = True,
            num_workers: int = 8,
            persistent_workers: bool = True
    ) -> DataLoader:
        """创建一个数据加载器（DataLoader），用于从给定数据集中加载训练数据"""

        filtered = [d for d in data if d["relation"] in relations]

        relation_map = {r: idx for idx, r in enumerate(self.model.learned_relations)}

        dataset = TACREDDataset(filtered, self.tokenizer, relation_map)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers
        )

    def train_epoch(
            self,
            loader: DataLoader,
            desc: str = "Training"
    ) -> float:
        """通用训练epoch"""

        self.model.train()
        total_loss = 0.0
        optimizer_params = [
            {"params": self.model.encoder.parameters(), "lr": self.config.encoder_lr},
            {"params": self.model.classifier.parameters(), "lr": self.config.classifier_lr},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)

        for batch in tqdm(loader, desc=desc):
            inputs = batch["input_ids"].to(self.device, non_blocking=True)
            masks = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            optimizer.zero_grad()
            outputs, _ = self.model(inputs, masks)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader: DataLoader) -> float:
        """模型评估"""

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                inputs = batch["input_ids"].to(self.device, non_blocking=True)
                masks = batch["attention_mask"].to(self.device, non_blocking=True)

                outputs, _ = self.model(inputs, masks)
                preds = torch.argmax(outputs, dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].numpy())

        return accuracy_score(all_labels, all_preds)


class IncrementalTrainer(BaseTrainer):
    """支持增量学习的训练器"""

    def __init__(self, config):
        super().__init__(config)
        self.memory = []
        self.relation_metrics = {
            "current": [],
            "historical": []
        }

    def _build_model(self) -> RelationClassifier:
        """构建增量学习模型"""

        encoder = Encoder(self.config)
        return RelationClassifier(self.config, encoder, 0)  # 初始0个关系

    def run(self):
        """执行完整训练流程"""

        random.shuffle(self.all_relations)

        for task_id in range(0, len(self.all_relations), self.config.relations_per_batch):
            relations = self.all_relations[
                        task_id: task_id + self.config.relations_per_batch
                        ]
            self._print_task_header(task_id // self.config.relations_per_batch + 1, relations)

            self._train_task(relations, task_id // self.config.relations_per_batch + 1)
            break

    @staticmethod
    def _print_task_header(task_num: int, relations: List[str]):
        """打印任务头信息"""

        color = COLORS["yellow"]
        print(f"{color}\nTraining task {task_num}: {relations}{COLORS['default']}")

    def _train_task(self, relations: List[str], task_id: int):
        """训练单个任务"""

        # 添加新关系
        self.model.add_relations(self.all_relations)

        # 创建数据加载器
        train_loader = self.create_dataloader(data=self.datasets["train"], relations=self.all_relations,
                                              batch_size=self.config.batch_size)
        dev_loader = self.create_dataloader(data=self.datasets["dev"], relations=self.all_relations,
                                            batch_size=32, shuffle=False)
        test_loader = self.create_dataloader(data=self.datasets["test"], relations=self.all_relations,
                                             batch_size=32,
                                             shuffle=False)

        # 阶段1: 训练新关系
        self._train_new_relations(train_loader, dev_loader, relations, task_id)

        # # 阶段2: 记忆训练
        # memory_loader = self._update_memory(relations, 10)
        # if memory_loader:
        #     self._train_memory(memory_loader)

        # 阶段3: 评估
        self._final_evaluation(test_loader)

    def _train_new_relations(
            self,
            train_loader: DataLoader,
            dev_loader: DataLoader,
            relations: List[str],
            task_id: int
    ) -> float:
        """训练新关系"""

        best_acc = 0.0
        patience = 0

        for epoch in range(self.config.num_epochs):
            avg_loss = self.train_epoch(train_loader, f"Training {relations}")

            # 验证
            current_acc = self.evaluate(dev_loader)
            print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={current_acc:.2%}")

            if current_acc > best_acc:
                best_acc = current_acc
                patience = 0
                self._save_checkpoint(task_id, is_best=True)
            else:
                patience += 1
                if patience >= self.config.patience:
                    print(f"{COLORS['yellow']}Early stopping at epoch {epoch + 1}{COLORS['default']}")
                    self._load_checkpoint(task_id)
                    break

        return best_acc

    def _update_memory(
            self,
            relations: List[str],
            samples_per_relation: int = 10
    ) -> DataLoader:
        """更新记忆样本并返回数据加载器"""

        # 获取当前所有已学关系
        all_relations = self.model.learned_relations

        # 选择典型样本
        new_samples = self._select_typical_samples(relations, samples_per_relation)
        self.memory.extend(new_samples)

        return self.create_dataloader(data=self.memory, relations=all_relations, batch_size=1, shuffle=False,
                                      pin_memory=False, persistent_workers=False)

    def _select_typical_samples(
            self,
            relations: List[str],
            k: int
    ) -> List[Dict]:
        """选择典型样本"""

        features, samples = self._extract_features(relations)
        centroids = self._cluster_features(features, k)
        return [samples[int(idx)] for idx in centroids]

    def _extract_features(
            self,
            relations: List[str]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """提取样本特征"""

        self.model.eval()
        features = []

        # 过滤数据
        filtered = [d for d in self.datasets["train"] if d["relation"] in relations]
        loader = self.create_dataloader(data=filtered, relations=relations, batch_size=1, shuffle=False,
                                        pin_memory=False, persistent_workers=False)

        with torch.no_grad():
            for batch in loader:
                inputs = batch["input_ids"].to(self.device)
                masks = batch["attention_mask"].to(self.device)

                # 获取CLS特征
                embeddings = self.model.encoder(inputs, masks).cpu().numpy()
                features.append(embeddings)

        return np.concatenate(features), filtered

    @staticmethod
    def _cluster_features(features: np.ndarray, k: int) -> List[int]:
        """特征聚类并返回中心样本索引"""
        actual_k = min(k, len(features))
        if actual_k < 1:
            return []

        kmeans = KMeans(n_clusters=actual_k, random_state=0).fit(features)
        _, indices = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
        return indices.tolist()

    def _train_memory(self, memory_loader: DataLoader):
        """记忆训练"""
        for _ in range(self.config.memory_epochs):
            self.train_epoch(memory_loader, "Memory Training")

    def _final_evaluation(self, test_loader: DataLoader):
        """最终评估"""
        current_acc = self.evaluate(test_loader)
        historical_acc = self.evaluate_all_learned()

        self.relation_metrics["current"].append(current_acc)
        self.relation_metrics["historical"].append(historical_acc)

        print(f"{COLORS['green']}Current Accuracy: {self.relation_metrics['current']}{COLORS['default']}")
        print(f"{COLORS['green']}Historical Accuracy: {self.relation_metrics['historical']}{COLORS['default']}")

    def evaluate_all_learned(self) -> float:
        """评估所有已学关系"""
        all_relations = self.model.learned_relations
        test_loader = self.create_dataloader(data=self.datasets["test"], relations=all_relations, batch_size=8,
                                             shuffle=False, pin_memory=False, persistent_workers=False)
        return self.evaluate(test_loader)

    def _save_checkpoint(self, task_id: int, is_best: bool = False):
        """保存检查点"""
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "memory": self.memory,
            "metrics": self.relation_metrics
        }
        suffix = "best" if is_best else "latest"
        path = os.path.join(self.config.save_dir, f"task{task_id}_{suffix}.pt")
        torch.save(state, path)

    def _load_checkpoint(self, task_id: int):
        """加载检查点"""
        path = os.path.join(self.config.save_dir, f"task{task_id}_best.pt")
        state = torch.load(path)

        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.memory = state.get("memory", [])
        self.relation_metrics = state.get("metrics", {"current": [], "historical": []})
