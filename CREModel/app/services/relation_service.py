from protos import relation_pb2, relation_pb2_grpc
import numpy as np
import logging
import torch
from protos import relation_pb2, relation_pb2_grpc


class RelationServicer(relation_pb2_grpc.RelationServiceServicer):
    def __init__(self, config, model, tokenizer):
        """初始化 RelationServicer 类，加载配置、模型和tokenizer"""

        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        # 初始化 label2id 和 id2label 映射
        self.id2label = {i: rel for i, rel in enumerate(self.model.learned_relations)}
        self.label2id = {rel: i for i, rel in enumerate(self.model.learned_relations)}

        # 如果有 GPU 可用，将模型转移到 GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # 设置为评估模式，不进行梯度计算
        self.model.eval()

    def _find_entity_positions(self, sentence: str, entity_text: str):
        """查找实体在句子中的位置"""

        start = sentence.find(entity_text)
        if start == -1:
            return None
        return (start, start + len(entity_text))

    def _insert_markers(self, sentence: str, entity1: str, entity2: str):
        """自动查找实体位置并插入标记，并处理偏移量"""

        # 查找实体在句子中的位置
        e1_pos = self._find_entity_positions(sentence, entity1)
        e2_pos = self._find_entity_positions(sentence, entity2)

        # 如果未找到实体位置，返回 None
        if not e1_pos or not e2_pos:
            return None

        # 确定实体处理顺序（从后到前）
        entities = []
        if e1_pos[0] > e2_pos[0]:
            entities.append(("E2", e2_pos))
            entities.append(("E1", e1_pos))
        else:
            entities.append(("E1", e1_pos))
            entities.append(("E2", e2_pos))

        # 初始化带标记的句子
        marked_sentence = sentence
        offset = 0  # 维护插入标记导致的偏移量

        # 插入实体标记
        for entity_type, (orig_start, orig_end) in entities:
            # 计算调整后的位置
            adj_start = orig_start + offset
            adj_end = orig_end + offset

            # 插入开始标记
            start_marker = f"[{entity_type}1]"
            marked_sentence = (
                    marked_sentence[:adj_start]
                    + start_marker
                    + marked_sentence[adj_start:]
            )
            offset += len(start_marker)

            # 更新结束位置（因为插入开始标记后，原结束位置后移）
            adj_end += len(start_marker)

            # 插入结束标记
            end_marker = f"[{entity_type}2]"
            marked_sentence = (
                    marked_sentence[:adj_end]
                    + end_marker
                    + marked_sentence[adj_end:]
            )
            offset += len(end_marker)

        return marked_sentence

    def PredictRelation(self, request, context):
        """根据输入的句子和实体，预测实体之间的关系"""

        try:
            # 1. 处理输入
            sentence = request.sentence
            entity1 = request.entity1
            entity2 = request.entity2

            # 2. 插入实体标记
            marked_text = self._insert_markers(sentence, entity1, entity2)
            if not marked_text:
                return relation_pb2.PredictResponse(
                    error="Entities not found in sentence"
                )

            # 3. Tokenize
            inputs = self.tokenizer(
                marked_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # 4. 转换到GPU（如果可用）
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 5. 推理
            with torch.no_grad():
                logits, _ = self.model(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # 6. 获取预测结果
            pred_idx = np.argmax(probs)  # 获取预测结果的索引
            confidence = float(probs[pred_idx])  # 获取预测的置信度
            relation = self.id2label.get(pred_idx, "unknown")  # 获取预测的关系标签

            # 返回预测结果
            return relation_pb2.PredictResponse(
                relation=relation,
                confidence=confidence
            )

        except Exception as e:
            # 错误处理
            logging.error(f"Prediction error: {str(e)}")
            return relation_pb2.PredictResponse(
                error=f"Internal server error: {str(e)}"
            )
