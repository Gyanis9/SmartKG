from protos import relation_pb2, relation_pb2_grpc
import numpy as np
import logging

import torch


class RelationServicer(relation_pb2_grpc.RelationServiceServicer):
    def __init__(self, config, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = {i: rel for i, rel in enumerate(self.model.learned_relations)}
        self.label2id = {rel: i for i, rel in enumerate(self.model.learned_relations)}
        print(f'relations id: {self.model.learned_relations}')

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def _find_entity_positions(self, sentence: str, entity_text: str):
        """查找实体在句子中的位置"""
        start = sentence.find(entity_text)
        if start == -1:
            return None
        return (start, start + len(entity_text))

    def _insert_markers(self, sentence: str, entity1: str, entity2: str):
        """自动查找实体位置并插入标记，并处理偏移量"""
        e1_pos = self._find_entity_positions(sentence, entity1)
        e2_pos = self._find_entity_positions(sentence, entity2)

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

        marked_sentence = sentence
        offset = 0  # 维护插入标记导致的偏移量

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
            print(f'marked_text: {marked_text}')
            # 3. Tokenize
            inputs = self.tokenizer(
                marked_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # 4. 转换到GPU
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 5. 推理
            with torch.no_grad():
                logits, _ = self.model(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            # 6. 获取预测结果
            pred_idx = np.argmax(probs)
            confidence = float(probs[pred_idx])
            print(f'confidence: {confidence}')
            print(f'pred_idx: {pred_idx}')
            print(f'relations: {self.id2label}')
            relation = self.id2label.get(pred_idx, "unknown")

            return relation_pb2.PredictResponse(
                relation=relation,
                confidence=confidence
            )

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return relation_pb2.PredictResponse(
                error=f"Internal server error: {str(e)}"
            )
