import argparse
import torch
import grpc
import logging
from concurrent import futures
from transformers import AutoTokenizer
from typing import Any

from model.model import Encoder, RelationClassifier
from services.relation_service import RelationServicer
from CREModel.app.protos import relation_pb2_grpc


def serve(config: argparse.Namespace) -> None:
    """启动gRPC服务，加载模型，并准备tokenizer"""

    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    special_tokens: list[str] = ["[E11]", "[E12]", "[E21]", "[E22]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # 初始化模型
    encoder = Encoder(config)
    model = RelationClassifier(config, encoder, num_relations=config.relations)

    # 加载模型检查点
    state: dict[str, Any] = torch.load(config.model_checkpoint)
    model.load_state_dict(state["model"])
    model.learned_relations = state["relations"]

    # 创建gRPC服务
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    relation_pb2_grpc.add_RelationServiceServicer_to_server(
        RelationServicer(config, model, tokenizer),
        server
    )
    server.add_insecure_port(f'[::]:{config.port}')
    server.start()
    print(f"Server started on port {config.port}")
    server.wait_for_termination()


if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_path", default="./bert-base-uncased", type=str, help="BERT模型的路径")
    parser.add_argument("--model_checkpoint", default="./checkpoints/best_model.pth", type=str,
                        help="训练好的模型检查点路径")
    parser.add_argument('--drop_out', default=0.1, type=float,
                        help='dropout比率，用于控制神经网络中随机失活的神经元比例')
    parser.add_argument('--pattern', default='entity_marker', type=str, help='标记模式')
    parser.add_argument("--port", type=int, default=50051, help="gRPC服务监听的端口号")
    parser.add_argument("--relations", type=int, default=42, help="关系的数量，用于初始化模型的输出层")

    config = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 启动gRPC服务
    serve(config)
