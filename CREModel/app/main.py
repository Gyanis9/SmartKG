import argparse
import torch
import grpc
import logging
from concurrent import futures
from transformers import AutoTokenizer

from model.model import Encoder, RelationClassifier
from services.relation_service import RelationServicer
from CREModel.app.protos import relation_pb2_grpc


def serve(config):
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    # 添加特殊标记（必须与训练时一致）
    special_tokens = ["[E11]", "[E12]", "[E21]", "[E22]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # 初始化模型
    encoder = Encoder(config)
    model = RelationClassifier(config, encoder, num_relations=config.relations)
    state = torch.load(config.model_checkpoint)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", default="./bert-base-uncased")
    parser.add_argument("--model_checkpoint", default="./checkpoints/best_model.pth")
    parser.add_argument('--drop_out', default=0.1, type=float,
                        help='dropout比率')
    parser.add_argument('--pattern', default='entity_marker', type=str)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--relations", type=int, default=42)
    config = parser.parse_args()

    # 启动服务
    logging.basicConfig(level=logging.INFO)
    serve(config)
