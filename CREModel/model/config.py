import argparse
from typing import Any


def get_config() -> Any:
    """获取命令行参数配置并返回"""

    parser = argparse.ArgumentParser()

    # =========================================模型路径和存储路径================================================
    parser.add_argument('--bert_path', default='model/bert-base-uncased', type=str,
                        help='BERT模型的路径')
    parser.add_argument('--save_dir', default='model/checkpoints', type=str,
                        help='模型检查点的保存路径')
    parser.add_argument('--data_path', default='model/data', type=str,
                        help='训练数据的路径')
    parser.add_argument('--pattern', default='entity_marker', type=str)

    # ==========================================超参数设置=======================================================
    # 注意：此类超参数通常影响模型训练过程中的优化和正则化
    parser.add_argument('--drop_out', default=0.1, type=float,
                        help='dropout比率')
    parser.add_argument('--relations_per_batch', default=4, type=int,
                        help='每个批次处理的关系数量')
    parser.add_argument('--encoder_lr', default=1e-5, type=float,
                        help='编码器学习率')
    parser.add_argument('--classifier_lr', default=1e-3, type=float,
                        help='分类器学习率')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='批次大小')

    parser.add_argument('--max_grad_norm', default=10.0, type=float,
                        help='每个批次处理的关系数量')
    parser.add_argument('--memory_epochs', default=0, type=int,
                        help='训练轮次')
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='训练轮次')

    # =========================================训练的早停设置=====================================================
    parser.add_argument('--patience', default=10, type=int,
                        help='早停耐心步数')

    # =========================================训练的总步骤======================================================
    parser.add_argument('--seed', default=2025, type=int,
                        help='随机种子')

    config = parser.parse_args()

    return config
