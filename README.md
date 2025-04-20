项目名称
欢迎使用本项目！本项目提供了一个基于 PyTorch 和 gRPC 的关系预测模型后端服务。本文档将指导您完成环境配置、gRPC 文件生成以及接口使用的步骤。
📋 项目概述
本项目利用 PyTorch 构建深度学习模型，并通过 gRPC 提供高效的接口服务，用于预测句子中两个实体之间的关系类型。模型后端运行在 gRPC 端口 50051 上。
🚀 快速开始
1. 安装 PyTorch（支持 CUDA）
   为了加速国内的下载速度，我们使用清华大学镜像源安装 PyTorch、torchvision 和 torchaudio（支持 CUDA 12.4）：
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple


注意：请确保您的系统已安装 CUDA 12.4 以支持 GPU 加速。

2. 生成 gRPC Python 文件
   编译 relation.proto 文件以生成 gRPC 的 Python 存根代码：
   python -m grpc_tools.protoc -Iapp/protos --python_out=app/protos --grpc_python_out=app/protos app/protos/relation.proto

参数说明：

-Iapp/protos：指定 .proto 文件所在目录。
--python_out=app/protos：生成 Python 文件（如 relation_pb2.py）的输出目录。
--grpc_python_out=app/protos：生成 gRPC Python 文件（如 relation_pb2_grpc.py）的输出目录。
app/protos/relation.proto：.proto 文件路径。

3. gRPC 接口文档
   PredictRelation 接口
   请求参数
   message PredictRequest {
   string sentence = 1;  // 待分析的句子
   string entity1 = 2;   // 第一个实体的名称
   string entity2 = 3;   // 第二个实体的名称
   }

响应参数
message PredictResponse {
string relation = 1;   // 预测的关系类型
float confidence = 2;  // 置信度（0-1）
string error = 3;      // 错误信息（如果有）
}

4. 模型后端配置

gRPC 端口：50051

确保您的客户端代码连接到此端口以调用 PredictRelation 接口。
🛠️ 环境要求

Python 3.8+
CUDA 12.4（若使用 GPU）
依赖库：
torch
torchvision
torchaudio
grpcio
grpcio-tools



您可以通过以下命令安装 gRPC 相关依赖：
pip install grpcio grpcio-tools

📖 使用示例
以下是一个简单的 Python 客户端代码示例，用于调用 PredictRelation 接口：
import grpc
import relation_pb2
import relation_pb2_grpc

def predict_relation(sentence, entity1, entity2):
with grpc.insecure_channel('localhost:50051') as channel:
stub = relation_pb2_grpc.RelationServiceStub(channel)
request = relation_pb2.PredictRequest(
sentence=sentence,
entity1=entity1,
entity2=entity2
)
response = stub.PredictRelation(request)
return response.relation, response.confidence, response.error

# 示例调用
relation, confidence, error = predict_relation(
"张三和李四是朋友。",
"张三",
"李四"
)
print(f"关系: {relation}, 置信度: {confidence}, 错误: {error}")

🤝 贡献
欢迎提交 Issue 或 Pull Request 来改进本项目！请遵循以下步骤：

Fork 本仓库
创建您的特性分支 (git checkout -b feature/AmazingFeature)
提交您的更改 (git commit -m 'Add some AmazingFeature')
推送到分支 (git push origin feature/AmazingFeature)
提交 Pull Request

📄 许可证
本项目采用 MIT 许可证。
📬 联系我们
如有任何问题，请通过 GitHub Issues 联系我们。

⭐ 如果您觉得本项目有用，请给个 Star 支持一下！😊
