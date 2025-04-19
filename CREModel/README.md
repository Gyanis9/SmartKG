pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -i https://pypi.tuna.tsinghua.edu.cn/simple


proto命令
python -m grpc_tools.protoc -Iapp/protos --python_out=app/protos --grpc_python_out=app/protos app/protos/relation.proto


接口文档
### PredictRelation

- **请求参数**:
  ```protobuf
  message PredictRequest {
    string sentence = 1;  // 待分析的句子
    string entity1 = 2;    // 实体1名称
    string entity2 = 3;    // 实体2名称
  }

- **响应参数**
  ```protobuf
  message PredictResponse {
  string relation = 1;   // 预测的关系类型
  float confidence = 2;  // 置信度（0-1）
  string error = 3;      // 错误信息
  }


模型后端Grpc端口号:50051
