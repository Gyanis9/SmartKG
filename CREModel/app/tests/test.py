import grpc
from CREModel.app.protos import relation_pb2_grpc, relation_pb2


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = relation_pb2_grpc.RelationServiceStub(channel)

    # 正常请求测试
    print("Test Case 1: Normal Request")
    request = relation_pb2.PredictRequest(
        sentence="Microsoft was founded by Bill Gates in 1975.",
        entity1="Microsoft",
        entity2="Bill Gates"
    )
    response = stub.PredictRelation(request)
    print_response(response)

    # 错误请求测试（实体不存在）
    print("\nTest Case 2: Missing Entity")
    request = relation_pb2.PredictRequest(
        sentence="A popular tech company in Silicon Valley.",
        entity1="Google",
        entity2="Sundar Pichai"
    )
    response = stub.PredictRelation(request)
    print_response(response)

    # 边界测试（长文本）
    print("\nTest Case 3: Long Text")
    long_text = "Apple Inc., founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976, " + \
                "is headquartered in Cupertino, California. "
    request = relation_pb2.PredictRequest(
        sentence=long_text,
        entity1="Apple Inc.",
        entity2="Cupertino"
    )
    response = stub.PredictRelation(request)
    print_response(response)

def print_response(response):
    if response.error:
        print(f"Error: {response.error}")
    else:
        print(f"Relation: {response.relation}")
        print(f"Confidence: {response.confidence:.2f}")
        print("="*50)

if __name__ == '__main__':
    run()