#include "RelationClient.h"
#include "base/Log.h"

namespace CRE
{
    static auto g_logger = LOG_NAME("system");

    RelationClient::RelationClient(const std::shared_ptr<grpc::Channel>& channel): m_stub(
        protos::RelationService::NewStub(channel))
    {
    }

    std::string RelationClient::PredictRelation(const std::string& sentence, const std::string& entity1,
                                                const std::string& entity2, float* confidence) const
    {
        protos::PredictRequest request;
        request.set_sentence(sentence);
        request.set_entity1(entity1);
        request.set_entity2(entity2);

        protos::PredictResponse response;
        grpc::ClientContext context;

        const grpc::Status status = m_stub->PredictRelation(&context, request, &response);

        if (status.ok())
        {
            if (confidence) *confidence = response.confidence();
            return response.relation();
        }
        LOG_ERROR(g_logger) << "RPC failed: " << status.error_code() << ": " << status.error_message();
        if (!response.error().empty())
        {
            LOG_ERROR(g_logger) << "RPC Server failed: " << response.error();
        }
        return "";
    }
}
