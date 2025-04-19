#ifndef RELATIONCLIENT_H
#define RELATIONCLIENT_H
#include <grpcpp/grpcpp.h>
#include "relation.pb.h"
#include "relation.grpc.pb.h"

namespace CRE
{
    class RelationClient
    {
    public:
        explicit RelationClient(const std::shared_ptr<grpc::Channel>& channel);

        std::string PredictRelation(const std::string& sentence,
                                    const std::string& entity1,
                                    const std::string& entity2,
                                    float* confidence) const;

    private:
        std::unique_ptr<protos::RelationService::Stub> m_stub;
    };
}

#endif //RELATIONCLIENT_H
