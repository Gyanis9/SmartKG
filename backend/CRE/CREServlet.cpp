#include "CREServlet.h"
#include "Protocol.h"
#include "base/Config.h"

namespace CRE
{
    static auto g_logger = LOG_NAME("system");
    static auto server_address = Gyanis::base::Config::LookUp<std::string>(
        "grpc.server_address", "localhost:50051", "C++ grpc server_address");

    static auto mysql_config = Gyanis::base::Config::LookUp<std::unordered_map<std::string, std::string>>(
        "mysql.config", std::unordered_map<std::string, std::string>(), "mysql config");

    CREServlet::CREServlet(): Servlet("CREServlet")
    {
        m_client = std::make_unique<RelationClient>(CreateChannel(server_address->getValue(),
                                                                  grpc::InsecureChannelCredentials()));
        m_redis = std::make_unique<Gyanis::db::RedisClient>();
        m_mysql = std::make_shared<Gyanis::db::MySQL>(mysql_config->getValue());
        if (!m_mysql->connect())
        {
            LOG_ERROR(LOG_ROOT()) << "Failed to connect to MySQL database";
            throw std::runtime_error("Failed to connect to MySQL database");
        }
    }

    int32_t CREServlet::handle(const std::shared_ptr<Gyanis::net::http::HttpRequest>& request,
                               const std::shared_ptr<Gyanis::net::http::HttpResponse>& response,
                               const std::shared_ptr<Gyanis::net::http::HttpSession>& session)
    {
        try
        {
            // 解析请求数据
            const auto data = ChatMessage::Create(request->getBody());
            const auto sentence = data->get("sentence");
            const auto entity1 = data->get("entity1");
            const auto entity2 = data->get("entity2");

            // 参数校验
            if (sentence.empty() || entity1.empty() || entity2.empty())
            {
                response->setStatus(Gyanis::net::http::HttpStatus::BAD_REQUEST);
                response->setBody("Missing required parameters");
                return 0;
            }
            // 生成Redis键
            std::string redis_key = generateRedisKey(sentence, entity1, entity2);
            // 先查询Redis缓存
            std::string cached_result;
            float confidence = 0.0f;
            std::string relation;
            if (auto catch_result = m_redis->get(redis_key))
            {
                auto cache_json = nlohmann::json::parse(catch_result.value());
                relation = cache_json["relation"].get<std::string>();
                confidence = cache_json["confidence"].get<float>();
            }
            else
            {
                // 缓存未命中，调用模型预测
                confidence = 0.0f;
                relation = m_client->PredictRelation(sentence, entity1, entity2, &confidence);
                if (relation.empty())
                {
                    response->setStatus(Gyanis::net::http::HttpStatus::NOT_FOUND);
                    response->setBody("No relation found");
                    return 0;
                }

                nlohmann::json cache_data;
                cache_data["relation"] = relation;
                cache_data["confidence"] = confidence;
                if (!m_redis->set(redis_key, cache_data.dump()))
                {
                    LOG_ERROR(g_logger) << "Failed to set redis configuration";
                }
                storeToDatabase(sentence, entity1, entity2, relation, confidence);
            }
            nlohmann::json response_json;
            response_json["relation"] = relation;
            response_json["confidence"] = confidence;
            response->setBody(response_json.dump());
            response->setHeader("Content-Type", "application/json");
        }
        catch (const std::exception& e)
        {
            // 异常处理
            response->setStatus(Gyanis::net::http::HttpStatus::INTERNAL_SERVER_ERROR);
            response->setBody("Internal Server Error: " + std::string(e.what()));
        }

        return 0;
    }

    std::string CREServlet::sanitize(const std::string& input)
    {
        std::string output = input;
        std::replace(output.begin(), output.end(), ' ', '_');
        return output;
    }

    std::string CREServlet::generateRedisKey(const std::string& sentence, const std::string& entity1,
                                             const std::string& entity2)
    {
        return "rel:" + sanitize(sentence) + ":"
            + sanitize(entity1) + ":" + sanitize(entity2);
    }

    void CREServlet::storeToDatabase(const std::string& sentence, const std::string& entity1,
                                     const std::string& entity2, const std::string& relation,
                                     const float confidence) const
    {
        const auto stmt = Gyanis::db::MySQLStmt::Create(m_mysql, "INSERT INTO relation_records "
                                                        "(sentence, entity1, entity2, relation, confidence, create_time) "
                                                        "VALUES (?, ?, ?, ?, ?, NOW())");
        stmt->bindString(1, sentence);
        stmt->bindString(2, entity1);
        stmt->bindString(3, entity2);
        stmt->bindString(4, relation);
        stmt->bindFloat(5, confidence);
        stmt->execute();
    }
}
