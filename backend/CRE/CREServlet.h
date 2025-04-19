#ifndef CRESERVLET_H
#define CRESERVLET_H
#include "net/http/servlets/Servlet.h"
#include "RelationClient.h"
#include "db/Redis.h"
#include "db/Mysql.h"

namespace CRE
{
    class CREServlet final : public Gyanis::net::http::Servlet
    {
    public:
        CREServlet();

        int32_t handle(const std::shared_ptr<Gyanis::net::http::HttpRequest>& request,
                       const std::shared_ptr<Gyanis::net::http::HttpResponse>& response,
                       const std::shared_ptr<Gyanis::net::http::HttpSession>& session) override;

    private:
        static std::string sanitize(const std::string& input);

        static std::string generateRedisKey(const std::string& sentence,
                                            const std::string& entity1,
                                            const std::string& entity2);
        void storeToDatabase(const std::string& sentence,
                             const std::string& entity1,
                             const std::string& entity2,
                             const std::string& relation,
                             float confidence) const;

    private:
        std::unique_ptr<RelationClient> m_client;
        std::unique_ptr<Gyanis::db::RedisClient> m_redis;
        std::shared_ptr<Gyanis::db::MySQL> m_mysql;
    };
}

#endif
