#include "my_module.h"

#include "CREServlet.h"
#include "base/Config.h"
#include "base/Log.h"
#include "core/Env.h"
#include "net/Application.h"
#include "net/http/HttpServer.h"

namespace Gyanis::net::http
{
    class HttpServer;
}

namespace Gyanis::net::web
{
    class TcpServer;
}

namespace CRE
{
    static auto g_logger = LOG_NAME("system");

    MyModule::MyModule(): Module("project_name", "1.0", "")
    {
    }

    bool MyModule::onLoad()
    {
        LOG_INFO(g_logger) << "onLoad";
        return true;
    }

    bool MyModule::onUnload()
    {
        LOG_INFO(g_logger) << "onUnload";
        return true;
    }

    bool MyModule::onServerReady()
    {
        LOG_INFO(g_logger) << "onServerReady";
        std::vector<std::shared_ptr<Gyanis::net::web::TcpServer>> servers;
        if (!Gyanis::net::Application::GetInstance()->getServer("http", servers))
        {
            LOG_INFO(g_logger) << "no httpserver alive";
            return false;
        }
        for (auto& i : servers)
        {
            const std::shared_ptr<Gyanis::net::http::HttpServer> http_server =
                std::dynamic_pointer_cast<Gyanis::net::http::HttpServer>(i);
            if (!i)
            {
                continue;
            }
            const auto slt_dispatch = http_server->getServletDispatch();

            auto slt = std::make_shared<CREServlet>();
            slt_dispatch->addGlobalServlet("/api/predict", slt);
        }
        return true;
    }

    bool MyModule::onServerUp()
    {
        LOG_INFO(g_logger) << "onServerUp";
        return true;
    }
}

extern "C" {
Gyanis::net::Module* CreateModule()
{
    Gyanis::net::Module* module = new CRE::MyModule;
    LOG_INFO(CRE::g_logger) << "CreateModule " << module;
    return module;
}

void DestoryModule(const Gyanis::net::Module* module)
{
    LOG_INFO(CRE::g_logger) << "CreateModule " << module;
    delete module;
}
}
