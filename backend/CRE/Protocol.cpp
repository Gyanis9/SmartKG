#include "Protocol.h"
#include "base/Log.h"

#include <nlohmann/json.hpp>

namespace CRE
{
    static auto g_logger = LOG_NAME("system");

    ChatMessage::ChatPtr ChatMessage::Create(const std::string& values)
    {
        nlohmann::json json;
        try
        {
            json = nlohmann::json::parse(values);
        }
        catch (const nlohmann::json::parse_error& e)
        {
            LOG_ERROR(g_logger) << "json parse error:" << e.what();
            return nullptr;
        }
        const auto result = std::make_shared<ChatMessage>();
        for (auto& [key, value] : json.items())
        {
            result->m_datas[key] = value;
        }
        return result;
    }

    ChatMessage::ChatMessage() = default;

    std::string ChatMessage::get(const std::string& name)
    {
        const auto it = m_datas.find(name);
        return it == m_datas.end() ? "" : it->second;
    }

    std::string ChatMessage::toString() const
    {
        nlohmann::json jsons;
        for (const auto& [fst, snd] : m_datas)
        {
            jsons[fst] = snd;
        }
        return jsons.dump();
    }
}
