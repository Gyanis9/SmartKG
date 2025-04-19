#ifndef PROTOCOL_H
#define PROTOCOL_H
#include <memory>
#include <string>
#include <unordered_map>

namespace CRE
{
    class ChatMessage
    {
    public:
        using ChatPtr = std::shared_ptr<ChatMessage>;

        static ChatPtr Create(const std::string& values);

        ChatMessage();

        std::string get(const std::string& name);

        std::string toString() const;

    private:
        std::unordered_map<std::string, std::string> m_datas;
    };
}

#endif
