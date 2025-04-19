#ifndef MY_MODULE_H
#define MY_MODULE_H

#include "net/Module.h"

namespace CRE
{
    class MyModule final : public Gyanis::net::Module
    {
    public:
        using ptr = std::shared_ptr<MyModule>;

        MyModule();

        bool onLoad() override;

        bool onUnload() override;

        bool onServerReady() override;

        bool onServerUp() override;
    };
}

#endif
