#pragma once
#include "../../Shaders/IndirectLighting.h"
#include <stdexcept>

template <typename Char>
unsigned int ParseIndirectBounceCount(const Char *text)
{
    if (!text || !*text) throw std::runtime_error("--bounces needs an integer from 0 to 8.");
    unsigned int count = 0;
    for (; *text; ++text)
    {
        if (*text < '0' || *text > '9')
            throw std::runtime_error("--bounces needs an integer from 0 to 8.");
        count = count*10 + static_cast<unsigned int>(*text-'0');
        if (count > MaxIndirectBounces) throw std::runtime_error("--bounces needs an integer from 0 to 8.");
    }
    return count;
}
