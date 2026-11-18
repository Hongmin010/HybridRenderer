#include "IndirectLighting.h"

extern "C" __global__ void AccumulateIndirect(const IndirectLighting *current,
    IndirectLighting *history, unsigned int *counts, unsigned int pixels, unsigned int reset, unsigned int planes)
{
    const unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=pixels) return;
    const bool valid = current[i].Valid==1;
    const unsigned int n = valid ? (reset ? 1 : counts[i]+1) : 0;
    for (unsigned int plane=0; plane<planes; ++plane)
    {
        const auto index = i+plane*pixels;
        if (!valid) { history[index]={}; continue; }
        const auto value = current[index];
        if (n==1) { history[index]=value; continue; }
        const float weight = 1.0f/float(n);
        const auto previous = history[index];
        history[index]={previous.R+(value.R-previous.R)*weight,previous.G+(value.G-previous.G)*weight,
            previous.B+(value.B-previous.B)*weight,1};
    }
    counts[i] = n;
}
