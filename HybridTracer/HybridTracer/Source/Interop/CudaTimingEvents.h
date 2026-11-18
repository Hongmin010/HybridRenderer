#pragma once
#include "InteropCommon.h"
#include <array>

enum class CudaTimingMark : size_t
{
    RasterWaitBegin, SurfaceBegin, InputsCopied, PositionEnd, AttributesCopied, SurfaceEnd,
    TraceBegin, TraceEnd, AccumulationBegin, AccumulationEnd, ShadowCopyBegin, ShadowCopyEnd, IndirectCopyBegin, IndirectCopyEnd, Count
};
struct CudaTimingEvents
{
    std::array<CUevent,static_cast<size_t>(CudaTimingMark::Count)> Events = {};
};
// A null hook leaves the existing rendering/validation path uninstrumented.
inline void RecordCudaTiming(const CudaTimingEvents *timing, CudaTimingMark mark, CUstream stream)
{
    if (timing) HybridInterop::CheckCuda(cuEventRecord(timing->Events[static_cast<size_t>(mark)],stream),"cuEventRecord(frame timing)");
}
