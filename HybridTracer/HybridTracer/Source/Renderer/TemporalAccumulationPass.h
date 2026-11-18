#pragma once
#include "Interop/CudaTimingEvents.h"
#include "../../Shaders/IndirectLighting.h"
#include <array>

enum class HistoryReset : unsigned int
{
    None=0, FirstFrame=1, Camera=2, Light=4, Sampling=8, Manual=16, Toggle=32, Limit=64
};
struct AccumulationState
{
    std::array<float,32> Camera={}; // View and Projection, exact comparison; no image readback.
    std::array<float,7> Light={};   // Normalized direction, linear color, intensity.
    unsigned int Spp=0;
    unsigned int MaxIndirectBounces=DefaultIndirectBounces;
};
struct AccumulationInfo
{
    bool Enabled=true;
    unsigned int Frames=0,ResetMask=0;
    UINT64 SamplesPerPixel=0; // Maximum for pixels valid throughout this epoch.
};

// Static-camera progressive mean in linear radiance. Same CUDA stream as Trace.
// Previous frame completion is protected by HybridFramePipeline's existing fence.
class TemporalAccumulationPass
{
public:
    TemporalAccumulationPass()=default;
    ~TemporalAccumulationPass();
    TemporalAccumulationPass(const TemporalAccumulationPass &)=delete;
    TemporalAccumulationPass &operator=(const TemporalAccumulationPass &)=delete;
    static constexpr unsigned int MaxFrames=1u<<20;
    void Initialize(CUcontext context, CUstream stream, unsigned int width, unsigned int height,
        unsigned int planes = 1);
    void SetEnabled(bool enabled);
    void Reset() { mPendingReset|=static_cast<unsigned int>(HistoryReset::Manual); }
    void Execute(CUdeviceptr input, const AccumulationState &state, const CudaTimingEvents *timing=nullptr);
    const AccumulationInfo &GetInfo() const { return mInfo; }
    CUdeviceptr GetOutput() const { return mOutput; }
    CUdeviceptr GetPixelCounts() const { return mCounts; } // Diagnostic use only.
private:
    CUcontext mContext=nullptr;
    CUstream mStream=nullptr;
    CUmodule mModule=nullptr;
    CUfunction mKernel=nullptr;
    CUdeviceptr mHistory=0,mCounts=0,mOutput=0;
    unsigned int mPixels=0,mPlanes=1,mPendingReset=0;
    bool mHasState=false;
    AccumulationState mPrevious;
    AccumulationInfo mInfo;
};
