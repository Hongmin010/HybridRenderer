#include "TemporalAccumulationPass.h"
#include <nvrtc.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <vector>
#include <cmath>
using namespace HybridInterop;
namespace
{
    void CheckNvrtc(nvrtcResult result,const char *operation)
    { if (result!=NVRTC_SUCCESS) throw std::runtime_error(std::string(operation)+": "+nvrtcGetErrorString(result)); }
    std::string Read(const std::filesystem::path &path)
    {
        std::ifstream input(path,std::ios::binary);
        if (!input) throw std::runtime_error("Cannot read temporal shader: "+path.u8string());
        return {(std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>()};
    }
    std::vector<char> Compile()
    {
        wchar_t executable[MAX_PATH]={}; GetModuleFileNameW(nullptr,executable,MAX_PATH);
        const auto folder=std::filesystem::path(executable).parent_path()/L"Shaders";
        const auto source=Read(folder/L"TemporalAccumulation.cu");
        const std::string data=Read(folder/L"IndirectLighting.h"),hit=Read(folder/L"SurfaceHit.h");
        const char *headers[]={data.c_str(),hit.c_str()},*names[]={"IndirectLighting.h","SurfaceHit.h"};
        struct Program { nvrtcProgram Handle=nullptr; ~Program() { if (Handle) nvrtcDestroyProgram(&Handle); } } program;
        CheckNvrtc(nvrtcCreateProgram(&program.Handle,source.c_str(),"TemporalAccumulation.cu",2,headers,names),"nvrtcCreateProgram(temporal)");
        CUdevice device; CheckCuda(cuCtxGetDevice(&device),"cuCtxGetDevice(temporal)");
        int major=0,minor=0;
        CheckCuda(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device),"cuDeviceGetAttribute(major)");
        CheckCuda(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device),"cuDeviceGetAttribute(minor)");
        const std::string arch="--gpu-architecture=compute_"+std::to_string(major*10+minor);
        const char *options[]={arch.c_str(),"--std=c++14","--fmad=false"};
        const auto result=nvrtcCompileProgram(program.Handle,3,options);
        if (result!=NVRTC_SUCCESS)
        {
            size_t size=0; CheckNvrtc(nvrtcGetProgramLogSize(program.Handle,&size),"nvrtcGetProgramLogSize(temporal)");
            std::vector<char> log(size); CheckNvrtc(nvrtcGetProgramLog(program.Handle,log.data()),"nvrtcGetProgramLog(temporal)");
            throw std::runtime_error(std::string("Temporal NVRTC compile failed: ")+log.data());
        }
        size_t size=0; CheckNvrtc(nvrtcGetPTXSize(program.Handle,&size),"nvrtcGetPTXSize(temporal)");
        std::vector<char> ptx(size); CheckNvrtc(nvrtcGetPTX(program.Handle,ptx.data()),"nvrtcGetPTX(temporal)");
        return ptx;
    }
}
TemporalAccumulationPass::~TemporalAccumulationPass()
{
    if (mContext) cuCtxSetCurrent(mContext);
    if (mStream) cuStreamSynchronize(mStream);
    if (mHistory) cuMemFree(mHistory);
    if (mCounts) cuMemFree(mCounts);
    if (mModule) cuModuleUnload(mModule);
}
void TemporalAccumulationPass::Initialize(CUcontext context,CUstream stream,unsigned int width,unsigned int height,
    unsigned int planes)
{
    if (mContext || !context || !stream || !width || !height || (planes!=1 && planes!=AdditionalLightingPlanes) ||
        size_t(width)*height>std::numeric_limits<unsigned int>::max()/planes)
        throw std::runtime_error("Temporal accumulation needs fresh ownership, live CUDA context/stream and supported dimensions.");
    mContext=context; mStream=stream; mPixels=width*height; mPlanes=planes;
    CheckCuda(cuCtxSetCurrent(context),"cuCtxSetCurrent(temporal)");
    CUcontext streamContext=nullptr; CheckCuda(cuStreamGetCtx(stream,&streamContext),"cuStreamGetCtx(temporal)");
    if (streamContext!=context) throw std::runtime_error("Temporal accumulation must share the trace context/stream.");
    const auto ptx=Compile();
    CheckCuda(cuModuleLoadData(&mModule,ptx.data()),"cuModuleLoadData(temporal)");
    CheckCuda(cuModuleGetFunction(&mKernel,mModule,"AccumulateIndirect"),"cuModuleGetFunction(temporal)");
    CheckCuda(cuMemAlloc(&mHistory,size_t(mPixels)*mPlanes*sizeof(IndirectLighting)),"cuMemAlloc(lighting history)");
    CheckCuda(cuMemAlloc(&mCounts,size_t(mPixels)*sizeof(unsigned int)),"cuMemAlloc(history counts)");
    // First launch writes every pixel without reading either uninitialized allocation.
}
void TemporalAccumulationPass::SetEnabled(bool enabled)
{
    if (mInfo.Enabled!=enabled) { mInfo.Enabled=enabled; mPendingReset|=static_cast<unsigned int>(HistoryReset::Toggle); }
}
void TemporalAccumulationPass::Execute(CUdeviceptr input,const AccumulationState &state,const CudaTimingEvents *timing)
{
    if (!mKernel || !input || input==mHistory || !state.Spp || state.Spp>4096)
        throw std::runtime_error("Temporal accumulation needs distinct live input and 1..4096 spp.");
    for (float value:state.Camera) if (!std::isfinite(value)) throw std::runtime_error("Invalid temporal camera state.");
    for (float value:state.Light) if (!std::isfinite(value)) throw std::runtime_error("Invalid temporal light state.");
    CheckCuda(cuCtxSetCurrent(mContext),"cuCtxSetCurrent(accumulate)");
    unsigned int reason=mPendingReset;
    if (!mHasState) reason|=static_cast<unsigned int>(HistoryReset::FirstFrame);
    else
    {
        if (state.Camera!=mPrevious.Camera) reason|=static_cast<unsigned int>(HistoryReset::Camera);
        if (state.Light!=mPrevious.Light) reason|=static_cast<unsigned int>(HistoryReset::Light);
        if (state.Spp!=mPrevious.Spp || state.MaxIndirectBounces!=mPrevious.MaxIndirectBounces)
            reason|=static_cast<unsigned int>(HistoryReset::Sampling);
    }
    if (mInfo.Frames==MaxFrames) reason|=static_cast<unsigned int>(HistoryReset::Limit);
    RecordCudaTiming(timing,CudaTimingMark::AccumulationBegin,mStream);
    if (mInfo.Enabled)
    {
        unsigned int reset=(reason!=0 || mInfo.Frames==0) ? 1u:0u;
        void *arguments[]={&input,&mHistory,&mCounts,&mPixels,&reset,&mPlanes};
        CheckCuda(cuLaunchKernel(mKernel,(mPixels-1)/256+1,1,1,256,1,1,0,mStream,arguments,nullptr),"cuLaunchKernel(AccumulateIndirect)");
        mInfo.Frames=reset ? 1u:mInfo.Frames+1;
        mInfo.SamplesPerPixel=UINT64(mInfo.Frames)*state.Spp;
        mOutput=mHistory;
    }
    else { mInfo.Frames=0; mInfo.SamplesPerPixel=0; mOutput=input; }
    RecordCudaTiming(timing,CudaTimingMark::AccumulationEnd,mStream);
    mInfo.ResetMask=reason; mPendingReset=0; mPrevious=state; mHasState=true;
}
