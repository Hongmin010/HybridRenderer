#include "FrameProfiler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
using namespace HybridInterop;
namespace
{
    constexpr UINT DxCount=static_cast<UINT>(DxTimingMark::Count);
    const char *Names[] = {"raster_ms","surface_input_ms","surface_copies_ms","surface_kernels_ms",
        "optix_trace_ms","accumulation_ms","shadow_copy_ms","indirect_copy_ms","composite_ms","tonemap_ms","debug_view_ms",
        "cuda_raster_wait_ms","dx_handoff_gap_ms","gpu_span_ms","cpu_submit_ms","cpu_end_frame_ms",
        "cpu_reuse_wait_ms","cpu_completion_wait_ms","cpu_surface_submit_ms","cpu_trace_submit_ms","cpu_publish_submit_ms","cpu_accumulation_submit_ms"};
    static_assert(std::size(Names)==FrameMetricCount);
}
FrameProfiler::~FrameProfiler()
{
    if (mContext) cuCtxSetCurrent(mContext);
    // HybridFramePipeline drains its queue before members (including this one) die.
    for (auto event:mCuda.Events) if (event) cuEventDestroy(event);
}
void FrameProfiler::Initialize(ID3D12Device *device, ID3D12CommandQueue *queue, CUcontext context, UINT width, UINT height)
{
    if (Enabled() || !context) throw std::runtime_error("Frame profiler needs a fresh object and live CUDA context.");
    mContext=context; mWidth=width; mHeight=height;
    CheckHR(queue->GetTimestampFrequency(&mFrequency),"GetTimestampFrequency(frame profiler)");
    if (!mFrequency) throw std::runtime_error("DX12 timestamp frequency is zero.");
    D3D12_QUERY_HEAP_DESC queries={}; queries.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP; queries.Count=DxCount;
    CheckHR(device->CreateQueryHeap(&queries,IID_PPV_ARGS(&mQueries)),"CreateQueryHeap(frame timestamps)");
    D3D12_HEAP_PROPERTIES heap={}; heap.Type=D3D12_HEAP_TYPE_READBACK; heap.CreationNodeMask=heap.VisibleNodeMask=1;
    D3D12_RESOURCE_DESC buffer={}; buffer.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; buffer.Width=DxCount*sizeof(UINT64);
    buffer.Height=buffer.DepthOrArraySize=buffer.MipLevels=1; buffer.SampleDesc.Count=1; buffer.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    CheckHR(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,IID_PPV_ARGS(&mReadback)),"CreateCommittedResource(timestamp readback)");
    CheckCuda(cuCtxSetCurrent(context),"cuCtxSetCurrent(frame profiler)");
    for (auto &event:mCuda.Events) CheckCuda(cuEventCreate(&event,CU_EVENT_DEFAULT),"cuEventCreate(frame timing)");
}
void FrameProfiler::Begin(UINT64 frame,double reuseWaitMs)
{
    if (!Enabled()) return;
    if (mPending) throw std::runtime_error("Collect completed timing before reusing timestamp resources.");
    mCurrent={}; mCurrent.Frame=frame; mCurrent.Width=mWidth; mCurrent.Height=mHeight;
    mCurrent.At(FrameMetric::CpuReuseWait)=reuseWaitMs;
}
void FrameProfiler::Metadata(UINT spp,UINT view) { if (Enabled()) { mCurrent.Spp=spp; mCurrent.View=view; } }
void FrameProfiler::Mark(ID3D12GraphicsCommandList *list,DxTimingMark mark)
{
    if (Enabled()) list->EndQuery(mQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,static_cast<UINT>(mark));
}
void FrameProfiler::Resolve(ID3D12GraphicsCommandList *list)
{
    if (!Enabled()) return;
    list->ResolveQueryData(mQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,DxCount,mReadback.Get(),0);
    mPending=true;
}
void FrameProfiler::Cpu(double submitMs,double endFrameMs,double completionWaitMs)
{
    if (!Enabled()) return;
    mCurrent.At(FrameMetric::CpuSubmit)=submitMs;
    mCurrent.At(FrameMetric::CpuEndFrame)=endFrameMs;
    mCurrent.At(FrameMetric::CpuCompletionWait)=completionWaitMs;
}
void FrameProfiler::Collect()
{
    if (!Enabled() || !mPending) return;
    CheckCuda(cuCtxSetCurrent(mContext),"cuCtxSetCurrent(collect timing)");
    CheckCuda(cuEventQuery(mCuda.Events.back()),"Frame timing must be complete before collection");
    std::array<UINT64,DxCount> ticks={};
    const D3D12_RANGE read={0,sizeof(ticks)},none={0,0}; void *mapped=nullptr;
    CheckHR(mReadback->Map(0,&read,&mapped),"Map(timestamp readback)");
    std::memcpy(ticks.data(),mapped,sizeof(ticks)); mReadback->Unmap(0,&none);
    for (UINT i=1;i<DxCount;++i) if (ticks[i]<ticks[i-1]) throw std::runtime_error("Nonmonotonic DX12 timestamps.");
    const auto dx=[&](DxTimingMark a,DxTimingMark b)
    { return double(ticks[static_cast<UINT>(b)]-ticks[static_cast<UINT>(a)])*1000.0/double(mFrequency); };
    const auto cuda=[&](CudaTimingMark a,CudaTimingMark b)
    {
        float ms=0; CheckCuda(cuEventElapsedTime(&ms,mCuda.Events[static_cast<size_t>(a)],mCuda.Events[static_cast<size_t>(b)]),"cuEventElapsedTime(frame)");
        return double(ms);
    };
    auto &r=mCurrent;
    r.At(FrameMetric::Raster)=dx(DxTimingMark::RasterBegin,DxTimingMark::RasterEnd);
    r.At(FrameMetric::Composite)=dx(DxTimingMark::CompositeBegin,DxTimingMark::CompositeEnd);
    r.At(FrameMetric::ToneMap)=dx(DxTimingMark::CompositeEnd,DxTimingMark::ToneEnd);
    r.At(FrameMetric::DebugView)=dx(DxTimingMark::ToneEnd,DxTimingMark::DebugEnd);
    r.At(FrameMetric::DxHandoffGap)=dx(DxTimingMark::RasterReady,DxTimingMark::CompositeBegin);
    r.At(FrameMetric::GpuSpan)=dx(DxTimingMark::RasterBegin,DxTimingMark::DebugEnd);
    r.At(FrameMetric::CudaRasterWait)=cuda(CudaTimingMark::RasterWaitBegin,CudaTimingMark::SurfaceBegin);
    r.At(FrameMetric::SurfaceInput)=cuda(CudaTimingMark::SurfaceBegin,CudaTimingMark::SurfaceEnd);
    r.At(FrameMetric::SurfaceCopies)=cuda(CudaTimingMark::SurfaceBegin,CudaTimingMark::InputsCopied)+
        cuda(CudaTimingMark::PositionEnd,CudaTimingMark::AttributesCopied);
    r.At(FrameMetric::SurfaceKernels)=cuda(CudaTimingMark::InputsCopied,CudaTimingMark::PositionEnd)+
        cuda(CudaTimingMark::AttributesCopied,CudaTimingMark::SurfaceEnd);
    r.At(FrameMetric::OptixTrace)=cuda(CudaTimingMark::TraceBegin,CudaTimingMark::TraceEnd);
    r.At(FrameMetric::Accumulation)=cuda(CudaTimingMark::AccumulationBegin,CudaTimingMark::AccumulationEnd);
    r.At(FrameMetric::ShadowCopy)=cuda(CudaTimingMark::ShadowCopyBegin,CudaTimingMark::ShadowCopyEnd);
    r.At(FrameMetric::IndirectCopy)=cuda(CudaTimingMark::IndirectCopyBegin,CudaTimingMark::IndirectCopyEnd);
    for (const double ms:r.Ms) if (!std::isfinite(ms) || ms<0) throw std::runtime_error("Invalid frame timing.");
    mLatest=r; mPending=false;
}
std::optional<FrameTiming> FrameProfiler::TakeCompleted() { auto result=mLatest; mLatest.reset(); return result; }

FrameTimingLog::FrameTimingLog(const std::filesystem::path &path):mOutput(path)
{
    if (!mOutput) throw std::runtime_error("Cannot open frame timing CSV: "+path.u8string());
    mOutput<<"frame,width,height,spp,view,accumulation_enabled,history_frames,history_samples,history_reset_mask";
    for (const auto name:Names) mOutput<<','<<name;
    mOutput<<'\n'<<std::setprecision(9);
}
void FrameTimingLog::Add(const FrameTiming &timing)
{
    mOutput<<timing.Frame<<','<<timing.Width<<','<<timing.Height<<','<<timing.Spp<<','<<timing.View<<','<<timing.AccumulationEnabled<<','<<timing.HistoryFrames<<','<<timing.HistorySamples<<','<<timing.HistoryResetMask;
    for (const double ms:timing.Ms) { if (!std::isfinite(ms) || ms<0) throw std::runtime_error("Invalid timing CSV value."); mOutput<<','<<ms; }
    mOutput<<'\n';
    if (!mOutput) throw std::runtime_error("Cannot write frame timing CSV.");
    mRows.push_back(timing);
}
void FrameTimingLog::Summary(std::ostream &out) const
{
    if (mRows.empty()) return;
    out<<"TIMING SUMMARY: "<<mRows.size()<<" frames; "<<mRows.front().Width<<'x'<<mRows.front().Height
        <<", "<<mRows.front().Spp<<" spp; mean / median / p95 (ms)\n"<<std::fixed<<std::setprecision(4);
    for (size_t i=0;i<FrameMetricCount;++i)
    {
        std::vector<double> values; values.reserve(mRows.size());
        for (const auto &row:mRows) values.push_back(row.Ms[i]);
        const double mean=std::accumulate(values.begin(),values.end(),0.0)/values.size();
        std::sort(values.begin(),values.end());
        const double median=values.size()%2 ? values[values.size()/2] : (values[values.size()/2-1]+values[values.size()/2])*0.5;
        const size_t p95=static_cast<size_t>(std::ceil(0.95*values.size()))-1;
        out<<Names[i]<<": "<<mean<<" / "<<median<<" / "<<values[p95]<<'\n';
    }
    out<<"NOTE: dx_handoff_gap includes CUDA work and host submission gaps; do not add it to CUDA durations.\n"
       <<"NOTE: GPU span excludes Present/vsync, query resolution and CPU completion wait.\n";
}
