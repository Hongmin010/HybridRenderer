#pragma once
#include "Interop/CudaTimingEvents.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <vector>

using ProfileClock = std::chrono::steady_clock;
inline double ProfileMilliseconds(ProfileClock::time_point begin)
{ return std::chrono::duration<double,std::milli>(ProfileClock::now()-begin).count(); }

enum class FrameMetric : size_t
{
    Raster, SurfaceInput, SurfaceCopies, SurfaceKernels, OptixTrace, Accumulation, ShadowCopy, IndirectCopy,
    Composite, ToneMap, DebugView, CudaRasterWait, DxHandoffGap, GpuSpan,
    CpuSubmit, CpuEndFrame, CpuReuseWait, CpuCompletionWait, CpuSurfaceSubmit, CpuTraceSubmit, CpuPublishSubmit, CpuAccumulationSubmit, Count
};
constexpr size_t FrameMetricCount=static_cast<size_t>(FrameMetric::Count);
struct FrameTiming
{
    UINT64 Frame=0;
    UINT Width=0,Height=0,Spp=0,View=0;
    UINT AccumulationEnabled=0,HistoryFrames=0,HistoryResetMask=0;
    UINT64 HistorySamples=0;
    std::array<double,FrameMetricCount> Ms = {};
    double &At(FrameMetric metric) { return Ms[static_cast<size_t>(metric)]; }
    double At(FrameMetric metric) const { return Ms[static_cast<size_t>(metric)]; }
};
enum class DxTimingMark : UINT { RasterBegin, RasterEnd, RasterReady, CompositeBegin, CompositeEnd, ToneEnd, DebugEnd, Count };

// One frame resource set, like HybridFramePipeline. Results are read ONLY after its
// existing completion fence. No event synchronization or additional GPU wait is added.
class FrameProfiler
{
public:
    FrameProfiler() = default;
    FrameProfiler(const FrameProfiler &) = delete;
    FrameProfiler &operator=(const FrameProfiler &) = delete;
    ~FrameProfiler();
    void Initialize(ID3D12Device *device, ID3D12CommandQueue *queue, CUcontext context, UINT width, UINT height);
    bool Enabled() const { return mContext != nullptr; }
    const CudaTimingEvents *CudaEvents() const { return Enabled() ? &mCuda : nullptr; }
    void Begin(UINT64 frame, double reuseWaitMs);
    void Metadata(UINT spp, UINT view);
    void View(UINT view) { if (Enabled()) mCurrent.View=view; }
    void History(bool enabled, UINT frames, UINT64 samples, UINT reset)
    { if (Enabled()) { mCurrent.AccumulationEnabled=enabled ? 1u:0u; mCurrent.HistoryFrames=frames;
        mCurrent.HistorySamples=samples; mCurrent.HistoryResetMask=reset; } }
    void Mark(ID3D12GraphicsCommandList *list, DxTimingMark mark);
    void Resolve(ID3D12GraphicsCommandList *list);
    void Cpu(double submitMs, double endFrameMs, double completionWaitMs);
    void CpuPhase(FrameMetric metric, double ms) { if (Enabled()) mCurrent.At(metric)=ms; }
    void Collect(); // Caller already completed DX12 commands, including CUDA output waits.
    std::optional<FrameTiming> TakeCompleted();
private:
    CUcontext mContext=nullptr;
    CudaTimingEvents mCuda;
    Microsoft::WRL::ComPtr<ID3D12QueryHeap> mQueries;
    Microsoft::WRL::ComPtr<ID3D12Resource> mReadback;
    UINT64 mFrequency=0;
    UINT mWidth=0,mHeight=0;
    bool mPending=false;
    FrameTiming mCurrent;
    std::optional<FrameTiming> mLatest;
};

// Per-frame CSV plus aggregate statistics; timings are durations in separate clock domains.
class FrameTimingLog
{
public:
    explicit FrameTimingLog(const std::filesystem::path &path);
    void Add(const FrameTiming &timing);
    void Summary(std::ostream &out) const;
    size_t Size() const { return mRows.size(); }
private:
    std::ofstream mOutput;
    std::vector<FrameTiming> mRows;
};
