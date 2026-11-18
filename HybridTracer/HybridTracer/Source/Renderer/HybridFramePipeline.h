#pragma once
#include "RasterGBufferPass.h"
#include "SurfaceInputPass.h"
#include "Optix/OptixSceneGAS.h"
#include "Optix/OptixTracePass.h"
#include "Interop/SharedShadowBuffer.h"
#include "CompositePass.h"
#include "ToneMapPass.h"
#include "DebugViewPass.h"
#include "FrameProfiler.h"
#include "TemporalAccumulationPass.h"

enum class HybridView : UINT { Lit = 0, Albedo = 1, Normal = 2, Depth = 3, Shadow = 4, Unshadowed = 5, Direct = 6, Indirect = 7 };

// The same frame orchestration is used by the windowed app and the headless verification.
// Single in-flight frame for this first integration; GPU handoffs use external fences.
class HybridFramePipeline
{
public:
    ~HybridFramePipeline();
    void Initialize(ID3D12Device *device, ID3D12CommandQueue *queue, DXGI_FORMAT outputFormat,
        UINT width, UINT height, UINT frameCount, const Scene &scene, bool enableProfiling = false, bool enableAccumulation = true);
    void RecordRaster(ID3D12GraphicsCommandList *list, D3D12_CPU_DESCRIPTOR_HANDLE output,
        UINT frameIndex, const Camera &camera, const Scene &scene);
    // Call AFTER submitting raster commands. Does not perform a CPU image readback/wait.
    void EnqueueTrace(const Camera &camera, const DirectionalLight &light, unsigned int samplesPerPixel = 4,
        unsigned int seed = 0, unsigned int maxIndirectBounces = DefaultIndirectBounces);
    // Submit this list AFTER EnqueueTrace inserted DX12 waits for both output copies.
    void RecordComposite(ID3D12GraphicsCommandList *list, D3D12_CPU_DESCRIPTOR_HANDLE output,
        HybridView view = HybridView::Lit, float exposure = 1.0f);
    void WaitForIdle();
    // Collect only after an existing GPU completion wait (RecordRaster does this automatically).
    void CollectTimings() { mProfiler.Collect(); }
    std::optional<FrameTiming> TakeCompletedTiming() { return mProfiler.TakeCompleted(); }
    void SetCpuTiming(double submitMs, double endFrameMs, double completionWaitMs = 0)
    { mProfiler.Cpu(submitMs,endFrameMs,completionWaitMs); }
    void SetAccumulationEnabled(bool enabled) { mAccumulation.SetEnabled(enabled); }
    void ResetAccumulation() { mAccumulation.Reset(); }
    const AccumulationInfo &GetAccumulationInfo() const { return mAccumulation.GetInfo(); }
    CUdeviceptr GetHistoryPixelCounts() const { return mAccumulation.GetPixelCounts(); }
    void ValidateGAS(const Scene &scene) { mGas.ValidateAgainstScene(scene); }

    // Read-only resource access for diagnostics. No CPU copies are made by the pipeline.
    ID3D12Resource *GetAlbedo() const { return mRaster.GetAlbedoBuffer(); }
    ID3D12Resource *GetNormal() const { return mRaster.GetNormalBuffer(); }
    ID3D12Resource *GetDepth() const { return mRaster.GetRasterDepthBuffer(); }
    ID3D12Resource *GetHDR() const { return mComposite.GetOutput(); }
    ID3D12Resource *GetSharedShadow() const { return mShadow.GetResource(); }
    ID3D12Resource *GetSharedIndirect() const { return mIndirect.GetResource(); }
    CUdeviceptr GetDeviceIndirect() const { return mAccumulation.GetOutput(); }
    CUdeviceptr GetDeviceRawIndirect() const { return mTrace.GetDeviceIndirect(); }
    CUdeviceptr GetDeviceShadow() const { return mTrace.GetDeviceOutput(); }
    CUdeviceptr GetDeviceSurfaces() const { return mSurface.GetOutput(); }
    CUcontext GetCudaContext() const { return mSurface.GetContext(); }
    const ShadowSettings &GetShadowSettings() const { return mShadowSettings; }
    const DirectionalLight &GetFrameLight() const { return mFrameLight; }
private:
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> mQueue;
    Microsoft::WRL::ComPtr<ID3D12Fence> mCompleted;
    HANDLE mCompletedEvent = nullptr;
    UINT64 mIdleValue = 0, mFrameValue = 0;
    // Destruction runs bottom-up: DX12 views -> mapped output -> trace -> GAS -> CUDA owner.
    RasterGBufferPass mRaster;
    SurfaceInputPass mSurface;
    OptixSceneGAS mGas;
    OptixTracePass mTrace;
    TemporalAccumulationPass mAccumulation;
    SharedShadowBuffer mShadow, mIndirect;
    CompositePass mComposite;
    ToneMapPass mToneMap;
    DebugViewPass mDebug;
    DirectionalLight mFrameLight;
    ShadowSettings mShadowSettings = {};
    FrameProfiler mProfiler;
};
