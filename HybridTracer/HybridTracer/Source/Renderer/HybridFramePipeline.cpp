#include "HybridFramePipeline.h"
#include "Interop/InteropCommon.h"
#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include <cmath>
#include <algorithm>
using namespace HybridInterop;
using namespace DirectX;

HybridFramePipeline::~HybridFramePipeline()
{
    try { WaitForIdle(); }
    catch (const std::exception &error) { std::cerr << "Hybrid shutdown wait: " << error.what() << '\n'; }
    if (mCompletedEvent) CloseHandle(mCompletedEvent);
}

void HybridFramePipeline::Initialize(ID3D12Device *device, ID3D12CommandQueue *queue, DXGI_FORMAT format,
    UINT width, UINT height, UINT frameCount, const Scene &scene, bool enableProfiling, bool enableAccumulation)
{
    mQueue = queue;
    CheckHR(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&mCompleted)),"CreateFence(hybrid frame)");
    mCompletedEvent = CreateEventW(nullptr,FALSE,FALSE,nullptr);
    if (!mCompletedEvent) throw std::runtime_error("CreateEventW(hybrid frame) failed.");
    mRaster.Initialize(device,format,width,height,frameCount,true,true,true);
    mSurface.Initialize(device,GetDepth(),GetNormal(),GetAlbedo());
    mGas.Initialize(mSurface.GetContext(),mSurface.GetStream());
    mGas.Build(scene);
    mTrace.Initialize(mSurface.GetContext(),mSurface.GetStream(),mGas,width,height,true,false,scene.GetAdditionalLights());
    mAccumulation.Initialize(mSurface.GetContext(),mSurface.GetStream(),width,height,mTrace.GetLightingPlaneCount());
    mAccumulation.SetEnabled(enableAccumulation);
    mShadow.Initialize(device,mSurface.GetContext(),mSurface.GetStream(),mTrace.GetOutputBytes());
    mIndirect.Initialize(device,mSurface.GetContext(),mSurface.GetStream(),mTrace.GetIndirectBytes());
    mComposite.Initialize(device,GetAlbedo(),GetNormal(),GetDepth(),mShadow.GetResource(),mIndirect.GetResource(),
        mTrace.GetLightingPlaneCount());
    mToneMap.Initialize(device,mComposite.GetOutput(),format);
    mDebug.Initialize(device,format,GetAlbedo(),GetNormal(),GetDepth());
    if (enableProfiling) mProfiler.Initialize(device,queue,mSurface.GetContext(),width,height);
    mTrace.SetTimingEvents(mProfiler.CudaEvents());
}

void HybridFramePipeline::WaitForIdle()
{
    if (!mQueue || !mCompleted || !mCompletedEvent) return;
    const UINT64 value = ++mIdleValue;
    CheckHR(mQueue->Signal(mCompleted.Get(),value),"Signal(hybrid frame complete)");
    CheckHR(mCompleted->SetEventOnCompletion(value,mCompletedEvent),"SetEventOnCompletion(hybrid frame)");
    if (WaitForSingleObject(mCompletedEvent,10000) != WAIT_OBJECT_0)
        throw std::runtime_error("Hybrid frame completion timed out after 10 seconds.");
    ComPtr<ID3D12Device> device;
    CheckHR(mQueue->GetDevice(IID_PPV_ARGS(&device)),"GetDevice(hybrid queue)");
    CheckHR(device->GetDeviceRemovedReason(),"GetDeviceRemovedReason(hybrid frame)");
}

void HybridFramePipeline::RecordRaster(ID3D12GraphicsCommandList *list, D3D12_CPU_DESCRIPTOR_HANDLE output,
    UINT frameIndex, const Camera &camera, const Scene &scene)
{
    // Protect the single G-buffer, packet, output, pinned matrix and launch parameters.
    // This is frame-resource reuse, not CPU transport of rendering data.
    const auto waitBegin=mProfiler.Enabled() ? ProfileClock::now() : ProfileClock::time_point{};
    WaitForIdle();
    const double waitMs=mProfiler.Enabled() ? ProfileMilliseconds(waitBegin) : 0;
    mProfiler.Collect();
    mProfiler.Begin(mFrameValue+1,waitMs);
    mProfiler.Mark(list,DxTimingMark::RasterBegin);
    mRaster.Render(list,output,frameIndex,camera,scene);
    mProfiler.Mark(list,DxTimingMark::RasterEnd);
    list->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    for (auto *resource : {GetAlbedo(),GetNormal(),GetDepth()})
        Transition(list,resource,D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_COMMON);
    mProfiler.Mark(list,DxTimingMark::RasterReady);
}

void HybridFramePipeline::EnqueueTrace(const Camera &camera, const DirectionalLight &light,
    unsigned int samplesPerPixel, unsigned int seed, unsigned int maxIndirectBounces)
{
    if (maxIndirectBounces > MaxIndirectBounces)
        throw std::runtime_error("Indirect bounce limit exceeds the supported maximum.");
    const float length = std::sqrt(light.Direction.x*light.Direction.x+light.Direction.y*light.Direction.y+light.Direction.z*light.Direction.z);
    if (!std::isfinite(length) || length <= 0 || !std::isfinite(light.Intensity) || light.Intensity < 0)
        throw std::runtime_error("Invalid directional light.");
    mFrameLight = light;
    mFrameLight.Direction = {light.Direction.x/length,light.Direction.y/length,light.Direction.z/length};
    XMFLOAT4X4 inverseVP;
    XMStoreFloat4x4(&inverseVP,XMMatrixInverse(nullptr,camera.GetView()*camera.GetProjection()));
    const UINT64 value = ++mFrameValue;
    mProfiler.Metadata(samplesPerPixel,0);
    const auto surfaceBegin=mProfiler.Enabled() ? ProfileClock::now() : ProfileClock::time_point{};
    mSurface.Execute(mQueue.Get(),value,&inverseVP._11,mProfiler.CudaEvents());
    if (mProfiler.Enabled()) mProfiler.CpuPhase(FrameMetric::CpuSurfaceSubmit,ProfileMilliseconds(surfaceBegin));
    const auto &direction = mFrameLight.Direction;
    mShadowSettings = {direction.x,direction.y,direction.z,0.001f,0.0001f,1e16f};
    const auto &color = light.Color;
    if (!std::isfinite(color.x) || !std::isfinite(color.y) || !std::isfinite(color.z) || color.x < 0 || color.y < 0 || color.z < 0)
        throw std::runtime_error("Directional light color must be finite and nonnegative.");
    XMFLOAT4X4 inverseView;
    XMStoreFloat4x4(&inverseView,XMMatrixInverse(nullptr,camera.GetView()));
    // seed=0 selects the monotonically changing frame number; explicit seeds support diagnostics.
    const IndirectSettings indirect = {inverseView._41,inverseView._42,inverseView._43,
        seed ? seed : static_cast<unsigned int>(value),color.x*light.Intensity,color.y*light.Intensity,
        color.z*light.Intensity,samplesPerPixel,maxIndirectBounces};
    const auto traceBegin=mProfiler.Enabled() ? ProfileClock::now() : ProfileClock::time_point{};
    mTrace.LaunchIndirect(mSurface.GetOutput(),mShadowSettings,indirect);
    if (mProfiler.Enabled()) mProfiler.CpuPhase(FrameMetric::CpuTraceSubmit,ProfileMilliseconds(traceBegin));
    AccumulationState state;
    XMFLOAT4X4 viewMatrix,projectionMatrix;
    XMStoreFloat4x4(&viewMatrix,camera.GetView()); XMStoreFloat4x4(&projectionMatrix,camera.GetProjection());
    std::copy_n(&viewMatrix._11,16,state.Camera.begin());
    std::copy_n(&projectionMatrix._11,16,state.Camera.begin()+16);
    state.Light={direction.x,direction.y,direction.z,color.x,color.y,color.z,light.Intensity};
    state.Spp=samplesPerPixel;
    state.MaxIndirectBounces=maxIndirectBounces;
    const auto accumulationBegin=mProfiler.Enabled() ? ProfileClock::now() : ProfileClock::time_point{};
    mAccumulation.Execute(mTrace.GetDeviceIndirect(),state,mProfiler.CudaEvents());
    if (mProfiler.Enabled()) mProfiler.CpuPhase(FrameMetric::CpuAccumulationSubmit,ProfileMilliseconds(accumulationBegin));
    const auto &history=mAccumulation.GetInfo();
    mProfiler.History(history.Enabled,history.Frames,history.SamplesPerPixel,history.ResetMask);
    const auto publishBegin=mProfiler.Enabled() ? ProfileClock::now() : ProfileClock::time_point{};
    mShadow.Publish(mQueue.Get(),mTrace.GetDeviceOutput(),value,mProfiler.CudaEvents());
    mIndirect.Publish(mQueue.Get(),mAccumulation.GetOutput(),value,mProfiler.CudaEvents(),true);
    if (mProfiler.Enabled()) mProfiler.CpuPhase(FrameMetric::CpuPublishSubmit,ProfileMilliseconds(publishBegin));
}

void HybridFramePipeline::RecordComposite(ID3D12GraphicsCommandList *list, D3D12_CPU_DESCRIPTOR_HANDLE output,
    HybridView view, float exposure)
{
    for (auto *resource : {GetAlbedo(),GetNormal(),GetDepth()})
        Transition(list,resource,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_RENDER_TARGET);
    mProfiler.View(static_cast<UINT>(view));
    mProfiler.Mark(list,DxTimingMark::CompositeBegin);
    mComposite.Render(list,mFrameLight,view != HybridView::Unshadowed,view == HybridView::Shadow,view == HybridView::Direct ? 1u : view == HybridView::Indirect ? 2u : 0u);
    mProfiler.Mark(list,DxTimingMark::CompositeEnd);
    mToneMap.Render(list,output,exposure,view == HybridView::Shadow);
    mProfiler.Mark(list,DxTimingMark::ToneEnd);
    if (view >= HybridView::Albedo && view <= HybridView::Depth)
        mDebug.Render(list,output,static_cast<UINT>(view));
    mProfiler.Mark(list,DxTimingMark::DebugEnd);
    mProfiler.Resolve(list);
}
