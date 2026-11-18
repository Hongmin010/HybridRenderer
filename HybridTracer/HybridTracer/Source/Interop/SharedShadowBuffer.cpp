#include "SharedShadowBuffer.h"
#include "InteropCommon.h"
using namespace HybridInterop;

SharedShadowBuffer::~SharedShadowBuffer()
{
    if (mContext) cuCtxSetCurrent(mContext);
    if (mStream) cuStreamSynchronize(mStream);
    if (mMapped) cuMemFree(mMapped);
    if (mMemory) cuDestroyExternalMemory(mMemory);
    if (mSemaphore) cuDestroyExternalSemaphore(mSemaphore);
}

void SharedShadowBuffer::Initialize(ID3D12Device *device, CUcontext context, CUstream stream, size_t bytes)
{
    if (mBuffer || !context || !stream || !bytes) throw std::runtime_error("Invalid shared shadow initialization.");
    mContext = context;
    mStream = stream;
    mBytes = bytes;
    CheckCuda(cuCtxSetCurrent(context), "cuCtxSetCurrent(shared shadow)");
    CUcontext streamContext = nullptr;
    CheckCuda(cuStreamGetCtx(stream,&streamContext), "cuStreamGetCtx(shared shadow)");
    if (streamContext != context) throw std::runtime_error("Shared shadow must use the SurfaceInput/OptiX stream.");
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc = {};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    CheckHR(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_SHARED,&desc,
        D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&mBuffer)), "CreateCommittedResource(shared shadow)");
    WinHandle handle;
    CheckHR(device->CreateSharedHandle(mBuffer.Get(),nullptr,GENERIC_ALL,nullptr,&handle.Value), "CreateSharedHandle(shadow)");
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC external = {};
    external.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    external.handle.win32.handle = handle.Value;
    external.size = device->GetResourceAllocationInfo(0,1,&desc).SizeInBytes;
    external.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    CheckCuda(cuImportExternalMemory(&mMemory,&external), "cuImportExternalMemory(shadow)");
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC mapping = {};
    mapping.size = bytes;
    CheckCuda(cuExternalMemoryGetMappedBuffer(&mMapped,mMemory,&mapping), "cuExternalMemoryGetMappedBuffer(shadow)");
    CheckHR(device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&mReady)), "CreateFence(shadow ready)");
    mSemaphore = ImportFence(device,mReady.Get());
}

void SharedShadowBuffer::Publish(ID3D12CommandQueue *queue, CUdeviceptr source, UINT64 value, const CudaTimingEvents *timing, bool indirect)
{
    if (!mMapped || !source || value <= mLastValue) throw std::runtime_error("Shared shadow needs increasing fence values.");
    CheckCuda(cuCtxSetCurrent(mContext), "cuCtxSetCurrent(publish shadow)");
    RecordCudaTiming(timing,indirect ? CudaTimingMark::IndirectCopyBegin : CudaTimingMark::ShadowCopyBegin,mStream);
    CheckCuda(cuMemcpyDtoDAsync(mMapped,source,mBytes,mStream), "cuMemcpyDtoDAsync(shared shadow)");
    RecordCudaTiming(timing,indirect ? CudaTimingMark::IndirectCopyEnd : CudaTimingMark::ShadowCopyEnd,mStream);
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal = {};
    signal.params.fence.value = value;
    CheckCuda(cuSignalExternalSemaphoresAsync(&mSemaphore,&signal,1,mStream), "cuSignalExternalSemaphoresAsync(shadow ready)");
    // This fence is AFTER optixLaunch and the output copy, unlike SurfaceInput's done fence.
    CheckHR(queue->Wait(mReady.Get(),value), "Wait(shadow ready)");
    mLastValue = value;
}
