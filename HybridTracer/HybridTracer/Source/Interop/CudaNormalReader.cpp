#include "CudaNormalReader.h"
#include "InteropCommon.h"

using namespace HybridInterop;

CudaNormalReader::~CudaNormalReader()
{
    if (mContext) cuCtxSetCurrent(mContext);
    if (mStream) cuStreamSynchronize(mStream);
    if (mPixels) cuMemFreeHost(mPixels);
    if (mMipmappedArray) cuMipmappedArrayDestroy(mMipmappedArray);
    if (mMemory) cuDestroyExternalMemory(mMemory);
    if (mRasterSemaphore) cuDestroyExternalSemaphore(mRasterSemaphore);
    if (mCudaSemaphore) cuDestroyExternalSemaphore(mCudaSemaphore);
    if (mStream) cuStreamDestroy(mStream);
    if (mContext) cuCtxDestroy(mContext);
}

void CudaNormalReader::Initialize(ID3D12Device *device, ID3D12Resource *normal)
{
    if (mNormal) throw std::runtime_error("CudaNormalReader is already initialized.");
    const auto description = normal->GetDesc();
    if (description.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        description.Format != DXGI_FORMAT_R16G16B16A16_FLOAT ||
        description.DepthOrArraySize != 1 || description.MipLevels != 1 ||
        description.SampleDesc.Count != 1)
        throw std::runtime_error("CudaNormalReader requires a single-mip RGBA16F 2D texture.");

    const CUdevice cudaDevice = FindCudaDevice(device);
    mNormal = normal; // Keep the DX12 resource alive until CUDA mappings are released.
    mRowBytes = static_cast<size_t>(description.Width) * 4 * sizeof(std::uint16_t);
    mHeight = description.Height;
    CheckCuda(cuCtxCreate(&mContext, CU_CTX_SCHED_AUTO, cudaDevice), "cuCtxCreate");
    CheckCuda(cuStreamCreate(&mStream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    CheckCuda(cuMemAllocHost(reinterpret_cast<void **>(&mPixels), mRowBytes * mHeight),
        "cuMemAllocHost(normal)");

    WinHandle handle;
    CheckHR(device->CreateSharedHandle(normal, nullptr, GENERIC_ALL, nullptr, &handle.Value),
        "CreateSharedHandle(normal)");
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memory = {};
    memory.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    memory.handle.win32.handle = handle.Value;
    memory.size = device->GetResourceAllocationInfo(0, 1, &description).SizeInBytes;
    memory.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    CheckCuda(cuImportExternalMemory(&mMemory, &memory), "cuImportExternalMemory(normal)");

    // DX12's tiled texture is mapped as an array, never as a linear pointer.
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mapping = {};
    mapping.arrayDesc.Width = static_cast<size_t>(description.Width);
    mapping.arrayDesc.Height = description.Height;
    mapping.arrayDesc.Depth = 0; // CUDA's convention for an ordinary 2D array.
    mapping.arrayDesc.Format = CU_AD_FORMAT_HALF;
    mapping.arrayDesc.NumChannels = 4;
    mapping.arrayDesc.Flags = CUDA_ARRAY3D_COLOR_ATTACHMENT;
    mapping.numLevels = 1;
    CheckCuda(cuExternalMemoryGetMappedMipmappedArray(&mMipmappedArray, mMemory, &mapping),
        "cuExternalMemoryGetMappedMipmappedArray(normal)");
    CheckCuda(cuMipmappedArrayGetLevel(&mArray, mMipmappedArray, 0), "cuMipmappedArrayGetLevel");

    CheckHR(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&mRasterReady)),
        "CreateFence(rasterReady)");
    CheckHR(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&mCudaReadDone)),
        "CreateFence(cudaReadDone)");
    mRasterSemaphore = ImportFence(device, mRasterReady.Get());
    mCudaSemaphore = ImportFence(device, mCudaReadDone.Get());
}

void CudaNormalReader::EnqueueRead(ID3D12CommandQueue *queue, UINT64 value)
{
    if (!mCudaSemaphore || value <= mLastValue)
        throw std::runtime_error("Initialize the reader and use strictly increasing fence values.");

    CheckCuda(cuCtxSetCurrent(mContext), "cuCtxSetCurrent(normal)");

    // This signal follows the caller's raster commands and RT -> COMMON barrier.
    CheckHR(queue->Signal(mRasterReady.Get(), value), "Signal(rasterReady)");
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait = {};
    wait.params.fence.value = value;
    CheckCuda(cuWaitExternalSemaphoresAsync(&mRasterSemaphore, &wait, 1, mStream),
        "cuWaitExternalSemaphoresAsync(normal)");

    CUDA_MEMCPY2D copy = {};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = mArray;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = mPixels;
    copy.dstPitch = mRowBytes;
    copy.WidthInBytes = mRowBytes;
    copy.Height = mHeight;
    CheckCuda(cuMemcpy2DAsync(&copy, mStream), "cuMemcpy2DAsync(normal)");

    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal = {};
    signal.params.fence.value = value;
    CheckCuda(cuSignalExternalSemaphoresAsync(&mCudaSemaphore, &signal, 1, mStream),
        "cuSignalExternalSemaphoresAsync(normal)");
    CheckHR(queue->Wait(mCudaReadDone.Get(), value), "Wait(cudaReadDone)");
    mLastValue = value;
}

void CudaNormalReader::WaitForReadback()
{
    CheckCuda(cuCtxSetCurrent(mContext), "cuCtxSetCurrent(normal readback)");
    CheckCuda(cuStreamSynchronize(mStream), "cuStreamSynchronize(normal readback)");
}
