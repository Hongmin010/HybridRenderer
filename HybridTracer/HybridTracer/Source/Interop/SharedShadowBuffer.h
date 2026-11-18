#pragma once
#include <cuda.h>
#include "CudaTimingEvents.h"
#include <d3d12.h>
#include <wrl.h>

// COMMON at both API boundaries. CUDA is the sole data writer, DX12 the sole reader.
// Caller submits raster before Publish and returns the buffer to COMMON after composition.
class SharedShadowBuffer
{
public:
    ~SharedShadowBuffer();
    SharedShadowBuffer() = default;
    SharedShadowBuffer(const SharedShadowBuffer &) = delete;
    SharedShadowBuffer &operator=(const SharedShadowBuffer &) = delete;
    void Initialize(ID3D12Device *device, CUcontext context, CUstream stream, size_t bytes);
    void Publish(ID3D12CommandQueue *queue, CUdeviceptr source, UINT64 value, const CudaTimingEvents *timing = nullptr, bool indirect = false);
    ID3D12Resource *GetResource() const { return mBuffer.Get(); }
private:
    Microsoft::WRL::ComPtr<ID3D12Resource> mBuffer;
    Microsoft::WRL::ComPtr<ID3D12Fence> mReady;
    CUcontext mContext = nullptr;
    CUstream mStream = nullptr;
    CUexternalMemory mMemory = nullptr;
    CUexternalSemaphore mSemaphore = nullptr;
    CUdeviceptr mMapped = 0;
    size_t mBytes = 0;
    UINT64 mLastValue = 0;
};
