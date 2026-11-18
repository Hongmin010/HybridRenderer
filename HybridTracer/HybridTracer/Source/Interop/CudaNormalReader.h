#pragma once

#include <d3d12.h>
#include <wrl.h>
#include <cuda.h>

#include <cstdint>

// Diagnostic reader for a shared R16G16B16A16_FLOAT normal render target.
// Create/use/destroy on the same CPU thread. Submit the raster work first and
// hand the texture over in COMMON; EnqueueRead adds a GPU wait before DX12 reuse.
class CudaNormalReader
{
public:
    CudaNormalReader() = default;
    ~CudaNormalReader();
    CudaNormalReader(const CudaNormalReader &) = delete;
    CudaNormalReader &operator=(const CudaNormalReader &) = delete;

    void Initialize(ID3D12Device *device, ID3D12Resource *normal);
    void EnqueueRead(ID3D12CommandQueue *queue, UINT64 value);
    void WaitForReadback();

    // Four original half-float bit patterns per pixel, rows top to bottom.
    // Valid after WaitForReadback(), until the next EnqueueRead().
    const std::uint16_t *GetPixels() const { return mPixels; }

private:
    Microsoft::WRL::ComPtr<ID3D12Resource> mNormal;
    Microsoft::WRL::ComPtr<ID3D12Fence> mRasterReady;
    Microsoft::WRL::ComPtr<ID3D12Fence> mCudaReadDone;
    CUcontext mContext = nullptr;
    CUstream mStream = nullptr;
    CUexternalMemory mMemory = nullptr;
    CUmipmappedArray mMipmappedArray = nullptr;
    CUarray mArray = nullptr; // Owned by mMipmappedArray; do not destroy separately.
    CUexternalSemaphore mRasterSemaphore = nullptr;
    CUexternalSemaphore mCudaSemaphore = nullptr;
    std::uint16_t *mPixels = nullptr;
    size_t mRowBytes = 0;
    size_t mHeight = 0;
    UINT64 mLastValue = 0;
};
