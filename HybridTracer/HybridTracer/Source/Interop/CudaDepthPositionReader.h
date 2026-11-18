#pragma once

#include <d3d12.h>
#include <wrl.h>
#include <cuda.h>
#include "CudaTimingEvents.h"
#include <cstddef>
#include <cstdint>
#include "../../Shaders/SurfaceInput.h"

static_assert(offsetof(SurfaceInput, NormalX) == 16);
static_assert(offsetof(SurfaceInput, AlbedoR) == 32);

// Same-thread diagnostic module; hand the shared R32_FLOAT depth RT over in COMMON.
// CPU copies are for validation. Reconstruction itself runs on CUDA device memory.
class CudaDepthPositionReader
{
public:
    CudaDepthPositionReader() = default;
    ~CudaDepthPositionReader();
    CudaDepthPositionReader(const CudaDepthPositionReader &) = delete;
    CudaDepthPositionReader &operator=(const CudaDepthPositionReader &) = delete;

    void Initialize(ID3D12Device *device, ID3D12Resource *depth,
        ID3D12Resource *normal = nullptr, ID3D12Resource *albedo = nullptr,
        bool enableReadback = true);
    // Matrix: 16 row-major floats from inverse(View * Projection), no transpose.
    // Finish the previous readback before reusing this module's pinned buffers.
    void EnqueueRead(ID3D12CommandQueue *queue, UINT64 value, const float *inverseViewProjection);
    // Production path: compute on GPU without allocating/copying diagnostic images.
    void EnqueueCompute(ID3D12CommandQueue *queue, UINT64 value, const float *inverseViewProjection, const CudaTimingEvents *timing = nullptr);
    void WaitForReadback();
    const float *GetDepthPixels() const { return mHostDepth; }
    const WorldPosition *GetPositions() const { return mHostPositions; }
    CUdeviceptr GetDevicePositions() const { return mDevicePositions; }
    const std::uint8_t *GetAlbedoPixels() const { return mHostAlbedo; }
    const SurfaceInput *GetSurfaceInputs() const { return mHostSurface; }
    CUdeviceptr GetDeviceSurfaceInputs() const { return mDeviceSurface; }
    CUcontext GetContext() const { return mContext; }
    CUstream GetStream() const { return mStream; }

private:
    void Enqueue(ID3D12CommandQueue *queue, UINT64 value, const float *matrix, bool readback, const CudaTimingEvents *timing = nullptr);
    bool mEnableReadback = true;
    Microsoft::WRL::ComPtr<ID3D12Resource> mDepth;
    Microsoft::WRL::ComPtr<ID3D12Fence> mRasterReady, mCudaDone;
    CUcontext mContext = nullptr;
    CUstream mStream = nullptr;
    CUexternalMemory mMemory = nullptr;
    CUmipmappedArray mMipmappedArray = nullptr;
    CUarray mArray = nullptr;
    CUexternalSemaphore mRasterSemaphore = nullptr, mCudaSemaphore = nullptr;
    CUmodule mModule = nullptr;
    CUfunction mKernel = nullptr;
    CUdeviceptr mDeviceDepth = 0, mDeviceMatrix = 0, mDevicePositions = 0;
    float *mHostDepth = nullptr, *mHostMatrix = nullptr;
    WorldPosition *mHostPositions = nullptr;
    UINT mWidth = 0, mHeight = 0;
    UINT64 mLastValue = 0;
    Microsoft::WRL::ComPtr<ID3D12Resource> mSurfaceTextures[2]; // normal, albedo
    CUexternalMemory mSurfaceMemory[2] = {};
    CUmipmappedArray mSurfaceMipmaps[2] = {};
    CUarray mSurfaceArrays[2] = {};
    CUdeviceptr mSurfaceSource[2] = {}; // half4 normal, uchar4 albedo
    CUfunction mSurfaceKernel = nullptr;
    CUdeviceptr mDeviceSurface = 0;
    std::uint8_t *mHostAlbedo = nullptr;
    SurfaceInput *mHostSurface = nullptr;
};
