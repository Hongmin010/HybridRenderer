#include "CudaDepthPositionReader.h"
#include "InteropCommon.h"

#include <nvrtc.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

using namespace HybridInterop;

namespace
{
    void CheckNvrtc(nvrtcResult result, const char *operation)
    {
        if (result != NVRTC_SUCCESS)
            throw std::runtime_error(std::string(operation) + ": " + nvrtcGetErrorString(result));
    }

    struct RuntimeProgram
    {
        nvrtcProgram Program = nullptr;
        ~RuntimeProgram() { if (Program) nvrtcDestroyProgram(&Program); }
    };

    void CompileKernel(CUdevice device, CUmodule &module, CUfunction &kernel)
    {
        wchar_t executable[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, executable, MAX_PATH);
        const auto path = std::filesystem::path(executable).parent_path() /
            L"Shaders" / L"ReconstructWorldPosition.cu";
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot open Shaders/ReconstructWorldPosition.cu.");
        const std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        int major = 0, minor = 0;
        CheckCuda(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute(major)");
        CheckCuda(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute(minor)");
        const std::string architecture = "--gpu-architecture=compute_" + std::to_string(major*10 + minor);
        const std::string includePath = std::string("--include-path=") + HYBRIDTRACER_CUDA_INCLUDE_DIR;
        const char *options[] = {architecture.c_str(), "--std=c++14", "--fmad=false", includePath.c_str()};
        RuntimeProgram program;
        CheckNvrtc(nvrtcCreateProgram(&program.Program, source.c_str(), "ReconstructWorldPosition.cu",
            0, nullptr, nullptr), "nvrtcCreateProgram");
        const auto result = nvrtcCompileProgram(program.Program, 4, options);
        if (result != NVRTC_SUCCESS)
        {
            size_t bytes = 0;
            CheckNvrtc(nvrtcGetProgramLogSize(program.Program, &bytes), "nvrtcGetProgramLogSize");
            std::vector<char> log(bytes);
            CheckNvrtc(nvrtcGetProgramLog(program.Program, log.data()), "nvrtcGetProgramLog");
            throw std::runtime_error(std::string("NVRTC kernel compile failed: ") + log.data());
        }
        size_t bytes = 0;
        CheckNvrtc(nvrtcGetPTXSize(program.Program, &bytes), "nvrtcGetPTXSize");
        std::vector<char> ptx(bytes);
        CheckNvrtc(nvrtcGetPTX(program.Program, ptx.data()), "nvrtcGetPTX");
        CheckCuda(cuModuleLoadData(&module, ptx.data()), "cuModuleLoadData(position)");
        CheckCuda(cuModuleGetFunction(&kernel, module, "ReconstructWorldPosition"), "cuModuleGetFunction(position)");
        std::cout << "Position CUDA kernel compiled with NVRTC for compute_" << major << minor << '\n';
    }
}

CudaDepthPositionReader::~CudaDepthPositionReader()
{
    if (mContext) cuCtxSetCurrent(mContext);
    if (mStream) cuStreamSynchronize(mStream);
    if (mHostPositions) cuMemFreeHost(mHostPositions);
    if (mHostDepth) cuMemFreeHost(mHostDepth);
    if (mHostMatrix) cuMemFreeHost(mHostMatrix);
    if (mHostAlbedo) cuMemFreeHost(mHostAlbedo);
    if (mHostSurface) cuMemFreeHost(mHostSurface);
    if (mDeviceSurface) cuMemFree(mDeviceSurface);
    for (size_t i = 0; i < 2; ++i)
    {
        if (mSurfaceSource[i]) cuMemFree(mSurfaceSource[i]);
        if (mSurfaceMipmaps[i]) cuMipmappedArrayDestroy(mSurfaceMipmaps[i]);
        if (mSurfaceMemory[i]) cuDestroyExternalMemory(mSurfaceMemory[i]);
    }
    if (mDevicePositions) cuMemFree(mDevicePositions);
    if (mDeviceDepth) cuMemFree(mDeviceDepth);
    if (mDeviceMatrix) cuMemFree(mDeviceMatrix);
    if (mModule) cuModuleUnload(mModule);
    if (mMipmappedArray) cuMipmappedArrayDestroy(mMipmappedArray);
    if (mMemory) cuDestroyExternalMemory(mMemory);
    if (mRasterSemaphore) cuDestroyExternalSemaphore(mRasterSemaphore);
    if (mCudaSemaphore) cuDestroyExternalSemaphore(mCudaSemaphore);
    if (mStream) cuStreamDestroy(mStream);
    if (mContext) cuCtxDestroy(mContext);
}

void CudaDepthPositionReader::Initialize(ID3D12Device *device, ID3D12Resource *depth,
    ID3D12Resource *normal, ID3D12Resource *albedo, bool enableReadback)
{
    if (mDepth) throw std::runtime_error("CudaDepthPositionReader is already initialized.");
    const auto desc = depth->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.Format != DXGI_FORMAT_R32_FLOAT ||
        desc.MipLevels != 1 || desc.DepthOrArraySize != 1 || desc.SampleDesc.Count != 1)
        throw std::runtime_error("Position reader requires a single-mip R32_FLOAT 2D render target.");
    if ((normal == nullptr) != (albedo == nullptr))
        throw std::runtime_error("Pass both Normal and Albedo for surface inputs.");
    const CUdevice cudaDevice = FindCudaDevice(device);
    mDepth = depth;
    mEnableReadback = enableReadback;
    mWidth = static_cast<UINT>(desc.Width);
    mHeight = desc.Height;
    const size_t depthBytes = static_cast<size_t>(mWidth) * mHeight * sizeof(float);
    const size_t positionBytes = static_cast<size_t>(mWidth) * mHeight * sizeof(WorldPosition);
    CheckCuda(cuCtxCreate(&mContext, CU_CTX_SCHED_AUTO, cudaDevice), "cuCtxCreate(position)");
    CheckCuda(cuStreamCreate(&mStream, CU_STREAM_NON_BLOCKING), "cuStreamCreate(position)");
    CheckCuda(cuMemAlloc(&mDeviceDepth, depthBytes), "cuMemAlloc(depth)");
    CheckCuda(cuMemAlloc(&mDevicePositions, positionBytes), "cuMemAlloc(positions)");
    CheckCuda(cuMemAlloc(&mDeviceMatrix, 16*sizeof(float)), "cuMemAlloc(matrix)");
    if (mEnableReadback)
    {
        CheckCuda(cuMemAllocHost(reinterpret_cast<void **>(&mHostDepth), depthBytes), "cuMemAllocHost(depth)");
        CheckCuda(cuMemAllocHost(reinterpret_cast<void **>(&mHostPositions), positionBytes), "cuMemAllocHost(positions)");
    }
    CheckCuda(cuMemAllocHost(reinterpret_cast<void **>(&mHostMatrix), 16*sizeof(float)), "cuMemAllocHost(matrix)");
    CompileKernel(cudaDevice, mModule, mKernel);

    WinHandle handle;
    CheckHR(device->CreateSharedHandle(depth, nullptr, GENERIC_ALL, nullptr, &handle.Value), "CreateSharedHandle(depth)");
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memory = {};
    memory.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    memory.handle.win32.handle = handle.Value;
    memory.size = device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;
    memory.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
    CheckCuda(cuImportExternalMemory(&mMemory, &memory), "cuImportExternalMemory(depth)");
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mapping = {};
    mapping.arrayDesc.Width = mWidth;
    mapping.arrayDesc.Height = mHeight;
    mapping.arrayDesc.Format = CU_AD_FORMAT_FLOAT;
    mapping.arrayDesc.NumChannels = 1;
    mapping.arrayDesc.Flags = CUDA_ARRAY3D_COLOR_ATTACHMENT;
    mapping.numLevels = 1;
    CheckCuda(cuExternalMemoryGetMappedMipmappedArray(&mMipmappedArray, mMemory, &mapping), "cuExternalMemoryGetMappedMipmappedArray(depth)");
    CheckCuda(cuMipmappedArrayGetLevel(&mArray, mMipmappedArray, 0), "cuMipmappedArrayGetLevel(depth)");
    CheckHR(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&mRasterReady)), "CreateFence(depth ready)");
    CheckHR(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&mCudaDone)), "CreateFence(position done)");
    mRasterSemaphore = ImportFence(device, mRasterReady.Get());
    mCudaSemaphore = ImportFence(device, mCudaDone.Get());

    if (normal && albedo)
    {
        const size_t pixels = static_cast<size_t>(mWidth)*mHeight;
        ID3D12Resource *textures[] = {normal, albedo};
        const DXGI_FORMAT formats[] = {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM};
        const size_t bytesPerPixel[] = {8, 4};
        for (size_t i = 0; i < 2; ++i)
        {
            const auto textureDesc = textures[i]->GetDesc();
            if (textureDesc.Width != mWidth || textureDesc.Height != mHeight ||
                textureDesc.Format != formats[i] || textureDesc.MipLevels != 1 ||
                textureDesc.DepthOrArraySize != 1 || textureDesc.SampleDesc.Count != 1 ||
                textureDesc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D)
                throw std::runtime_error("Surface texture dimensions/format do not match Depth.");
            mSurfaceTextures[i] = textures[i];
            WinHandle textureHandle;
            CheckHR(device->CreateSharedHandle(textures[i], nullptr, GENERIC_ALL, nullptr,
                &textureHandle.Value), "CreateSharedHandle(surface input)");
            CUDA_EXTERNAL_MEMORY_HANDLE_DESC external = {};
            external.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
            external.handle.win32.handle = textureHandle.Value;
            external.size = device->GetResourceAllocationInfo(0, 1, &textureDesc).SizeInBytes;
            external.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
            CheckCuda(cuImportExternalMemory(&mSurfaceMemory[i], &external), "cuImportExternalMemory(surface input)");
            CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC textureMapping = {};
            textureMapping.arrayDesc.Width = mWidth;
            textureMapping.arrayDesc.Height = mHeight;
            textureMapping.arrayDesc.Format = i == 0 ? CU_AD_FORMAT_HALF : CU_AD_FORMAT_UNSIGNED_INT8;
            textureMapping.arrayDesc.NumChannels = 4;
            textureMapping.arrayDesc.Flags = CUDA_ARRAY3D_COLOR_ATTACHMENT;
            textureMapping.numLevels = 1;
            CheckCuda(cuExternalMemoryGetMappedMipmappedArray(&mSurfaceMipmaps[i], mSurfaceMemory[i],
                &textureMapping), "cuExternalMemoryGetMappedMipmappedArray(surface input)");
            CheckCuda(cuMipmappedArrayGetLevel(&mSurfaceArrays[i], mSurfaceMipmaps[i], 0), "cuMipmappedArrayGetLevel(surface input)");
            CheckCuda(cuMemAlloc(&mSurfaceSource[i], pixels*bytesPerPixel[i]), "cuMemAlloc(surface source)");
        }
        CheckCuda(cuMemAlloc(&mDeviceSurface, pixels*sizeof(SurfaceInput)), "cuMemAlloc(surface inputs)");
        if (mEnableReadback)
        {
            CheckCuda(cuMemAllocHost(reinterpret_cast<void **>(&mHostSurface), pixels*sizeof(SurfaceInput)), "cuMemAllocHost(surface inputs)");
            CheckCuda(cuMemAllocHost(reinterpret_cast<void **>(&mHostAlbedo), pixels*4), "cuMemAllocHost(albedo)");
        }
        CheckCuda(cuModuleGetFunction(&mSurfaceKernel, mModule, "BuildSurfaceInputs"), "cuModuleGetFunction(surface inputs)");
    }
}

void CudaDepthPositionReader::EnqueueRead(ID3D12CommandQueue *queue, UINT64 value, const float *matrix)
{
    if (!mEnableReadback) throw std::runtime_error("Readback was disabled for this surface interop owner.");
    Enqueue(queue,value,matrix,true);
}

void CudaDepthPositionReader::EnqueueCompute(ID3D12CommandQueue *queue, UINT64 value, const float *matrix, const CudaTimingEvents *timing)
{
    Enqueue(queue,value,matrix,false,timing);
}

void CudaDepthPositionReader::Enqueue(ID3D12CommandQueue *queue, UINT64 value, const float *matrix, bool readback, const CudaTimingEvents *timing)
{
    if (!mCudaSemaphore || value <= mLastValue)
        throw std::runtime_error("Initialize the position reader and use increasing fence values.");
    CheckCuda(cuCtxSetCurrent(mContext), "cuCtxSetCurrent(position)");
    std::memcpy(mHostMatrix, matrix, 16*sizeof(float));
    CheckHR(queue->Signal(mRasterReady.Get(), value), "Signal(depth ready)");
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait = {};
    wait.params.fence.value = value;
    RecordCudaTiming(timing,CudaTimingMark::RasterWaitBegin,mStream);
    CheckCuda(cuWaitExternalSemaphoresAsync(&mRasterSemaphore, &wait, 1, mStream), "cuWaitExternalSemaphoresAsync(depth)");
    RecordCudaTiming(timing,CudaTimingMark::SurfaceBegin,mStream);

    CUDA_MEMCPY2D copy = {};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = mArray;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = mDeviceDepth;
    copy.dstPitch = copy.WidthInBytes = static_cast<size_t>(mWidth)*sizeof(float);
    copy.Height = mHeight;
    CheckCuda(cuMemcpy2DAsync(&copy, mStream), "cuMemcpy2DAsync(depth GPU copy)");
    CheckCuda(cuMemcpyHtoDAsync(mDeviceMatrix, mHostMatrix, 16*sizeof(float), mStream), "cuMemcpyHtoDAsync(matrix)");
    RecordCudaTiming(timing,CudaTimingMark::InputsCopied,mStream);
    void *arguments[] = {&mDeviceDepth, &mDeviceMatrix, &mDevicePositions, &mWidth, &mHeight};
    CheckCuda(cuLaunchKernel(mKernel, (mWidth+15)/16, (mHeight+15)/16, 1, 16, 16, 1,
        0, mStream, arguments, nullptr), "cuLaunchKernel(ReconstructWorldPosition)");
    RecordCudaTiming(timing,CudaTimingMark::PositionEnd,mStream);
    const size_t pixels = static_cast<size_t>(mWidth)*mHeight;
    if (mSurfaceKernel)
    {
        const size_t bytesPerPixel[] = {8, 4};
        for (size_t i = 0; i < 2; ++i)
        {
            CUDA_MEMCPY2D textureCopy = {};
            textureCopy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
            textureCopy.srcArray = mSurfaceArrays[i];
            textureCopy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            textureCopy.dstDevice = mSurfaceSource[i];
            textureCopy.dstPitch = textureCopy.WidthInBytes = static_cast<size_t>(mWidth)*bytesPerPixel[i];
            textureCopy.Height = mHeight;
            CheckCuda(cuMemcpy2DAsync(&textureCopy, mStream), "cuMemcpy2DAsync(surface source)");
        }
        RecordCudaTiming(timing,CudaTimingMark::AttributesCopied,mStream);
        void *surfaceArguments[] = {&mDevicePositions, &mSurfaceSource[0], &mSurfaceSource[1],
            &mDeviceSurface, &mWidth, &mHeight};
        CheckCuda(cuLaunchKernel(mSurfaceKernel, (mWidth+15)/16, (mHeight+15)/16, 1, 16, 16, 1,
            0, mStream, surfaceArguments, nullptr), "cuLaunchKernel(BuildSurfaceInputs)");
        RecordCudaTiming(timing,CudaTimingMark::SurfaceEnd,mStream);
        if (readback)
        {
            CheckCuda(cuMemcpyDtoHAsync(mHostAlbedo, mSurfaceSource[1], pixels*4, mStream), "cuMemcpyDtoHAsync(albedo)");
            CheckCuda(cuMemcpyDtoHAsync(mHostSurface, mDeviceSurface, pixels*sizeof(SurfaceInput), mStream), "cuMemcpyDtoHAsync(surface inputs)");
        }
    }
    if (readback)
    {
        CheckCuda(cuMemcpyDtoHAsync(mHostDepth, mDeviceDepth, pixels*sizeof(float), mStream), "cuMemcpyDtoHAsync(depth validation)");
        CheckCuda(cuMemcpyDtoHAsync(mHostPositions, mDevicePositions, pixels*sizeof(WorldPosition), mStream), "cuMemcpyDtoHAsync(position validation)");
    }
    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal = {};
    signal.params.fence.value = value;
    CheckCuda(cuSignalExternalSemaphoresAsync(&mCudaSemaphore, &signal, 1, mStream), "cuSignalExternalSemaphoresAsync(position)");
    CheckHR(queue->Wait(mCudaDone.Get(), value), "Wait(position done)");
    mLastValue = value;
}

void CudaDepthPositionReader::WaitForReadback()
{
    CheckCuda(cuCtxSetCurrent(mContext), "cuCtxSetCurrent(position readback)");
    CheckCuda(cuStreamSynchronize(mStream), "cuStreamSynchronize(position readback)");
}
