#pragma once

#include "../../Shaders/OptixTraceShared.h"
#include "Scene/Light.h"
#include <cuda.h>
#include "Interop/CudaTimingEvents.h"
#include <array>
#include <cstddef>
#include <vector>

class OptixSceneGAS;

// Borrows the SurfaceInput CUDA context/stream and GAS OptiX context.
// Lifetime: destroy this pass before GAS, and GAS before the CUDA owner.
class OptixTracePass
{
public:
    OptixTracePass() = default;
    ~OptixTracePass();
    OptixTracePass(const OptixTracePass &) = delete;
    OptixTracePass &operator=(const OptixTracePass &) = delete;

    void Initialize(CUcontext context, CUstream stream, const OptixSceneGAS &gas,
        unsigned int width, unsigned int height, bool enableIndirect = false, bool enableDiagnostics = false,
        const std::vector<SceneLight> &additionalLights = {});
    // Enqueues on the SAME stream as SurfaceInput production. Does not wait on the CPU.
    // Input stays alive until this stream completes. Finish prior launch before reuse.
    void Launch(CUdeviceptr surfaces, const ShadowSettings &settings);
    void SetTimingEvents(const CudaTimingEvents *timing) { mTiming=timing; }
    // Primary shadow and direct/indirect diffuse transport in one launch.
    void LaunchIndirect(CUdeviceptr surfaces, const ShadowSettings &settings, const IndirectSettings &indirect);
    CUdeviceptr GetDeviceIndirect() const { return mIndirect; }
    CUdeviceptr GetDeviceIndirectDebug() const { return mIndirectDebug; }
    unsigned int GetLightingPlaneCount() const { return mAdditionalLightCount ? AdditionalLightingPlanes : 1; }
    size_t GetIndirectBytes() const
    { return static_cast<size_t>(mWidth)*mHeight*GetLightingPlaneCount()*sizeof(IndirectLighting); }
    // GPU SurfaceRay[] -> caller-owned GPU SurfaceHit[], Width*Height records each.
    // Both buffers must be distinct/live through completion on this same stream.
    void LaunchSurfaceQueries(CUdeviceptr rays, CUdeviceptr hits);
    CUdeviceptr GetDeviceOutput() const { return mOutput; }
    size_t GetOutputBytes() const { return mOutputBytes; }

private:
    void Enqueue(const OptixTraceLaunchParams &parameters);
    CUcontext mCudaContext = nullptr;
    CUstream mStream = nullptr;
    OptixDeviceContext mOptixContext = nullptr;
    OptixModule mModule = nullptr;
    OptixProgramGroup mRaygen = nullptr, mMiss = nullptr, mHit = nullptr;
    OptixPipeline mPipeline = nullptr;
    OptixShaderBindingTable mSbt = {};
    CUdeviceptr mParams = 0, mOutput = 0;
    CUdeviceptr mIndirect = 0, mIndirectHits = 0, mIndirectDebug = 0;
    OptixTraceLaunchParams *mHostParams = nullptr;
    size_t mOutputBytes = 0;
    unsigned int mWidth = 0, mHeight = 0;
    unsigned int mAdditionalLightCount = 0;
    std::array<TraceLight,MaxAdditionalLights> mAdditionalLights = {};
    OptixTraversableHandle mGas = 0;
    CUevent mLaunchDone = nullptr;
    bool mLaunched = false;
    const CudaTimingEvents *mTiming = nullptr;
};
