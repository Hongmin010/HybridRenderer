#pragma once

#include <cuda.h>
#include <optix.h>

#include <atomic>
#include <cstdint>
#include <ostream>
#include <vector>
#include "../../Shaders/OptixSceneData.h"

class Scene;

struct GasGeometry
{
    unsigned int ObjectIndex = 0;
    unsigned int VertexCount = 0;
    unsigned int TriangleCount = 0;
    CUdeviceptr Vertices = 0;
    CUdeviceptr Indices = 0;
    CUdeviceptr Normals = 0;
    float AlbedoR = 0, AlbedoG = 0, AlbedoB = 0;
};

// Static world-space triangle GAS. Borrows the SurfaceInput owner's CUDA context/stream.
// Destroy trace passes before this object, and this object before the CUDA owner.
// No IAS, update or compaction. Ray launches belong to OptixTracePass.
class OptixSceneGAS
{
public:
    OptixSceneGAS() = default;
    ~OptixSceneGAS();
    OptixSceneGAS(const OptixSceneGAS &) = delete;
    OptixSceneGAS &operator=(const OptixSceneGAS &) = delete;

    void Initialize(CUcontext cudaContext, CUstream stream);
    void Build(const Scene &scene);
    void ValidateAgainstScene(const Scene &scene, std::ostream *report = nullptr);

    OptixTraversableHandle GetHandle() const { return mHandle; }
    OptixDeviceContext GetContext() const { return mOptixContext; }
    CUcontext GetCudaContext() const { return mCudaContext; }
    const std::vector<GasGeometry> &GetGeometry() const { return mGeometry; }
    size_t GetOutputBytes() const { return mOutputBytes; }

private:
    static void LogCallback(unsigned int level, const char *tag, const char *message, void *data);

    CUcontext mCudaContext = nullptr;
    CUstream mStream = nullptr;
    OptixDeviceContext mOptixContext = nullptr;
    OptixTraversableHandle mHandle = 0;
    std::vector<GasGeometry> mGeometry;
    CUdeviceptr mScratch = 0;
    CUdeviceptr mOutput = 0;
    CUdeviceptr mEmittedBounds = 0;
    size_t mOutputBytes = 0;
    std::atomic_uint mValidationIssues{0};
    std::atomic_uint mAllocationNotices{0};
};
