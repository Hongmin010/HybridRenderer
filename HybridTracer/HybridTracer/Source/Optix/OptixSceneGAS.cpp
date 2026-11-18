#include "OptixSceneGAS.h"
#include "Interop/InteropCommon.h"
#include "Scene/Scene.h"

// Exactly one translation unit defines the SDK function table.
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace HybridInterop;
using namespace DirectX;

namespace
{
    constexpr double GeometryTolerance = 0.00001;

    void CheckOptix(OptixResult result, const char *operation)
    {
        if (result != OPTIX_SUCCESS)
            throw std::runtime_error(std::string(operation) + ": " +
                optixGetErrorName(result) + " - " + optixGetErrorString(result));
    }

    std::array<double, 6> EmptyBounds()
    {
        const double inf = std::numeric_limits<double>::infinity();
        return {inf, inf, inf, -inf, -inf, -inf};
    }
    void Include(std::array<double, 6> &bounds, const std::array<double, 3> &point)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            bounds[c] = std::min(bounds[c], point[c]);
            bounds[c+3] = std::max(bounds[c+3], point[c]);
        }
    }
    void WriteBounds(std::ostream &out, const std::array<double, 6> &bounds)
    {
        for (const auto value : bounds) out << ',' << value;
    }
}

void OptixSceneGAS::LogCallback(unsigned int level, const char *tag, const char *message, void *data)
{
    auto *self = static_cast<OptixSceneGAS *>(data);
    if (level <= 3)
    {
        // NVIDIA identifies exactly this R555+ launcher allocation notice as benign:
        // https://forums.developer.nvidia.com/t/301834/2
        // Keep it visible/countable; every other warning/error still fails validation.
        std::string_view text(message);
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.remove_suffix(1);
        if (level == 3 && std::string_view(tag) == "MEMORY" &&
            text == "cblCmdListBufferSize insufficient, attempting allocation")
            ++self->mAllocationNotices;
        else
            ++self->mValidationIssues;
        std::cerr << "OptiX [" << level << "][" << tag << "] " << message << '\n';
    }
}

OptixSceneGAS::~OptixSceneGAS()
{
    if (mCudaContext) cuCtxSetCurrent(mCudaContext);
    if (mStream) cuStreamSynchronize(mStream);
    if (mScratch) cuMemFree(mScratch);
    if (mOutput) cuMemFree(mOutput);
    if (mEmittedBounds) cuMemFree(mEmittedBounds);
    for (const auto &geometry : mGeometry)
    {
        if (geometry.Vertices) cuMemFree(geometry.Vertices);
        if (geometry.Indices) cuMemFree(geometry.Indices);
        if (geometry.Normals) cuMemFree(geometry.Normals);
    }
    if (mOptixContext) optixDeviceContextDestroy(mOptixContext);
}

void OptixSceneGAS::Initialize(CUcontext cudaContext, CUstream stream)
{
    if (mOptixContext || !cudaContext || !stream)
        throw std::runtime_error("OptixSceneGAS needs an uninitialized object and an existing CUDA context/stream.");
    mCudaContext = cudaContext;
    mStream = stream;
    CheckCuda(cuCtxSetCurrent(mCudaContext), "cuCtxSetCurrent(OptiX)");
    CUcontext streamContext = nullptr;
    CheckCuda(cuStreamGetCtx(mStream, &streamContext), "cuStreamGetCtx(OptiX)");
    if (streamContext != mCudaContext)
        throw std::runtime_error("OptiX stream must belong to the SurfaceInput CUDA context.");

    const auto initResult = optixInit();
    if (initResult != OPTIX_SUCCESS)
        throw std::runtime_error("optixInit failed with result " + std::to_string(initResult));
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = LogCallback;
    options.logCallbackData = this;
    options.logCallbackLevel = 3;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    CheckOptix(optixDeviceContextCreate(mCudaContext, &options, &mOptixContext), "optixDeviceContextCreate");
    CheckOptix(optixDeviceContextSetCacheEnabled(mOptixContext, 0), "optixDeviceContextSetCacheEnabled");
    std::cout << "OptiX SDK " << OPTIX_VERSION << ": validation ALL, same CUDA context and stream as SurfaceInput.\n";
}

void OptixSceneGAS::Build(const Scene &scene)
{
    if (!mOptixContext || !mGeometry.empty())
        throw std::runtime_error("Initialize OptiX first; this static GAS is built once.");
    CheckCuda(cuCtxSetCurrent(mCudaContext), "cuCtxSetCurrent(GAS build)");
    const auto &objects = scene.GetObjects();
    if (objects.empty() || objects.size() > std::numeric_limits<unsigned int>::max())
        throw std::runtime_error("GAS needs at least one supported scene object.");
    mGeometry.resize(objects.size()); // Stable addresses for triangleArray.vertexBuffers.
    std::vector<OptixBuildInput> inputs(objects.size());
    std::vector<unsigned int> flags(objects.size(), OPTIX_GEOMETRY_FLAG_NONE);
    size_t totalVertices = 0, totalTriangles = 0;
    for (size_t objectIndex = 0; objectIndex < objects.size(); ++objectIndex)
    {
        const auto &object = objects[objectIndex];
        if (!object.MeshData) throw std::runtime_error("GAS scene object has no mesh.");
        const auto &sourceVertices = object.MeshData->GetVertices();
        const auto &sourceIndices = object.MeshData->GetIndices();
        if (sourceVertices.empty() || sourceIndices.empty() || sourceIndices.size()%3 != 0 ||
            sourceVertices.size() > std::numeric_limits<unsigned int>::max() ||
            sourceIndices.size()/3 > std::numeric_limits<unsigned int>::max())
            throw std::runtime_error("Unsupported triangle mesh size for GAS.");

        const XMMATRIX world = object.ObjectTransform.GetWorldMatrix();
        const float determinant = XMVectorGetX(XMMatrixDeterminant(world));
        if (!std::isfinite(determinant) || std::abs(determinant) < 1e-12f)
            throw std::runtime_error("GAS surface shading requires an invertible world transform.");
        if (!object.MaterialData) throw std::runtime_error("GAS surface shading requires a material.");
        const auto albedo = object.MaterialData->Albedo;
        if (!std::isfinite(albedo.x) || !std::isfinite(albedo.y) || !std::isfinite(albedo.z) ||
            albedo.x < 0 || albedo.x > 1 || albedo.y < 0 || albedo.y > 1 || albedo.z < 0 || albedo.z > 1)
            throw std::runtime_error("GAS diffuse reflectance must be finite linear RGB in [0,1].");
        const XMMATRIX normalTransform = XMMatrixTranspose(XMMatrixInverse(nullptr,world));
        std::vector<GasVertex> vertices;
        std::vector<GasNormal> normals;
        vertices.reserve(sourceVertices.size());
        normals.reserve(sourceVertices.size());
        for (const auto &source : sourceVertices)
        {
            XMFLOAT3 p;
            XMStoreFloat3(&p, XMVector3TransformCoord(XMLoadFloat3(&source.Position), world));
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                throw std::runtime_error("Non-finite world vertex in GAS.");
            vertices.push_back({p.x, p.y, p.z, 0});
            const auto transformed = XMVector3TransformNormal(XMLoadFloat3(&source.Normal),normalTransform);
            const float length2 = XMVectorGetX(XMVector3LengthSq(transformed));
            if (!std::isfinite(length2) || length2 <= 1e-20f) throw std::runtime_error("GAS needs finite nonzero vertex normals.");
            XMFLOAT3 n;
            XMStoreFloat3(&n,XMVector3Normalize(transformed));
            normals.push_back({n.x,n.y,n.z,0});
        }
        std::vector<GasTriangle> triangles;
        triangles.reserve(sourceIndices.size()/3);
        for (size_t i = 0; i < sourceIndices.size(); i += 3)
        {
            if (sourceIndices[i] >= vertices.size() || sourceIndices[i+1] >= vertices.size() || sourceIndices[i+2] >= vertices.size())
                throw std::runtime_error("GAS mesh index is out of range.");
            triangles.push_back({sourceIndices[i], sourceIndices[i+1], sourceIndices[i+2]});
        }
        auto &geometry = mGeometry[objectIndex];
        geometry.ObjectIndex = static_cast<unsigned int>(objectIndex);
        geometry.VertexCount = static_cast<unsigned int>(vertices.size());
        geometry.TriangleCount = static_cast<unsigned int>(triangles.size());
        geometry.AlbedoR = albedo.x; geometry.AlbedoG = albedo.y; geometry.AlbedoB = albedo.z;
        const size_t vertexBytes = vertices.size()*sizeof(GasVertex);
        const size_t indexBytes = triangles.size()*sizeof(GasTriangle);
        CheckCuda(cuMemAlloc(&geometry.Vertices, vertexBytes), "cuMemAlloc(GAS vertices)");
        CheckCuda(cuMemAlloc(&geometry.Indices, indexBytes), "cuMemAlloc(GAS indices)");
        CheckCuda(cuMemAlloc(&geometry.Normals, normals.size()*sizeof(GasNormal)), "cuMemAlloc(GAS normals)");
        CheckCuda(cuMemcpyHtoD(geometry.Vertices, vertices.data(), vertexBytes), "cuMemcpyHtoD(GAS vertices)");
        CheckCuda(cuMemcpyHtoD(geometry.Indices, triangles.data(), indexBytes), "cuMemcpyHtoD(GAS indices)");
        CheckCuda(cuMemcpyHtoD(geometry.Normals,normals.data(),normals.size()*sizeof(GasNormal)), "cuMemcpyHtoD(GAS normals)");

        auto &input = inputs[objectIndex];
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray.vertexBuffers = &geometry.Vertices;
        input.triangleArray.numVertices = geometry.VertexCount;
        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.vertexStrideInBytes = sizeof(GasVertex);
        input.triangleArray.indexBuffer = geometry.Indices;
        input.triangleArray.numIndexTriplets = geometry.TriangleCount;
        input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        input.triangleArray.indexStrideInBytes = sizeof(GasTriangle);
        input.triangleArray.flags = &flags[objectIndex];
        input.triangleArray.numSbtRecords = 1; // Future hitgroup record order = SceneObject order.
        totalVertices += vertices.size();
        totalTriangles += triangles.size();
    }

    // Pageable HtoD may return after staging, before default-stream DMA completes.
    // Complete one-time geometry uploads before building on the non-blocking stream.
    CheckCuda(cuCtxSynchronize(), "cuCtxSynchronize(GAS geometry upload)");
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    options.motionOptions.numKeys = 1;
    OptixAccelBufferSizes sizes = {};
    CheckOptix(optixAccelComputeMemoryUsage(mOptixContext, &options, inputs.data(),
        static_cast<unsigned int>(inputs.size()), &sizes), "optixAccelComputeMemoryUsage");
    if (!sizes.outputSizeInBytes || !sizes.tempSizeInBytes)
        throw std::runtime_error("OptiX returned an empty GAS allocation size.");
    mOutputBytes = sizes.outputSizeInBytes;
    CheckCuda(cuMemAlloc(&mScratch, sizes.tempSizeInBytes), "cuMemAlloc(GAS scratch)");
    CheckCuda(cuMemAlloc(&mOutput, mOutputBytes), "cuMemAlloc(GAS output)");
    CheckCuda(cuMemAlloc(&mEmittedBounds, sizeof(OptixAabb)), "cuMemAlloc(GAS emitted AABB)");
    OptixAccelEmitDesc emitted = {};
    emitted.type = OPTIX_PROPERTY_TYPE_AABBS;
    emitted.result = mEmittedBounds;
    CheckOptix(optixAccelBuild(mOptixContext, mStream, &options, inputs.data(),
        static_cast<unsigned int>(inputs.size()), mScratch, sizes.tempSizeInBytes,
        mOutput, mOutputBytes, &mHandle, &emitted, 1), "optixAccelBuild");
    CheckCuda(cuStreamSynchronize(mStream), "cuStreamSynchronize(GAS build)");
    if (!mHandle) throw std::runtime_error("OptiX returned an empty traversable handle.");
    CheckCuda(cuMemFree(mScratch), "cuMemFree(GAS scratch)");
    mScratch = 0;
    std::cout << "GAS built: " << mGeometry.size() << " world-space object inputs, "
              << totalVertices << " vertices, " << totalTriangles << " triangles; output="
              << mOutputBytes << " bytes, scratch=" << sizes.tempSizeInBytes << " bytes.\n";
}

void OptixSceneGAS::ValidateAgainstScene(const Scene &scene, std::ostream *report)
{
    CheckCuda(cuCtxSetCurrent(mCudaContext), "cuCtxSetCurrent(GAS validation)");
    CheckCuda(cuStreamSynchronize(mStream), "cuStreamSynchronize(GAS validation)");
    if (!mHandle || mGeometry.size() != scene.GetObjects().size())
        throw std::runtime_error("GAS object/handle contract does not match Scene.");
    auto expectedSceneBounds = EmptyBounds();
    size_t totalVertices = 0, totalTriangles = 0;
    double maxVertexError = 0;
    if (report) *report << std::setprecision(10)
        << "kind,object,vertices,triangles,cpu_min_x,cpu_min_y,cpu_min_z,cpu_max_x,cpu_max_y,cpu_max_z,"
        << "gpu_min_x,gpu_min_y,gpu_min_z,gpu_max_x,gpu_max_y,gpu_max_z,max_vertex_error\n";
    for (size_t objectIndex = 0; objectIndex < mGeometry.size(); ++objectIndex)
    {
        const auto &object = scene.GetObjects()[objectIndex];
        const auto &source = object.MeshData->GetVertices();
        const auto &indices = object.MeshData->GetIndices();
        const auto &geometry = mGeometry[objectIndex];
        if (geometry.ObjectIndex != objectIndex || geometry.VertexCount != source.size() || geometry.TriangleCount != indices.size()/3)
            throw std::runtime_error("GAS per-object counts differ from Scene.");
        std::vector<GasVertex> vertices(geometry.VertexCount);
        std::vector<GasTriangle> triangles(geometry.TriangleCount);
        CheckCuda(cuMemcpyDtoH(vertices.data(), geometry.Vertices, vertices.size()*sizeof(GasVertex)), "cuMemcpyDtoH(GAS vertices)");
        CheckCuda(cuMemcpyDtoH(triangles.data(), geometry.Indices, triangles.size()*sizeof(GasTriangle)), "cuMemcpyDtoH(GAS indices)");
        XMFLOAT4X4 world;
        XMStoreFloat4x4(&world, object.ObjectTransform.GetWorldMatrix());
        std::vector<std::array<double, 3>> expectedVertices(source.size());
        double objectError = 0;
        for (size_t i = 0; i < source.size(); ++i)
        {
            const auto &p = source[i].Position;
            // Independent scalar/double evaluation, not a copy of the upload staging vector.
            expectedVertices[i] = {double(p.x)*world._11+double(p.y)*world._21+double(p.z)*world._31+world._41,
                double(p.x)*world._12+double(p.y)*world._22+double(p.z)*world._32+world._42,
                double(p.x)*world._13+double(p.y)*world._23+double(p.z)*world._33+world._43};
            const double actual[] = {vertices[i].X, vertices[i].Y, vertices[i].Z};
            for (size_t c = 0; c < 3; ++c)
            {
                if (!std::isfinite(actual[c])) throw std::runtime_error("Non-finite GPU GAS vertex.");
                objectError = std::max(objectError, std::abs(actual[c]-expectedVertices[i][c]));
            }
            if (vertices[i].Padding != 0) throw std::runtime_error("Unexpected GAS vertex padding.");
        }
        if (objectError > GeometryTolerance) throw std::runtime_error("GPU GAS vertices do not match world-transformed Scene.");
        auto cpuBounds = EmptyBounds(), gpuBounds = EmptyBounds();
        for (size_t triangle = 0; triangle < triangles.size(); ++triangle)
        {
            const uint32_t actual[] = {triangles[triangle].X, triangles[triangle].Y, triangles[triangle].Z};
            for (size_t corner = 0; corner < 3; ++corner)
            {
                const auto index = indices[triangle*3+corner];
                if (actual[corner] != index) throw std::runtime_error("GPU GAS indices differ from original Scene mesh.");
                Include(cpuBounds, expectedVertices[index]);
                Include(expectedSceneBounds, expectedVertices[index]);
                Include(gpuBounds, {vertices[index].X, vertices[index].Y, vertices[index].Z});
            }
        }
        if (report)
        {
            *report << "geometry," << objectIndex << ',' << geometry.VertexCount << ',' << geometry.TriangleCount;
            WriteBounds(*report, cpuBounds);
            WriteBounds(*report, gpuBounds);
            *report << ',' << objectError << '\n';
        }
        maxVertexError = std::max(maxVertexError, objectError);
        totalVertices += geometry.VertexCount;
        totalTriangles += geometry.TriangleCount;
    }
    OptixAabb bounds = {};
    CheckCuda(cuMemcpyDtoH(&bounds, mEmittedBounds, sizeof(bounds)), "cuMemcpyDtoH(OptiX root bounds)");
    const std::array<double, 6> actualBounds = {bounds.minX,bounds.minY,bounds.minZ,bounds.maxX,bounds.maxY,bounds.maxZ};
    double maxBoundsError = 0;
    for (size_t c = 0; c < 6; ++c)
    {
        if (!std::isfinite(actualBounds[c])) throw std::runtime_error("Non-finite OptiX root AABB.");
        maxBoundsError = std::max(maxBoundsError, std::abs(actualBounds[c]-expectedSceneBounds[c]));
    }
    if (maxBoundsError > GeometryTolerance) throw std::runtime_error("OptiX GAS root AABB differs from original world-space Scene.");
    OptixRelocationInfo info = {};
    CheckOptix(optixAccelGetRelocationInfo(mOptixContext, mHandle, &info), "optixAccelGetRelocationInfo(handle validation)");
    if (mValidationIssues.load() != 0) throw std::runtime_error("OptiX reported validation warnings/errors.");
    if (report)
    {
        *report << "gas,-1," << totalVertices << ',' << totalTriangles;
        WriteBounds(*report, expectedSceneBounds);
        WriteBounds(*report, actualBounds);
        *report << ',' << maxVertexError << '\n';
    }
    std::cout << std::setprecision(10) << "PASS: GAS upload/index counts and emitted root AABB; max vertex error="
              << maxVertexError << ", max bounds error=" << maxBoundsError << "; AABB=("
              << bounds.minX << ',' << bounds.minY << ',' << bounds.minZ << ")..("
              << bounds.maxX << ',' << bounds.maxY << ',' << bounds.maxZ << ").\n"
              << "PASS: OptiX validation reported no errors or unexpected warnings.\n";
    if (mAllocationNotices.load())
        std::cout << "NOTE: " << mAllocationNotices.load()
            << " known benign OptiX launcher allocation notices were logged (NVIDIA forum 301834/2).\n";
}
