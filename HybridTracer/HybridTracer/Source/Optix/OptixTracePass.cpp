#include "OptixTracePass.h"
#include "OptixSceneGAS.h"
#include "Interop/InteropCommon.h"

#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

using namespace HybridInterop;

namespace
{
    void CheckOptix(OptixResult result, const char *operation, const char *log = "")
    {
        if (result != OPTIX_SUCCESS)
            throw std::runtime_error(std::string(operation) + ": " + optixGetErrorName(result) + "\n" + log);
    }
    void CheckNvrtc(nvrtcResult result, const char *operation)
    {
        if (result != NVRTC_SUCCESS)
            throw std::runtime_error(std::string(operation) + ": " + nvrtcGetErrorString(result));
    }
    std::string ReadText(const std::filesystem::path &path)
    {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Cannot read OptiX device source/header: " + path.u8string());
        return {(std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()};
    }
    std::vector<char> CompileDeviceCode()
    {
        wchar_t executable[MAX_PATH] = {};
        GetModuleFileNameW(nullptr, executable, MAX_PATH);
        const auto folder = std::filesystem::path(executable).parent_path() / L"Shaders";
        const auto source = ReadText(folder / L"OptixShadow.cu");
        const auto surface = ReadText(folder / L"SurfaceInput.h");
        const auto shared = ReadText(folder / L"OptixTraceShared.h");
        std::vector<std::string> headerSources = {surface, shared,ReadText(folder/L"OptixSceneData.h"),
            ReadText(folder/L"SurfaceHit.h"),ReadText(folder/L"OptixSurface.cuh"),
            ReadText(folder/L"IndirectLighting.h"),ReadText(folder/L"OptixIndirect.cuh"),
            ReadText(folder/L"DiffuseBSDFTypes.h"),ReadText(folder/L"DiffuseBSDF.cuh")};
        std::vector<std::string> headerNames = {"SurfaceInput.h", "OptixTraceShared.h","OptixSceneData.h","SurfaceHit.h","OptixSurface.cuh",
            "IndirectLighting.h","OptixIndirect.cuh","DiffuseBSDFTypes.h","DiffuseBSDF.cuh"};
        // NVRTC 12.0 on Windows cannot open this SDK's Korean path via -I.
        // Read with filesystem's Unicode path support and supply named in-memory headers.
        // SDK files stay external; they are never copied into the project/executable folder.
        const auto sdkInclude = std::filesystem::u8path(HYBRIDTRACER_OPTIX_INCLUDE_DIR);
        for (const auto &entry : std::filesystem::recursive_directory_iterator(sdkInclude))
        {
            if (!entry.is_regular_file() || entry.path().extension() != L".h") continue;
            headerNames.push_back(std::filesystem::relative(entry.path(),sdkInclude).generic_u8string());
            headerSources.push_back(ReadText(entry.path()));
        }
        std::vector<const char *> headers, names;
        for (size_t i = 0; i < headerSources.size(); ++i)
        {
            headers.push_back(headerSources[i].c_str());
            names.push_back(headerNames[i].c_str());
        }
        struct Program
        {
            nvrtcProgram Handle = nullptr;
            ~Program() { if (Handle) nvrtcDestroyProgram(&Handle); }
        } program;
        CheckNvrtc(nvrtcCreateProgram(&program.Handle, source.c_str(), "OptixShadow.cu",
            static_cast<int>(headers.size()), headers.data(), names.data()), "nvrtcCreateProgram(shadow)");
        CUdevice device;
        CheckCuda(cuCtxGetDevice(&device), "cuCtxGetDevice(shadow)");
        int major = 0, minor = 0;
        CheckCuda(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute(major)");
        CheckCuda(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute(minor)");
        const std::string architecture = "--gpu-architecture=compute_" + std::to_string(major*10+minor);
        const std::string cudaInclude = std::string("--include-path=") + HYBRIDTRACER_CUDA_INCLUDE_DIR;
        const char *options[] = {architecture.c_str(), "--std=c++14", "--fmad=false", cudaInclude.c_str(),
            "--device-as-default-execution-space"};
        const auto result = nvrtcCompileProgram(program.Handle, 5, options);
        size_t logBytes = 0;
        CheckNvrtc(nvrtcGetProgramLogSize(program.Handle, &logBytes), "nvrtcGetProgramLogSize(shadow)");
        std::vector<char> log(logBytes);
        CheckNvrtc(nvrtcGetProgramLog(program.Handle, log.data()), "nvrtcGetProgramLog(shadow)");
        if (result != NVRTC_SUCCESS)
            throw std::runtime_error(std::string("NVRTC OptiX compile failed: ") + log.data());
        size_t bytes = 0;
        CheckNvrtc(nvrtcGetPTXSize(program.Handle, &bytes), "nvrtcGetPTXSize(shadow)");
        std::vector<char> ptx(bytes);
        CheckNvrtc(nvrtcGetPTX(program.Handle, ptx.data()), "nvrtcGetPTX(shadow)");
        std::cout << "OptiX shadow device programs compiled with NVRTC for compute_" << major << minor << ".\n";
        return ptx;
    }
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord
    {
        char Header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitRecord
    {
        char Header[OPTIX_SBT_RECORD_HEADER_SIZE];
        SceneHitData Data;
    };
    static_assert(offsetof(HitRecord, Data) == OPTIX_SBT_RECORD_HEADER_SIZE);
    static_assert(sizeof(HitRecord) == 80);

    template<class Record> void UploadRecords(CUdeviceptr &output, const Record *records, size_t count)
    {
        CheckCuda(cuMemAlloc(&output, sizeof(Record)*count), "cuMemAlloc(SBT)");
        CheckCuda(cuMemcpyHtoD(output, records, sizeof(Record)*count), "cuMemcpyHtoD(SBT)");
    }
}

OptixTracePass::~OptixTracePass()
{
    if (mCudaContext) cuCtxSetCurrent(mCudaContext);
    if (mStream) cuStreamSynchronize(mStream);
    if (mLaunchDone) cuEventDestroy(mLaunchDone);
    if (mHostParams) cuMemFreeHost(mHostParams);
    if (mParams) cuMemFree(mParams);
    if (mOutput) cuMemFree(mOutput);
    if (mIndirect) cuMemFree(mIndirect);
    if (mIndirectHits) cuMemFree(mIndirectHits);
    if (mIndirectDebug) cuMemFree(mIndirectDebug);
    if (mSbt.raygenRecord) cuMemFree(mSbt.raygenRecord);
    if (mSbt.missRecordBase) cuMemFree(mSbt.missRecordBase);
    if (mSbt.hitgroupRecordBase) cuMemFree(mSbt.hitgroupRecordBase);
    if (mPipeline) optixPipelineDestroy(mPipeline);
    if (mHit) optixProgramGroupDestroy(mHit);
    if (mMiss) optixProgramGroupDestroy(mMiss);
    if (mRaygen) optixProgramGroupDestroy(mRaygen);
    if (mModule) optixModuleDestroy(mModule);
}

namespace
{
    TraceLight PackLight(const SceneLight &light)
    {
        using namespace DirectX;
        const auto store = [](float (&target)[3], const XMFLOAT3 &value)
        {
            if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
                throw std::runtime_error("Light vectors must be finite.");
            target[0]=value.x; target[1]=value.y; target[2]=value.z;
        };
        return std::visit([&](const auto &source)
        {
            TraceLight packed = {};
            using Type = std::decay_t<decltype(source)>;
            float strength;
            if constexpr (std::is_same_v<Type,DirectionalLight>)
            {
                packed.Kind = TraceLightKind::Directional;
                XMFLOAT3 direction;
                XMStoreFloat3(&direction, XMVector3Normalize(XMLoadFloat3(&source.Direction)));
                store(packed.PositionOrDirection, direction);
                if (XMVectorGetX(XMVector3LengthSq(XMLoadFloat3(&direction))) < 0.9999f)
                    throw std::runtime_error("Directional light needs a nonzero direction.");
                strength = source.Intensity;
            }
            else if constexpr (std::is_same_v<Type,PointLight>)
            {
                packed.Kind = TraceLightKind::Point;
                store(packed.PositionOrDirection, source.Position);
                strength = source.Intensity;
            }
            else
            {
                packed.Kind = TraceLightKind::Rectangle;
                store(packed.PositionOrDirection, source.Center);
                store(packed.HalfU, source.HalfU);
                store(packed.HalfV, source.HalfV);
                const auto u = XMLoadFloat3(&source.HalfU);
                const auto v = XMLoadFloat3(&source.HalfV);
                const auto cross = XMVector3Cross(u,v);
                packed.Area = 4*XMVectorGetX(XMVector3Length(cross));
                const auto cosine = XMVector3Dot(XMVector3Normalize(u),XMVector3Normalize(v));
                if (!std::isfinite(packed.Area) || packed.Area <= 0 || std::abs(XMVectorGetX(cosine)) > 0.0001f)
                    throw std::runtime_error("Rectangle light needs finite, nonzero perpendicular half edges.");
                XMFLOAT3 normal;
                XMStoreFloat3(&normal, XMVector3Normalize(cross));
                store(packed.Normal, normal);
                strength = source.Radiance;
            }
            const auto &color = source.Color;
            if (!std::isfinite(strength) || strength < 0 || color.x < 0 || color.y < 0 || color.z < 0)
                throw std::runtime_error("Light color and strength must be finite and nonnegative.");
            store(packed.Emission, {color.x*strength,color.y*strength,color.z*strength});
            return packed;
        }, light);
    }
}

void OptixTracePass::Initialize(CUcontext context, CUstream stream, const OptixSceneGAS &gas,
    unsigned int width, unsigned int height, bool enableIndirect, bool enableDiagnostics,
    const std::vector<SceneLight> &additionalLights)
{
    if (mCudaContext || !context || !stream || !gas.GetContext() || !gas.GetHandle() ||
        !width || !height || static_cast<size_t>(width)*height > std::numeric_limits<unsigned int>::max())
        throw std::runtime_error("OptixTracePass needs a new object, live GAS, same CUDA context/stream, and supported dimensions.");
    if (additionalLights.size() > MaxAdditionalLights || (!enableIndirect && !additionalLights.empty()) ||
        (!additionalLights.empty() && size_t(width)*height > std::numeric_limits<unsigned int>::max()/AdditionalLightingPlanes))
        throw std::runtime_error("Additional lighting requires lighting outputs, at most 16 lights and supported dimensions.");
    mAdditionalLightCount = static_cast<unsigned int>(additionalLights.size());
    std::transform(additionalLights.begin(),additionalLights.end(),mAdditionalLights.begin(),PackLight);
    mCudaContext = context;
    mStream = stream;
    mOptixContext = gas.GetContext();
    mGas = gas.GetHandle();
    mWidth = width;
    mHeight = height;
    CheckCuda(cuCtxSetCurrent(context), "cuCtxSetCurrent(trace initialize)");
    CUcontext streamContext = nullptr;
    CheckCuda(cuStreamGetCtx(stream, &streamContext), "cuStreamGetCtx(trace)");
    if (streamContext != context || gas.GetCudaContext() != context)
        throw std::runtime_error("Trace stream, GAS and SurfaceInput must share one CUDA context.");
    CheckCuda(cuEventCreate(&mLaunchDone, CU_EVENT_DISABLE_TIMING), "cuEventCreate(trace)");

    const auto ptx = CompileDeviceCode();
    OptixModuleCompileOptions moduleOptions = {};
    moduleOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    OptixPipelineCompileOptions compile = {};
    compile.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    compile.numPayloadValues = 2;
    compile.numAttributeValues = 2;
    compile.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    compile.pipelineLaunchParamsVariableName = "params";
    compile.usesPrimitiveTypeFlags = static_cast<unsigned int>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    char log[8192] = {};
    size_t logSize = sizeof(log);
    auto result = optixModuleCreate(mOptixContext, &moduleOptions, &compile, ptx.data(), ptx.size(), log, &logSize, &mModule);
    CheckOptix(result, "optixModuleCreate(shadow)", log);

    OptixProgramGroupOptions groupOptions = {};
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = mModule;
    desc.raygen.entryFunctionName = "__raygen__shadow";
    logSize = sizeof(log);
    result = optixProgramGroupCreate(mOptixContext, &desc, 1, &groupOptions, log, &logSize, &mRaygen);
    CheckOptix(result, "optixProgramGroupCreate(raygen)", log);
    desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = mModule;
    desc.miss.entryFunctionName = "__miss__shadow";
    logSize = sizeof(log);
    result = optixProgramGroupCreate(mOptixContext, &desc, 1, &groupOptions, log, &logSize, &mMiss);
    CheckOptix(result, "optixProgramGroupCreate(miss)", log);
    desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc.hitgroup.moduleCH = mModule;
    desc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    logSize = sizeof(log);
    result = optixProgramGroupCreate(mOptixContext, &desc, 1, &groupOptions, log, &logSize, &mHit);
    CheckOptix(result, "optixProgramGroupCreate(hit)", log);

    const std::array<OptixProgramGroup, 3> groups = {mRaygen, mMiss, mHit};
    OptixPipelineLinkOptions link = {};
    link.maxTraceDepth = 1;
    logSize = sizeof(log);
    result = optixPipelineCreate(mOptixContext, &compile, &link, groups.data(), static_cast<unsigned int>(groups.size()),
        log, &logSize, &mPipeline);
    CheckOptix(result, "optixPipelineCreate(shadow)", log);
    OptixStackSizes stack = {};
    for (const auto group : groups)
        CheckOptix(optixUtilAccumulateStackSizes(group, &stack, mPipeline), "optixUtilAccumulateStackSizes");
    unsigned int traversal = 0, state = 0, continuation = 0;
    CheckOptix(optixUtilComputeStackSizes(&stack, 1, 0, 0, &traversal, &state, &continuation), "optixUtilComputeStackSizes");
    CheckOptix(optixPipelineSetStackSize(mPipeline, traversal, state, continuation, 1), "optixPipelineSetStackSize");

    EmptyRecord raygen = {}, miss = {};
    CheckOptix(optixSbtRecordPackHeader(mRaygen, &raygen), "optixSbtRecordPackHeader(raygen)");
    CheckOptix(optixSbtRecordPackHeader(mMiss, &miss), "optixSbtRecordPackHeader(miss)");
    UploadRecords(mSbt.raygenRecord, &raygen, 1);
    UploadRecords(mSbt.missRecordBase, &miss, 1);
    mSbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    mSbt.missRecordCount = 1;
    std::vector<HitRecord> hits(gas.GetGeometry().size());
    for (size_t i = 0; i < hits.size(); ++i)
    {
        CheckOptix(optixSbtRecordPackHeader(mHit, &hits[i]), "optixSbtRecordPackHeader(hit)");
        const auto &geometry = gas.GetGeometry()[i];
        hits[i].Data = {reinterpret_cast<const GasVertex *>(geometry.Vertices),
            reinterpret_cast<const GasTriangle *>(geometry.Indices),reinterpret_cast<const GasNormal *>(geometry.Normals),
            geometry.ObjectIndex,geometry.TriangleCount,geometry.AlbedoR,geometry.AlbedoG,geometry.AlbedoB,0};
    }
    UploadRecords(mSbt.hitgroupRecordBase, hits.data(), hits.size());
    mSbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);
    mSbt.hitgroupRecordCount = static_cast<unsigned int>(hits.size());
    // SBT records were uploaded from pageable staging vectors on the default stream.
    // Finish initialization uploads before any launch on the non-blocking frame stream.
    CheckCuda(cuCtxSynchronize(), "cuCtxSynchronize(SBT upload)");
    mOutputBytes = static_cast<size_t>(width)*height*sizeof(ShadowResult);
    CheckCuda(cuMemAlloc(&mOutput, mOutputBytes), "cuMemAlloc(shadow output)");
    if (enableIndirect)
    {
        const size_t pixels = static_cast<size_t>(width)*height;
        CheckCuda(cuMemAlloc(&mIndirect,GetIndirectBytes()),"cuMemAlloc(diffuse lighting)");
        CheckCuda(cuMemAlloc(&mIndirectHits,pixels*sizeof(SurfaceHit)),"cuMemAlloc(indirect hit scratch)");
        if (enableDiagnostics)
            CheckCuda(cuMemAlloc(&mIndirectDebug,pixels*sizeof(IndirectSampleDebug)),"cuMemAlloc(indirect diagnostics)");
    }
    else if (enableDiagnostics) throw std::runtime_error("Indirect diagnostics need indirect mode.");
    CheckCuda(cuMemAlloc(&mParams, sizeof(OptixTraceLaunchParams)), "cuMemAlloc(trace params)");
    CheckCuda(cuMemHostAlloc(reinterpret_cast<void **>(&mHostParams), sizeof(OptixTraceLaunchParams), 0), "cuMemHostAlloc(trace params)");
    std::cout << "OptiX shadow pipeline ready: 1 raygen, 1 miss, " << hits.size()
              << " object hit records; trace depth=1, single GAS; output=" << mOutputBytes << " bytes.\n";
}

namespace
{
void ValidateShadowSettings(const ShadowSettings &settings)
{
    const float lengthSquared = settings.LightX*settings.LightX + settings.LightY*settings.LightY + settings.LightZ*settings.LightZ;
    if (!std::isfinite(lengthSquared) || std::abs(lengthSquared-1.0f) > 0.00001f ||
        !std::isfinite(settings.NormalOffset) || settings.NormalOffset <= 0 ||
        !std::isfinite(settings.TMin) || !std::isfinite(settings.TMax) || settings.TMin < 0 || settings.TMax <= settings.TMin)
        throw std::runtime_error("Shadow settings need unit surface-to-light direction and finite positive ray bounds/offset.");
}
}
void OptixTracePass::Launch(CUdeviceptr surfaces, const ShadowSettings &settings)
{
    if (!mPipeline || !surfaces) throw std::runtime_error("Initialize trace pass and provide a device SurfaceInput array.");
    ValidateShadowSettings(settings);
    Enqueue({reinterpret_cast<const SurfaceInput *>(surfaces), reinterpret_cast<ShadowResult *>(mOutput),
        mGas,mWidth,mHeight,settings,nullptr,nullptr});
}

void OptixTracePass::LaunchIndirect(CUdeviceptr surfaces, const ShadowSettings &settings, const IndirectSettings &indirect)
{
    if (!mIndirect || !surfaces) throw std::runtime_error("Enable indirect lighting and provide SurfaceInput.");
    ValidateShadowSettings(settings);
    if (!std::isfinite(indirect.CameraX) || !std::isfinite(indirect.CameraY) || !std::isfinite(indirect.CameraZ) ||
        !std::isfinite(indirect.IrradianceR) || !std::isfinite(indirect.IrradianceG) || !std::isfinite(indirect.IrradianceB) ||
        indirect.IrradianceR < 0 || indirect.IrradianceG < 0 || indirect.IrradianceB < 0 ||
        !indirect.SamplesPerPixel || indirect.SamplesPerPixel > MaxIndirectSamples)
        throw std::runtime_error("Indirect lighting requires finite camera/irradiance, nonnegative light and 1..4096 samples.");
    if (indirect.MaxBounces > MaxIndirectBounces)
        throw std::runtime_error("Indirect bounce limit exceeds the supported maximum.");
    OptixTraceLaunchParams parameters = {};
    parameters.Surfaces = reinterpret_cast<const SurfaceInput *>(surfaces);
    parameters.Output = reinterpret_cast<ShadowResult *>(mOutput);
    parameters.Gas=mGas; parameters.Width=mWidth; parameters.Height=mHeight; parameters.Settings=settings;
    parameters.QueryHits = reinterpret_cast<SurfaceHit *>(mIndirectHits);
    parameters.IndirectOutput = reinterpret_cast<IndirectLighting *>(mIndirect);
    parameters.IndirectDebug = reinterpret_cast<IndirectSampleDebug *>(mIndirectDebug);
    parameters.Indirect=indirect;
    parameters.AdditionalLightCount=mAdditionalLightCount;
    std::copy(mAdditionalLights.begin(),mAdditionalLights.end(),parameters.AdditionalLights);
    Enqueue(parameters);
}

void OptixTracePass::LaunchSurfaceQueries(CUdeviceptr rays, CUdeviceptr hits)
{
    if (!mPipeline || !rays || !hits || rays == hits || hits == mOutput)
        throw std::runtime_error("Initialize trace pass and supply distinct GPU SurfaceRay/SurfaceHit buffers.");
    Enqueue({nullptr,nullptr,mGas,mWidth,mHeight,{},reinterpret_cast<const SurfaceRay *>(rays),reinterpret_cast<SurfaceHit *>(hits)});
}

void OptixTracePass::Enqueue(const OptixTraceLaunchParams &parameters)
{
    CheckCuda(cuCtxSetCurrent(mCudaContext), "cuCtxSetCurrent(trace launch)");
    if (mLaunched) CheckCuda(cuEventQuery(mLaunchDone), "Previous trace must complete before reusing launch params/output");
    *mHostParams = parameters;
    CheckCuda(cuMemcpyHtoDAsync(mParams,mHostParams,sizeof(OptixTraceLaunchParams),mStream), "cuMemcpyHtoDAsync(trace params)");
    RecordCudaTiming(mTiming,CudaTimingMark::TraceBegin,mStream);
    CheckOptix(optixLaunch(mPipeline,mStream,mParams,sizeof(OptixTraceLaunchParams),&mSbt,mWidth,mHeight,1), "optixLaunch(trace)");
    RecordCudaTiming(mTiming,CudaTimingMark::TraceEnd,mStream);
    CheckCuda(cuEventRecord(mLaunchDone, mStream), "cuEventRecord(trace)");
    mLaunched = true;
}
