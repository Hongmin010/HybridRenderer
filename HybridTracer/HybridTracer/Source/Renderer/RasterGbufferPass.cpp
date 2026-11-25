#include "RasterGBufferPass.h"

#include "Scene/Camera.h"
#include "Scene/Scene.h"

#include <Windows.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>

#include <climits>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

namespace
{
    struct ObjectConstants
    {
        XMFLOAT4X4 WorldViewProjection;
        XMFLOAT4X4 WorldInverseTranspose;

        XMFLOAT4 Albedo;
    };

    struct PassConstants
    {
        XMFLOAT4 LightDirectionIntensity;
        XMFLOAT4 LightColorUnused;
    };

    constexpr UINT AlignConstantBufferSize(
        UINT size)
    {
        return (size + 255) & ~255;
    }

    void CheckHR(HRESULT hr)
    {
        if (FAILED(hr))
        {
            throw std::runtime_error(
                "RasterGBufferPass D3D12 call failed.");
        }
    }

    std::wstring GetShaderPath(
        const wchar_t *filename)
    {
        wchar_t executablePath[MAX_PATH] = {};

        GetModuleFileNameW(
            nullptr,
            executablePath,
            MAX_PATH);

        std::filesystem::path path(
            executablePath);

        path =
            path.parent_path() / L"Shaders" / filename;

        return path.wstring();
    }

    ComPtr<ID3DBlob> CompileShader(
        const wchar_t *filename,
        const char *entryPoint,
        const char *target)
    {
        std::wstring shaderPath =
            GetShaderPath(filename);

        UINT compileFlags =
            D3DCOMPILE_ENABLE_STRICTNESS;

#ifdef _DEBUG
        compileFlags |=
            D3DCOMPILE_DEBUG |
            D3DCOMPILE_SKIP_OPTIMIZATION;
#endif

        ComPtr<ID3DBlob> shaderBlob;
        ComPtr<ID3DBlob> errorBlob;

        HRESULT hr =
            D3DCompileFromFile(
                shaderPath.c_str(),
                nullptr,
                D3D_COMPILE_STANDARD_FILE_INCLUDE,
                entryPoint,
                target,
                compileFlags,
                0,
                &shaderBlob,
                &errorBlob);

        if (FAILED(hr))
        {
            if (errorBlob)
            {
                std::string message(
                    static_cast<const char *>(
                        errorBlob->GetBufferPointer()),
                    errorBlob->GetBufferSize());

                throw std::runtime_error(
                    message);
            }

            throw std::runtime_error(
                "Failed to compile shader.");
        }

        return shaderBlob;
    }
}

RasterGBufferPass::~RasterGBufferPass()
{
    for (auto &buffer : mObjectConstants)
    {
        if (buffer.Resource &&
            buffer.MappedData)
        {
            buffer.Resource->Unmap(
                0,
                nullptr);

            buffer.MappedData = nullptr;
        }
    }

    for (auto &buffer : mPassConstants)
    {
        if (buffer.Resource &&
            buffer.MappedData)
        {
            buffer.Resource->Unmap(
                0,
                nullptr);

            buffer.MappedData = nullptr;
        }
    }
}

void RasterGBufferPass::Initialize(
    ID3D12Device *device,
    DXGI_FORMAT renderTargetFormat,
    UINT width,
    UINT height,
    UINT frameCount,
    bool shareNormalWithCuda,
    bool shareDepthWithCuda,
    bool shareAlbedoWithCuda)
{
    CreatePipeline(
        device,
        renderTargetFormat);

    CreateDepthBuffer(
        device,
        width,
        height);

    CreateGBufferTargets(
        device,
        width,
        height,
        shareNormalWithCuda,
        shareDepthWithCuda,
        shareAlbedoWithCuda);

    CreateConstantBuffers(
        device,
        frameCount);

    mViewport.TopLeftX = 0.0f;
    mViewport.TopLeftY = 0.0f;

    mViewport.Width =
        static_cast<float>(width);

    mViewport.Height =
        static_cast<float>(height);

    mViewport.MinDepth = 0.0f;
    mViewport.MaxDepth = 1.0f;

    mScissorRect.left = 0;
    mScissorRect.top = 0;

    mScissorRect.right =
        static_cast<LONG>(width);

    mScissorRect.bottom =
        static_cast<LONG>(height);
}

void RasterGBufferPass::CreatePipeline(
    ID3D12Device *device,
    DXGI_FORMAT backBufferFormat)
{
    //
    // Root Signature
    //
    // b0 = Object + Material
    // b1 = Light / Pass
    //

    D3D12_ROOT_PARAMETER
    rootParameters[2] = {};

    //
    // b0
    //

    rootParameters[0].ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_CBV;

    rootParameters[0]
        .Descriptor.ShaderRegister = 0;

    rootParameters[0]
        .Descriptor.RegisterSpace = 0;

    rootParameters[0].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_ALL;

    //
    // b1
    //

    rootParameters[1].ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_CBV;

    rootParameters[1]
        .Descriptor.ShaderRegister = 1;

    rootParameters[1]
        .Descriptor.RegisterSpace = 0;

    rootParameters[1].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC
    rootDesc = {};

    rootDesc.NumParameters = 2;
    rootDesc.pParameters = rootParameters;

    rootDesc.NumStaticSamplers = 0;
    rootDesc.pStaticSamplers = nullptr;

    rootDesc.Flags =
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ComPtr<ID3DBlob>
        serializedRootSignature;

    ComPtr<ID3DBlob>
        rootError;

    HRESULT rootResult =
        D3D12SerializeRootSignature(
            &rootDesc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &serializedRootSignature,
            &rootError);

    if (FAILED(rootResult))
    {
        if (rootError)
        {
            std::string message(
                static_cast<const char *>(
                    rootError->GetBufferPointer()),
                rootError->GetBufferSize());

            throw std::runtime_error(
                message);
        }

        CheckHR(rootResult);
    }

    CheckHR(
        device->CreateRootSignature(
            0,

            serializedRootSignature
                ->GetBufferPointer(),

            serializedRootSignature
                ->GetBufferSize(),

            IID_PPV_ARGS(
                &mRootSignature)));

    //
    // Shader
    //

    ComPtr<ID3DBlob> vertexShader =
        CompileShader(
            L"Raster.hlsl",
            "VSMain",
            "vs_5_1");

    ComPtr<ID3DBlob> pixelShader =
        CompileShader(
            L"Raster.hlsl",
            "PSMain",
            "ps_5_1");

    //
    // Input Layout
    //

    D3D12_INPUT_ELEMENT_DESC
    inputElements[] =
        {
            {"POSITION",
             0,
             DXGI_FORMAT_R32G32B32_FLOAT,
             0,
             0,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
             0},

            {"NORMAL",
             0,
             DXGI_FORMAT_R32G32B32_FLOAT,
             0,
             12,
             D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
             0}};

    //
    // Rasterizer
    //

    D3D12_RASTERIZER_DESC
    rasterizerDesc = {};

    rasterizerDesc.FillMode =
        D3D12_FILL_MODE_SOLID;

    rasterizerDesc.CullMode =
        D3D12_CULL_MODE_NONE;

    rasterizerDesc.FrontCounterClockwise =
        FALSE;

    rasterizerDesc.DepthBias =
        D3D12_DEFAULT_DEPTH_BIAS;

    rasterizerDesc.DepthBiasClamp =
        D3D12_DEFAULT_DEPTH_BIAS_CLAMP;

    rasterizerDesc.SlopeScaledDepthBias =
        D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;

    rasterizerDesc.DepthClipEnable =
        TRUE;

    rasterizerDesc.MultisampleEnable =
        FALSE;

    rasterizerDesc.AntialiasedLineEnable =
        FALSE;

    rasterizerDesc.ForcedSampleCount =
        0;

    rasterizerDesc.ConservativeRaster =
        D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;

    //
    // Blend
    //

    D3D12_BLEND_DESC
    blendDesc = {};

    blendDesc.AlphaToCoverageEnable =
        FALSE;

    blendDesc.IndependentBlendEnable =
        FALSE;

    D3D12_RENDER_TARGET_BLEND_DESC
    targetBlend = {};

    targetBlend.BlendEnable =
        FALSE;

    targetBlend.LogicOpEnable =
        FALSE;

    targetBlend.SrcBlend =
        D3D12_BLEND_ONE;

    targetBlend.DestBlend =
        D3D12_BLEND_ZERO;

    targetBlend.BlendOp =
        D3D12_BLEND_OP_ADD;

    targetBlend.SrcBlendAlpha =
        D3D12_BLEND_ONE;

    targetBlend.DestBlendAlpha =
        D3D12_BLEND_ZERO;

    targetBlend.BlendOpAlpha =
        D3D12_BLEND_OP_ADD;

    targetBlend.LogicOp =
        D3D12_LOGIC_OP_NOOP;

    targetBlend.RenderTargetWriteMask =
        D3D12_COLOR_WRITE_ENABLE_ALL;

    //
    // RT0~RT3 전부 쓰기 허용
    //

    for (UINT i = 0; i < 4; ++i)
    {
        blendDesc.RenderTarget[i] =
            targetBlend;
    }

    //
    // Depth Test
    //

    D3D12_DEPTH_STENCIL_DESC
    depthStencilDesc = {};

    depthStencilDesc.DepthEnable =
        TRUE;

    depthStencilDesc.DepthWriteMask =
        D3D12_DEPTH_WRITE_MASK_ALL;

    depthStencilDesc.DepthFunc =
        D3D12_COMPARISON_FUNC_LESS;

    depthStencilDesc.StencilEnable =
        FALSE;

    //
    // PSO
    //

    D3D12_GRAPHICS_PIPELINE_STATE_DESC
    psoDesc = {};

    psoDesc.pRootSignature =
        mRootSignature.Get();

    psoDesc.VS =
        {
            vertexShader->GetBufferPointer(),
            vertexShader->GetBufferSize()};

    psoDesc.PS =
        {
            pixelShader->GetBufferPointer(),
            pixelShader->GetBufferSize()};

    psoDesc.BlendState =
        blendDesc;

    psoDesc.SampleMask =
        UINT_MAX;

    psoDesc.RasterizerState =
        rasterizerDesc;

    psoDesc.DepthStencilState =
        depthStencilDesc;

    psoDesc.InputLayout =
        {
            inputElements,
            2};

    psoDesc.PrimitiveTopologyType =
        D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

    //
    // MRT
    //

    psoDesc.NumRenderTargets = 4;

    //
    // RT0 = BackBuffer
    //

    psoDesc.RTVFormats[0] =
        backBufferFormat;

    //
    // RT1 = Albedo
    //

    psoDesc.RTVFormats[1] =
        AlbedoFormat;

    //
    // RT2 = World Normal
    //

    psoDesc.RTVFormats[2] =
        NormalFormat;

    //
    // RT3 = Raster Depth
    //

    psoDesc.RTVFormats[3] =
        RasterDepthFormat;

    psoDesc.DSVFormat =
        DXGI_FORMAT_D32_FLOAT;

    psoDesc.SampleDesc.Count = 1;

    CheckHR(
        device->CreateGraphicsPipelineState(
            &psoDesc,
            IID_PPV_ARGS(
                &mPipelineState)));
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    CheckHR(device->CreateGraphicsPipelineState(&psoDesc,IID_PPV_ARGS(&mTwoSidedPipelineState)));
}

void RasterGBufferPass::CreateDepthBuffer(
    ID3D12Device *device,
    UINT width,
    UINT height)
{
    D3D12_DESCRIPTOR_HEAP_DESC
    heapDesc = {};

    heapDesc.Type =
        D3D12_DESCRIPTOR_HEAP_TYPE_DSV;

    heapDesc.NumDescriptors = 1;

    heapDesc.Flags =
        D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    CheckHR(
        device->CreateDescriptorHeap(
            &heapDesc,
            IID_PPV_ARGS(
                &mDsvHeap)));

    D3D12_HEAP_PROPERTIES
    heapProperties = {};

    heapProperties.Type =
        D3D12_HEAP_TYPE_DEFAULT;

    heapProperties.CreationNodeMask = 1;
    heapProperties.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC
    depthDesc = {};

    depthDesc.Dimension =
        D3D12_RESOURCE_DIMENSION_TEXTURE2D;

    depthDesc.Width =
        width;

    depthDesc.Height =
        height;

    depthDesc.DepthOrArraySize = 1;
    depthDesc.MipLevels = 1;

    depthDesc.Format =
        DXGI_FORMAT_D32_FLOAT;

    depthDesc.SampleDesc.Count = 1;

    depthDesc.Layout =
        D3D12_TEXTURE_LAYOUT_UNKNOWN;

    depthDesc.Flags =
        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE
    clearValue = {};

    clearValue.Format =
        DXGI_FORMAT_D32_FLOAT;

    clearValue.DepthStencil.Depth =
        1.0f;

    clearValue.DepthStencil.Stencil =
        0;

    CheckHR(
        device->CreateCommittedResource(
            &heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &depthDesc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            &clearValue,
            IID_PPV_ARGS(
                &mDepthStencilBuffer)));

    D3D12_DEPTH_STENCIL_VIEW_DESC
    dsvDesc = {};

    dsvDesc.Format =
        DXGI_FORMAT_D32_FLOAT;

    dsvDesc.ViewDimension =
        D3D12_DSV_DIMENSION_TEXTURE2D;

    dsvDesc.Flags =
        D3D12_DSV_FLAG_NONE;

    device->CreateDepthStencilView(
        mDepthStencilBuffer.Get(),
        &dsvDesc,

        mDsvHeap
            ->GetCPUDescriptorHandleForHeapStart());
}

void RasterGBufferPass::CreateGBufferTargets(
    ID3D12Device *device,
    UINT width,
    UINT height,
    bool shareNormalWithCuda,
    bool shareDepthWithCuda,
    bool shareAlbedoWithCuda)
{
    //
    // RTV Heap
    //

    D3D12_DESCRIPTOR_HEAP_DESC
    heapDesc = {};

    heapDesc.Type =
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV;

    heapDesc.NumDescriptors =
        GBufferTargetCount;

    heapDesc.Flags =
        D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    CheckHR(
        device->CreateDescriptorHeap(
            &heapDesc,
            IID_PPV_ARGS(
                &mGBufferRtvHeap)));

    mGBufferRtvDescriptorSize =
        device
            ->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_HEAP_PROPERTIES
    heapProperties = {};

    heapProperties.Type =
        D3D12_HEAP_TYPE_DEFAULT;

    heapProperties.CreationNodeMask = 1;
    heapProperties.VisibleNodeMask = 1;

    auto createRenderTarget =
        [&](DXGI_FORMAT format,
            const float clearColor[4],
            ComPtr<ID3D12Resource> &resource,
            UINT rtvIndex,
            D3D12_HEAP_FLAGS sharedFlags = D3D12_HEAP_FLAG_NONE)
    {
        D3D12_RESOURCE_DESC desc = {};

        desc.Dimension =
            D3D12_RESOURCE_DIMENSION_TEXTURE2D;

        desc.Width =
            width;

        desc.Height =
            height;

        desc.DepthOrArraySize =
            1;

        desc.MipLevels =
            1;

        desc.Format =
            format;

        desc.SampleDesc.Count =
            1;

        desc.SampleDesc.Quality =
            0;

        desc.Layout =
            D3D12_TEXTURE_LAYOUT_UNKNOWN;

        desc.Flags =
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

        D3D12_CLEAR_VALUE
        clearValue = {};

        clearValue.Format =
            format;

        clearValue.Color[0] =
            clearColor[0];

        clearValue.Color[1] =
            clearColor[1];

        clearValue.Color[2] =
            clearColor[2];

        clearValue.Color[3] =
            clearColor[3];

        CheckHR(
            device->CreateCommittedResource(
                &heapProperties,
                sharedFlags,
                &desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                &clearValue,
                IID_PPV_ARGS(
                    &resource)));

        D3D12_CPU_DESCRIPTOR_HANDLE
        rtvHandle =
            GetGBufferRTV(
                rtvIndex);

        device->CreateRenderTargetView(
            resource.Get(),
            nullptr,
            rtvHandle);
    };

    //
    // Albedo
    //

    const float albedoClear[4] =
        {
            0.0f,
            0.0f,
            0.0f,
            0.0f};

    createRenderTarget(
        AlbedoFormat,
        albedoClear,
        mAlbedoBuffer,
        0,
        shareAlbedoWithCuda ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE);

    //
    // Normal
    //
    // background normal = 0
    //

    const float normalClear[4] =
        {
            0.0f,
            0.0f,
            0.0f,
            0.0f};

    createRenderTarget(
        NormalFormat,
        normalClear,
        mNormalBuffer,
        1,
        shareNormalWithCuda ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE);

    //
    // D3D Depth
    //
    // background = 1.0
    //

    const float depthClear[4] =
        {
            1.0f,
            0.0f,
            0.0f,
            0.0f};

    createRenderTarget(
        RasterDepthFormat,
        depthClear,
        mRasterDepthBuffer,
        2,
        shareDepthWithCuda ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE);
}

D3D12_CPU_DESCRIPTOR_HANDLE
RasterGBufferPass::GetGBufferRTV(
    UINT index) const
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle =
        mGBufferRtvHeap
            ->GetCPUDescriptorHandleForHeapStart();

    handle.ptr +=
        static_cast<SIZE_T>(index) *
        mGBufferRtvDescriptorSize;

    return handle;
}

void RasterGBufferPass::CreateConstantBuffers(
    ID3D12Device *device,
    UINT frameCount)
{
    mObjectConstants.resize(
        frameCount);

    mPassConstants.resize(
        frameCount);

    //
    // Object CB
    //

    mObjectConstantStride =
        AlignConstantBufferSize(
            sizeof(ObjectConstants));

    const UINT objectBufferSize =
        mObjectConstantStride *
        MaxObjects;

    D3D12_HEAP_PROPERTIES
    uploadHeap = {};

    uploadHeap.Type =
        D3D12_HEAP_TYPE_UPLOAD;

    uploadHeap.CreationNodeMask = 1;
    uploadHeap.VisibleNodeMask = 1;

    D3D12_RESOURCE_DESC
    objectBufferDesc = {};

    objectBufferDesc.Dimension =
        D3D12_RESOURCE_DIMENSION_BUFFER;

    objectBufferDesc.Width =
        objectBufferSize;

    objectBufferDesc.Height = 1;
    objectBufferDesc.DepthOrArraySize = 1;
    objectBufferDesc.MipLevels = 1;

    objectBufferDesc.SampleDesc.Count = 1;

    objectBufferDesc.Layout =
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    for (auto &frame :
         mObjectConstants)
    {
        CheckHR(
            device->CreateCommittedResource(
                &uploadHeap,
                D3D12_HEAP_FLAG_NONE,
                &objectBufferDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(
                    &frame.Resource)));

        D3D12_RANGE readRange =
            {
                0,
                0};

        CheckHR(
            frame.Resource->Map(
                0,
                &readRange,

                reinterpret_cast<void **>(
                    &frame.MappedData)));
    }

    //
    // Pass / Light CB
    //

    const UINT passBufferSize =
        AlignConstantBufferSize(
            sizeof(PassConstants));

    D3D12_RESOURCE_DESC
    passBufferDesc = {};

    passBufferDesc.Dimension =
        D3D12_RESOURCE_DIMENSION_BUFFER;

    passBufferDesc.Width =
        passBufferSize;

    passBufferDesc.Height = 1;
    passBufferDesc.DepthOrArraySize = 1;
    passBufferDesc.MipLevels = 1;

    passBufferDesc.SampleDesc.Count = 1;

    passBufferDesc.Layout =
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    for (auto &frame :
         mPassConstants)
    {
        CheckHR(
            device->CreateCommittedResource(
                &uploadHeap,
                D3D12_HEAP_FLAG_NONE,
                &passBufferDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS(
                    &frame.Resource)));

        D3D12_RANGE readRange =
            {
                0,
                0};

        CheckHR(
            frame.Resource->Map(
                0,
                &readRange,

                reinterpret_cast<void **>(
                    &frame.MappedData)));
    }
}

void RasterGBufferPass::Render(
    ID3D12GraphicsCommandList *commandList,
    D3D12_CPU_DESCRIPTOR_HANDLE backBufferRTV,
    UINT frameIndex,
    const Camera &camera,
    const Scene &scene)
{
    //
    // Clear BackBuffer
    //

    const float backBufferClear[4] =
        {
            0.08f,
            0.12f,
            0.18f,
            1.0f};

    commandList->ClearRenderTargetView(
        backBufferRTV,
        backBufferClear,
        0,
        nullptr);

    //
    // Clear G-buffer
    //

    const float albedoClear[4] =
        {
            0.0f,
            0.0f,
            0.0f,
            0.0f};

    const float normalClear[4] =
        {
            0.0f,
            0.0f,
            0.0f,
            0.0f};

    const float rasterDepthClear[4] =
        {
            1.0f,
            0.0f,
            0.0f,
            0.0f};

    D3D12_CPU_DESCRIPTOR_HANDLE
    albedoRTV =
        GetGBufferRTV(0);

    D3D12_CPU_DESCRIPTOR_HANDLE
    normalRTV =
        GetGBufferRTV(1);

    D3D12_CPU_DESCRIPTOR_HANDLE
    rasterDepthRTV =
        GetGBufferRTV(2);

    commandList->ClearRenderTargetView(
        albedoRTV,
        albedoClear,
        0,
        nullptr);

    commandList->ClearRenderTargetView(
        normalRTV,
        normalClear,
        0,
        nullptr);

    commandList->ClearRenderTargetView(
        rasterDepthRTV,
        rasterDepthClear,
        0,
        nullptr);

    //
    // Clear Depth Test Buffer
    //

    D3D12_CPU_DESCRIPTOR_HANDLE dsv =
        mDsvHeap
            ->GetCPUDescriptorHandleForHeapStart();

    commandList->ClearDepthStencilView(
        dsv,
        D3D12_CLEAR_FLAG_DEPTH,
        1.0f,
        0,
        0,
        nullptr);

    //
    // MRT
    //

    D3D12_CPU_DESCRIPTOR_HANDLE
    renderTargets[4] =
        {
            backBufferRTV,
            albedoRTV,
            normalRTV,
            rasterDepthRTV};

    //
    // Pipeline
    //

    commandList->RSSetViewports(
        1,
        &mViewport);

    commandList->RSSetScissorRects(
        1,
        &mScissorRect);

    commandList->OMSetRenderTargets(
        4,
        renderTargets,
        FALSE,
        &dsv);

    commandList->SetGraphicsRootSignature(
        mRootSignature.Get());

    auto *activePipeline = mPipelineState.Get();
    commandList->SetPipelineState(activePipeline);

    commandList->IASetPrimitiveTopology(
        D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    //
    // Camera
    //

    XMMATRIX view =
        camera.GetView();

    XMMATRIX projection =
        camera.GetProjection();

    //
    // Light -> b1
    //

    const DirectionalLight &light =
        scene.GetDirectionalLight();

    PassConstants passConstants = {};

    passConstants.LightDirectionIntensity =
        {
            light.Direction.x,
            light.Direction.y,
            light.Direction.z,
            light.Intensity};

    passConstants.LightColorUnused =
        {
            light.Color.x,
            light.Color.y,
            light.Color.z,
            0.0f};

    std::memcpy(
        mPassConstants[frameIndex]
            .MappedData,

        &passConstants,

        sizeof(passConstants));

    commandList
        ->SetGraphicsRootConstantBufferView(
            1,

            mPassConstants[frameIndex]
                .Resource
                ->GetGPUVirtualAddress());

    //
    // Scene
    //

    const auto &objects =
        scene.GetObjects();

    if (objects.size() > MaxObjects)
    {
        throw std::runtime_error(
            "Scene contains too many objects.");
    }

    for (std::size_t i = 0;
         i < objects.size();
         ++i)
    {
        const SceneObject &object =
            objects[i];

        const Mesh *mesh =
            object.MeshData;

        if (!mesh)
        {
            continue;
        }
        auto *pipeline = object.MaterialData && object.MaterialData->TwoSided ?
            mTwoSidedPipelineState.Get() : mPipelineState.Get();
        if (pipeline != activePipeline)
        {
            commandList->SetPipelineState(pipeline);
            activePipeline = pipeline;
        }

        //
        // Transform
        //

        XMMATRIX world =
            object.ObjectTransform
                .GetWorldMatrix();

        XMMATRIX worldViewProjection =
            world *
            view *
            projection;

        XMMATRIX worldInverseTranspose =
            XMMatrixTranspose(
                XMMatrixInverse(
                    nullptr,
                    world));

        //
        // Object constants
        //

        ObjectConstants constants = {};

        XMStoreFloat4x4(
            &constants.WorldViewProjection,

            XMMatrixTranspose(
                worldViewProjection));

        XMStoreFloat4x4(
            &constants.WorldInverseTranspose,

            XMMatrixTranspose(
                worldInverseTranspose));

        //
        // Material
        //

        if (object.MaterialData)
        {
            const XMFLOAT3 &albedo =
                object.MaterialData->Albedo;

            constants.Albedo =
                {
                    albedo.x,
                    albedo.y,
                    albedo.z,
                    (255.0f-static_cast<float>(object.MaterialData->EmitterLight))/255.0f};
        }
        else
        {
            constants.Albedo =
                {
                    1.0f,
                    1.0f,
                    1.0f,
                    1.0f};
        }

        //
        // Object CB
        //

        const UINT objectIndex =
            static_cast<UINT>(
                i);

        std::uint8_t *constantAddress =
            mObjectConstants[frameIndex]
                .MappedData +
            objectIndex *
                mObjectConstantStride;

        std::memcpy(
            constantAddress,
            &constants,
            sizeof(constants));

        D3D12_GPU_VIRTUAL_ADDRESS
        gpuConstantAddress =
            mObjectConstants[frameIndex]
                .Resource
                ->GetGPUVirtualAddress() +
            objectIndex *
                mObjectConstantStride;

        commandList
            ->SetGraphicsRootConstantBufferView(
                0,
                gpuConstantAddress);

        //
        // Mesh
        //

        const auto &vertexView =
            mesh->GetVertexBufferView();

        const auto &indexView =
            mesh->GetIndexBufferView();

        commandList->IASetVertexBuffers(
            0,
            1,
            &vertexView);

        commandList->IASetIndexBuffer(
            &indexView);

        //
        // Draw
        //

        commandList->DrawIndexedInstanced(
            mesh->GetIndexCount(),
            1,
            0,
            0,
            0);
    }
}
