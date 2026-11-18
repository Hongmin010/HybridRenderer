#include "DebugViewPass.h"

#include <Windows.h>
#include <d3dcompiler.h>

#include <climits>
#include <filesystem>
#include <stdexcept>
#include <string>

using Microsoft::WRL::ComPtr;

namespace
{
    void CheckHR(HRESULT hr)
    {
        if (FAILED(hr))
        {
            throw std::runtime_error(
                "DebugViewPass D3D12 call failed.");
        }
    }

    ComPtr<ID3DBlob> CompileShader(
        const char *entryPoint,
        const char *target)
    {
        wchar_t executablePath[MAX_PATH] = {};

        GetModuleFileNameW(
            nullptr,
            executablePath,
            MAX_PATH);

        std::filesystem::path shaderPath =
            std::filesystem::path(executablePath)
                .parent_path() /
            L"Shaders" / L"DebugView.hlsl";

        ComPtr<ID3DBlob> shader;
        ComPtr<ID3DBlob> errors;

        HRESULT hr = D3DCompileFromFile(
            shaderPath.c_str(),
            nullptr,
            D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entryPoint,
            target,
            D3DCOMPILE_ENABLE_STRICTNESS,
            0,
            &shader,
            &errors);

        if (FAILED(hr))
        {
            if (errors)
            {
                throw std::runtime_error(
                    std::string(
                        static_cast<const char *>(
                            errors->GetBufferPointer()),
                        errors->GetBufferSize()));
            }

            CheckHR(hr);
        }

        return shader;
    }
}

void DebugViewPass::Initialize(
    ID3D12Device *device,
    DXGI_FORMAT backBufferFormat,
    ID3D12Resource *albedo,
    ID3D12Resource *normal,
    ID3D12Resource *depth)
{
    mAlbedo = albedo;
    mNormal = normal;
    mDepth = depth;

    if (!mAlbedo || !mNormal || !mDepth)
    {
        throw std::runtime_error(
            "G-buffer resources are missing.");
    }

    mWidth =
        static_cast<UINT>(mAlbedo->GetDesc().Width);

    mHeight =
        mAlbedo->GetDesc().Height;

    //
    // 1. SRV Descriptor Heap
    //

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};

    heapDesc.Type =
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

    heapDesc.NumDescriptors = 3;

    heapDesc.Flags =
        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

    CheckHR(
        device->CreateDescriptorHeap(
            &heapDesc,
            IID_PPV_ARGS(&mSrvHeap)));

    ID3D12Resource *resources[3] =
        {
            mAlbedo,
            mNormal,
            mDepth};

    DXGI_FORMAT formats[3] =
        {
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            DXGI_FORMAT_R32_FLOAT};

    UINT descriptorSize =
        device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_CPU_DESCRIPTOR_HANDLE handle =
        mSrvHeap->GetCPUDescriptorHandleForHeapStart();

    for (UINT i = 0; i < 3; ++i)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};

        srvDesc.Format = formats[i];

        srvDesc.ViewDimension =
            D3D12_SRV_DIMENSION_TEXTURE2D;

        srvDesc.Shader4ComponentMapping =
            D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

        srvDesc.Texture2D.MipLevels = 1;

        device->CreateShaderResourceView(
            resources[i],
            &srvDesc,
            handle);

        handle.ptr += descriptorSize;
    }

    //
    // 2. Root Signature
    //
    // t0 ~ t2 : G-buffer SRVs
    // b0      : Debug Mode (Root Constant)
    //

    D3D12_DESCRIPTOR_RANGE srvRange = {};

    srvRange.RangeType =
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV;

    srvRange.NumDescriptors = 3;
    srvRange.BaseShaderRegister = 0;
    srvRange.RegisterSpace = 0;

    srvRange.OffsetInDescriptorsFromTableStart =
        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER rootParameters[2] = {};

    rootParameters[0].ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;

    rootParameters[0].DescriptorTable.NumDescriptorRanges =
        1;

    rootParameters[0].DescriptorTable.pDescriptorRanges =
        &srvRange;

    rootParameters[0].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_PIXEL;

    rootParameters[1].ParameterType =
        D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;

    rootParameters[1].Constants.ShaderRegister = 0;
    rootParameters[1].Constants.RegisterSpace = 0;
    rootParameters[1].Constants.Num32BitValues = 1;

    rootParameters[1].ShaderVisibility =
        D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rootDesc = {};

    rootDesc.NumParameters = 2;
    rootDesc.pParameters = rootParameters;

    rootDesc.Flags =
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ComPtr<ID3DBlob> serializedRoot;
    ComPtr<ID3DBlob> rootErrors;

    HRESULT rootResult = D3D12SerializeRootSignature(
        &rootDesc,
        D3D_ROOT_SIGNATURE_VERSION_1,
        &serializedRoot,
        &rootErrors);

    if (FAILED(rootResult))
    {
        if (rootErrors)
        {
            throw std::runtime_error(
                std::string(
                    static_cast<const char *>(
                        rootErrors->GetBufferPointer()),
                    rootErrors->GetBufferSize()));
        }

        CheckHR(rootResult);
    }

    CheckHR(
        device->CreateRootSignature(
            0,
            serializedRoot->GetBufferPointer(),
            serializedRoot->GetBufferSize(),
            IID_PPV_ARGS(&mRootSignature)));

    //
    // 3. Fullscreen Pipeline
    //

    auto vs = CompileShader(
        "VSMain",
        "vs_5_1");

    auto ps = CompileShader(
        "PSMain",
        "ps_5_1");

    D3D12_RASTERIZER_DESC rasterizer = {};

    rasterizer.FillMode =
        D3D12_FILL_MODE_SOLID;

    rasterizer.CullMode =
        D3D12_CULL_MODE_NONE;

    rasterizer.DepthClipEnable = TRUE;

    D3D12_BLEND_DESC blend = {};

    auto &rtBlend = blend.RenderTarget[0];

    rtBlend.BlendEnable = FALSE;
    rtBlend.LogicOpEnable = FALSE;

    rtBlend.SrcBlend = D3D12_BLEND_ONE;
    rtBlend.DestBlend = D3D12_BLEND_ZERO;
    rtBlend.BlendOp = D3D12_BLEND_OP_ADD;

    rtBlend.SrcBlendAlpha = D3D12_BLEND_ONE;
    rtBlend.DestBlendAlpha = D3D12_BLEND_ZERO;
    rtBlend.BlendOpAlpha = D3D12_BLEND_OP_ADD;

    rtBlend.LogicOp = D3D12_LOGIC_OP_NOOP;

    rtBlend.RenderTargetWriteMask =
        D3D12_COLOR_WRITE_ENABLE_ALL;

    D3D12_DEPTH_STENCIL_DESC depthStencil = {};

    depthStencil.DepthEnable = FALSE;

    depthStencil.DepthWriteMask =
        D3D12_DEPTH_WRITE_MASK_ZERO;

    depthStencil.DepthFunc =
        D3D12_COMPARISON_FUNC_ALWAYS;

    depthStencil.StencilEnable = FALSE;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};

    psoDesc.pRootSignature =
        mRootSignature.Get();

    psoDesc.VS =
        {
            vs->GetBufferPointer(),
            vs->GetBufferSize()};

    psoDesc.PS =
        {
            ps->GetBufferPointer(),
            ps->GetBufferSize()};

    psoDesc.RasterizerState = rasterizer;
    psoDesc.BlendState = blend;
    psoDesc.DepthStencilState = depthStencil;

    psoDesc.SampleMask = UINT_MAX;

    // SV_VertexID를 사용하므로 Vertex Buffer 불필요
    psoDesc.InputLayout = {nullptr, 0};

    psoDesc.PrimitiveTopologyType =
        D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;

    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = backBufferFormat;

    psoDesc.DSVFormat =
        DXGI_FORMAT_UNKNOWN;

    psoDesc.SampleDesc.Count = 1;

    CheckHR(
        device->CreateGraphicsPipelineState(
            &psoDesc,
            IID_PPV_ARGS(&mPipelineState)));
}

void DebugViewPass::Render(
    ID3D12GraphicsCommandList *commandList,
    D3D12_CPU_DESCRIPTOR_HANDLE backBufferRTV,
    UINT mode)
{
    ID3D12Resource *resources[3] =
        {
            mAlbedo,
            mNormal,
            mDepth};

    //
    // 1. 이전 Raster Pass의 MRT 바인딩 해제
    //    BackBuffer만 출력 대상으로 설정
    //

    commandList->OMSetRenderTargets(
        1,
        &backBufferRTV,
        FALSE,
        nullptr);

    //
    // 2. G-buffer:
    //    RENDER_TARGET -> PIXEL_SHADER_RESOURCE
    //

    D3D12_RESOURCE_BARRIER barriers[3] = {};

    for (UINT i = 0; i < 3; ++i)
    {
        barriers[i].Type =
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;

        barriers[i].Transition.pResource =
            resources[i];

        barriers[i].Transition.Subresource =
            D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        barriers[i].Transition.StateBefore =
            D3D12_RESOURCE_STATE_RENDER_TARGET;

        barriers[i].Transition.StateAfter =
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    }

    commandList->ResourceBarrier(
        3,
        barriers);

    //
    // 3. Fullscreen viewport
    //

    D3D12_VIEWPORT viewport = {};

    viewport.Width =
        static_cast<float>(mWidth);

    viewport.Height =
        static_cast<float>(mHeight);

    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;

    D3D12_RECT scissor =
        {
            0,
            0,
            static_cast<LONG>(mWidth),
            static_cast<LONG>(mHeight)};

    commandList->RSSetViewports(
        1,
        &viewport);

    commandList->RSSetScissorRects(
        1,
        &scissor);

    commandList->SetGraphicsRootSignature(
        mRootSignature.Get());

    commandList->SetPipelineState(
        mPipelineState.Get());

    ID3D12DescriptorHeap *heaps[] =
        {
            mSrvHeap.Get()};

    commandList->SetDescriptorHeaps(
        1,
        heaps);

    commandList->SetGraphicsRootDescriptorTable(
        0,
        mSrvHeap->GetGPUDescriptorHandleForHeapStart());

    commandList->SetGraphicsRoot32BitConstant(
        1,
        mode,
        0);

    commandList->IASetPrimitiveTopology(
        D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    //
    // 4. Fullscreen Triangle
    //

    commandList->DrawInstanced(
        3,
        1,
        0,
        0);

    //
    // 5. 다음 프레임 Raster Pass를 위해
    //    G-buffer를 다시 RTV 상태로 복원
    //

    for (UINT i = 0; i < 3; ++i)
    {
        barriers[i].Transition.StateBefore =
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

        barriers[i].Transition.StateAfter =
            D3D12_RESOURCE_STATE_RENDER_TARGET;
    }

    commandList->ResourceBarrier(
        3,
        barriers);
}