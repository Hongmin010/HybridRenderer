#include "FullscreenPipeline.h"
#include "Interop/InteropCommon.h"
#include <filesystem>
#include <climits>
using namespace HybridInterop;

namespace
{
    ComPtr<ID3DBlob> Compile(const wchar_t *name, const char *entry, const char *target,
        const D3D_SHADER_MACRO *defines = nullptr)
    {
        wchar_t exe[MAX_PATH] = {};
        GetModuleFileNameW(nullptr,exe,MAX_PATH);
        const auto path = std::filesystem::path(exe).parent_path()/L"Shaders"/name;
        ComPtr<ID3DBlob> code, errors;
        const HRESULT result = D3DCompileFromFile(path.c_str(),defines,D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entry,target,D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_WARNINGS_ARE_ERRORS,0,&code,&errors);
        if (FAILED(result) && errors)
            throw std::runtime_error(std::string(static_cast<const char *>(errors->GetBufferPointer()),errors->GetBufferSize()));
        CheckHR(result,"Compile(fullscreen shader)");
        return code;
    }
}

void FullscreenPipeline::Initialize(ID3D12Device *device, const wchar_t *shader, DXGI_FORMAT format,
    UINT srvCount, UINT constantCount, const D3D_SHADER_MACRO *pixelDefines)
{
    mConstantCount = constantCount;
    D3D12_DESCRIPTOR_RANGE range = {};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = srvCount;
    D3D12_ROOT_PARAMETER parameters[2] = {};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[0].DescriptorTable = {1,&range};
    parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[1].Constants = {0,0,constantCount};
    parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC root = {};
    root.NumParameters = 2;
    root.pParameters = parameters;
    root.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ComPtr<ID3DBlob> serialized, errors;
    CheckHR(D3D12SerializeRootSignature(&root,D3D_ROOT_SIGNATURE_VERSION_1,&serialized,&errors),"D3D12SerializeRootSignature(fullscreen)");
    CheckHR(device->CreateRootSignature(0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(&mRoot)),"CreateRootSignature(fullscreen)");
    const auto vs = Compile(shader,"VSMain","vs_5_0"), ps = Compile(shader,"PSMain","ps_5_0",pixelDefines);
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso = {};
    pso.pRootSignature = mRoot.Get();
    pso.VS = {vs->GetBufferPointer(),vs->GetBufferSize()};
    pso.PS = {ps->GetBufferPointer(),ps->GetBufferSize()};
    pso.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    auto &blend = pso.BlendState.RenderTarget[0];
    blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
    blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.LogicOp = D3D12_LOGIC_OP_NOOP;
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.RasterizerState.DepthClipEnable = TRUE;
    pso.DepthStencilState.DepthEnable = FALSE;
    pso.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    pso.DepthStencilState.StencilEnable = FALSE;
    pso.SampleMask = UINT_MAX;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = format;
    pso.SampleDesc.Count = 1;
    CheckHR(device->CreateGraphicsPipelineState(&pso,IID_PPV_ARGS(&mPipeline)),"CreateGraphicsPipelineState(fullscreen)");
}

void FullscreenPipeline::Draw(ID3D12GraphicsCommandList *list, D3D12_CPU_DESCRIPTOR_HANDLE output,
    ID3D12DescriptorHeap *inputs, UINT width, UINT height, const void *constants)
{
    const D3D12_VIEWPORT viewport = {0,0,float(width),float(height),0,1};
    const D3D12_RECT scissor = {0,0,LONG(width),LONG(height)};
    list->OMSetRenderTargets(1,&output,FALSE,nullptr);
    list->RSSetViewports(1,&viewport);
    list->RSSetScissorRects(1,&scissor);
    list->SetGraphicsRootSignature(mRoot.Get());
    list->SetPipelineState(mPipeline.Get());
    ID3D12DescriptorHeap *heaps[] = {inputs};
    list->SetDescriptorHeaps(1,heaps);
    list->SetGraphicsRootDescriptorTable(0,inputs->GetGPUDescriptorHandleForHeapStart());
    list->SetGraphicsRoot32BitConstants(1,mConstantCount,constants,0);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list->DrawInstanced(3,1,0,0);
}
