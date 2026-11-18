#include "ToneMapPass.h"
#include "Interop/InteropCommon.h"
#include <cmath>
using namespace HybridInterop;

void ToneMapPass::Initialize(ID3D12Device *device, ID3D12Resource *hdr, DXGI_FORMAT format)
{
    mHDR = hdr;
    mWidth = static_cast<UINT>(hdr->GetDesc().Width);
    mHeight = hdr->GetDesc().Height;
    D3D12_DESCRIPTOR_HEAP_DESC heap = {};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = 1;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CheckHR(device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&mInput)), "CreateDescriptorHeap(tonemap)");
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Format = hdr->GetDesc().Format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(hdr,&srv,mInput->GetCPUDescriptorHandleForHeapStart());
    mPipeline.Initialize(device,L"ToneMap.hlsl",format,1,2);
}

void ToneMapPass::Render(ID3D12GraphicsCommandList *list, D3D12_CPU_DESCRIPTOR_HANDLE output, float exposure, bool rawVisibility)
{
    if (!std::isfinite(exposure) || exposure <= 0) throw std::runtime_error("Exposure must be finite and positive.");
    // Unbind HDR before using its SRV; apply display conversion exactly once to UNORM RTV.
    list->OMSetRenderTargets(1,&output,FALSE,nullptr);
    Transition(list,mHDR,D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    struct Constants { float Exposure; UINT Raw; } data = {exposure,rawVisibility ? 1u : 0u};
    mPipeline.Draw(list,output,mInput.Get(),mWidth,mHeight,&data);
    Transition(list,mHDR,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_RENDER_TARGET);
}
