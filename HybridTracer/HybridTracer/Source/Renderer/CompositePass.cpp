#include "CompositePass.h"
#include "Interop/InteropCommon.h"
#include "../../Shaders/OptixTraceShared.h"
using namespace HybridInterop;

void CompositePass::Initialize(ID3D12Device *device, ID3D12Resource *albedo, ID3D12Resource *normal,
    ID3D12Resource *depth, ID3D12Resource *shadow, ID3D12Resource *indirect, UINT lightingPlanes)
{
    mSources = {albedo,normal,depth,shadow,indirect};
    mLightingPlanes = lightingPlanes;
    const auto description = albedo->GetDesc();
    mWidth = static_cast<UINT>(description.Width);
    mHeight = description.Height;
    D3D12_HEAP_PROPERTIES heap = {};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    auto hdr = description;
    hdr.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    hdr.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear = {};
    clear.Format = hdr.Format;
    CheckHR(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&hdr,D3D12_RESOURCE_STATE_RENDER_TARGET,
        &clear,IID_PPV_ARGS(&mHDR)), "CreateCommittedResource(HDR)");
    D3D12_DESCRIPTOR_HEAP_DESC descriptors = {};
    descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    descriptors.NumDescriptors = 1;
    CheckHR(device->CreateDescriptorHeap(&descriptors,IID_PPV_ARGS(&mRTV)), "CreateDescriptorHeap(HDR RTV)");
    device->CreateRenderTargetView(mHDR.Get(),nullptr,mRTV->GetCPUDescriptorHandleForHeapStart());
    descriptors.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptors.NumDescriptors = 5;
    descriptors.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    CheckHR(device->CreateDescriptorHeap(&descriptors,IID_PPV_ARGS(&mInputs)), "CreateDescriptorHeap(composite SRVs)");
    auto handle = mInputs->GetCPUDescriptorHandleForHeapStart();
    const auto stride = device->GetDescriptorHandleIncrementSize(descriptors.Type);
    for (UINT i = 0; i < 5; ++i)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if (i < 3)
        {
            srv.Format = mSources[i]->GetDesc().Format;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Texture2D.MipLevels = 1;
        }
        else
        {
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Buffer.NumElements = mWidth*mHeight*(i == 3 ? 1u : mLightingPlanes);
            srv.Buffer.StructureByteStride = i == 3 ? sizeof(ShadowResult) : sizeof(IndirectLighting);
        }
        device->CreateShaderResourceView(mSources[i],&srv,handle);
        handle.ptr += stride;
    }
    const bool hasAdditionalLights = mLightingPlanes == AdditionalLightingPlanes;
    const D3D_SHADER_MACRO defines[] = {{"ADDITIONAL_LIGHTS","1"},{nullptr,nullptr}};
    mPipeline.Initialize(device,L"Composite.hlsl",hdr.Format,5,hasAdditionalLights ? 13 : 12,
        hasAdditionalLights ? defines : nullptr);
}

void CompositePass::Render(ID3D12GraphicsCommandList *list, const DirectionalLight &light, bool useShadows, bool showVisibility, UINT lightingView)
{
    list->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
    for (size_t i = 0; i < mSources.size(); ++i)
        Transition(list,mSources[i],i < 3 ? D3D12_RESOURCE_STATE_RENDER_TARGET : D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    struct Constants
    {
        float DirectionIntensity[4], ColorUnused[4];
        UINT Width, UseShadows, ShowVisibility, LightingView, PixelCount;
    } data = {{light.Direction.x,light.Direction.y,light.Direction.z,light.Intensity},
        {light.Color.x,light.Color.y,light.Color.z,0},
        mWidth,useShadows ? 1u : 0u,showVisibility ? 1u : 0u,lightingView,mWidth*mHeight};
    static_assert(sizeof(Constants) == 52);
    mPipeline.Draw(list,mRTV->GetCPUDescriptorHandleForHeapStart(),mInputs.Get(),mWidth,mHeight,&data);
    for (size_t i = 0; i < mSources.size(); ++i)
        Transition(list,mSources[i],D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            i < 3 ? D3D12_RESOURCE_STATE_RENDER_TARGET : D3D12_RESOURCE_STATE_COMMON);
}
