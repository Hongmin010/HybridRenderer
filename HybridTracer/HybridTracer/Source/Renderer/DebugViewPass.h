#pragma once

#include <d3d12.h>
#include <dxgiformat.h>
#include <wrl.h>

class DebugViewPass
{
public:
    void Initialize(
        ID3D12Device *device,
        DXGI_FORMAT backBufferFormat,
        ID3D12Resource *albedo,
        ID3D12Resource *normal,
        ID3D12Resource *depth);

    void Render(
        ID3D12GraphicsCommandList *commandList,
        D3D12_CPU_DESCRIPTOR_HANDLE backBufferRTV,
        UINT mode);

private:
    Microsoft::WRL::ComPtr<ID3D12RootSignature>
        mRootSignature;

    Microsoft::WRL::ComPtr<ID3D12PipelineState>
        mPipelineState;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>
        mSrvHeap;

    // RasterGBufferPass가 소유하는 리소스를 참조
    ID3D12Resource *mAlbedo = nullptr;
    ID3D12Resource *mNormal = nullptr;
    ID3D12Resource *mDepth = nullptr;

    UINT mWidth = 0;
    UINT mHeight = 0;
};