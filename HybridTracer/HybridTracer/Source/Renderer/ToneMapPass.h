#pragma once
#include "FullscreenPipeline.h"

class ToneMapPass
{
public:
    void Initialize(ID3D12Device *device, ID3D12Resource *hdr, DXGI_FORMAT backBufferFormat);
    // HDR enters/leaves RENDER_TARGET. Destination RTV is already in RENDER_TARGET.
    void Render(ID3D12GraphicsCommandList *list, D3D12_CPU_DESCRIPTOR_HANDLE output,
        float exposure = 1.0f, bool rawVisibility = false);
private:
    FullscreenPipeline mPipeline;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mInput;
    ID3D12Resource *mHDR = nullptr;
    UINT mWidth = 0, mHeight = 0;
};
