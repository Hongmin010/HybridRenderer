#pragma once
#include "FullscreenPipeline.h"
#include "Scene/Light.h"
#include <array>

class CompositePass
{
public:
    void Initialize(ID3D12Device *device, ID3D12Resource *albedo, ID3D12Resource *normal,
        ID3D12Resource *depth, ID3D12Resource *shadow, ID3D12Resource *indirect, UINT lightingPlanes);
    // Inputs: G-buffer in RENDER_TARGET, shadow/indirect in COMMON. Restores on return.
    // Output: owned linear RGBA16F scene color, left in RENDER_TARGET.
    void Render(ID3D12GraphicsCommandList *list, const DirectionalLight &light, bool useShadows, bool showVisibility, UINT lightingView = 0);
    ID3D12Resource *GetOutput() const { return mHDR.Get(); }
private:
    FullscreenPipeline mPipeline;
    Microsoft::WRL::ComPtr<ID3D12Resource> mHDR;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mRTV, mInputs;
    std::array<ID3D12Resource *,5> mSources = {};
    UINT mWidth = 0, mHeight = 0;
    UINT mLightingPlanes = 1;
};
