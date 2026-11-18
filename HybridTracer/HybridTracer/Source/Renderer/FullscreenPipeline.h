#pragma once
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl.h>

// Small shared DX12 draw machinery; CompositePass/ToneMapPass keep their own data contracts.
class FullscreenPipeline
{
public:
    void Initialize(ID3D12Device *device, const wchar_t *shader, DXGI_FORMAT outputFormat,
        UINT srvCount, UINT constantCount, const D3D_SHADER_MACRO *pixelDefines = nullptr);
    void Draw(ID3D12GraphicsCommandList *list, D3D12_CPU_DESCRIPTOR_HANDLE output,
        ID3D12DescriptorHeap *inputs, UINT width, UINT height, const void *constants);
private:
    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRoot;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPipeline;
    UINT mConstantCount = 0;
};
