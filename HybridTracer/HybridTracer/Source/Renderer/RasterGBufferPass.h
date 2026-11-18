#pragma once

#include <d3d12.h>
#include <dxgiformat.h>
#include <wrl.h>

#include <cstdint>
#include <vector>

class Camera;
class Scene;

class RasterGBufferPass
{
public:
    RasterGBufferPass() = default;
    ~RasterGBufferPass();

    void Initialize(
        ID3D12Device *device,
        DXGI_FORMAT renderTargetFormat,
        UINT width,
        UINT height,
        UINT frameCount,
        bool shareNormalWithCuda = false,
        bool shareDepthWithCuda = false,
        bool shareAlbedoWithCuda = false);

    void Render(
        ID3D12GraphicsCommandList *commandList,
        D3D12_CPU_DESCRIPTOR_HANDLE backBufferRTV,
        UINT frameIndex,
        const Camera &camera,
        const Scene &scene);

    ID3D12Resource *GetAlbedoBuffer() const
    {
        return mAlbedoBuffer.Get();
    }

    ID3D12Resource *GetNormalBuffer() const
    {
        return mNormalBuffer.Get();
    }

    ID3D12Resource *GetRasterDepthBuffer() const
    {
        return mRasterDepthBuffer.Get();
    }

private:
    struct ConstantBuffer
    {
        Microsoft::WRL::ComPtr<ID3D12Resource>
            Resource;

        std::uint8_t *MappedData = nullptr;
    };

private:
    static constexpr UINT MaxObjects = 64;

    static constexpr UINT GBufferTargetCount = 3;

    static constexpr DXGI_FORMAT AlbedoFormat =
        DXGI_FORMAT_R8G8B8A8_UNORM;

    static constexpr DXGI_FORMAT NormalFormat =
        DXGI_FORMAT_R16G16B16A16_FLOAT;

    static constexpr DXGI_FORMAT RasterDepthFormat =
        DXGI_FORMAT_R32_FLOAT;

private:
    void CreatePipeline(
        ID3D12Device *device,
        DXGI_FORMAT backBufferFormat);

    void CreateDepthBuffer(
        ID3D12Device *device,
        UINT width,
        UINT height);

    void CreateGBufferTargets(
        ID3D12Device *device,
        UINT width,
        UINT height,
        bool shareNormalWithCuda,
        bool shareDepthWithCuda,
        bool shareAlbedoWithCuda);

    void CreateConstantBuffers(
        ID3D12Device *device,
        UINT frameCount);

    D3D12_CPU_DESCRIPTOR_HANDLE
    GetGBufferRTV(UINT index) const;

private:
    Microsoft::WRL::ComPtr<ID3D12RootSignature>
        mRootSignature;

    Microsoft::WRL::ComPtr<ID3D12PipelineState>
        mPipelineState;

    //
    // Depth test용
    //

    Microsoft::WRL::ComPtr<ID3D12Resource>
        mDepthStencilBuffer;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>
        mDsvHeap;

    //
    // 실제 G-buffer
    //

    Microsoft::WRL::ComPtr<ID3D12Resource>
        mAlbedoBuffer;

    Microsoft::WRL::ComPtr<ID3D12Resource>
        mNormalBuffer;

    Microsoft::WRL::ComPtr<ID3D12Resource>
        mRasterDepthBuffer;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>
        mGBufferRtvHeap;

    UINT mGBufferRtvDescriptorSize = 0;

    //
    // Raster state
    //

    D3D12_VIEWPORT mViewport = {};
    D3D12_RECT mScissorRect = {};

    //
    // Constant buffers
    //

    std::vector<ConstantBuffer>
        mObjectConstants;

    std::vector<ConstantBuffer>
        mPassConstants;

    UINT mObjectConstantStride = 0;
};
