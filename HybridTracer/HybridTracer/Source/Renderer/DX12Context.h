#pragma once

#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include <array>

#include "FrameResource.h"

class DX12Context
{
public:
    static constexpr UINT FrameCount = 2;

public:
    DX12Context() = default;
    ~DX12Context();

    void Initialize(HWND hwnd, UINT width, UINT height);
    void InitializeHeadless();

    ID3D12GraphicsCommandList *BeginFrame();
    // Submit raster work, then continue recording without resetting the in-use allocator.
    ID3D12GraphicsCommandList *SubmitAndContinue();
    void EndFrame();
    void Flush();

    ID3D12Device *GetDevice() const
    {
        return mDevice.Get();
    }
    ID3D12CommandQueue *GetCommandQueue() const
    {
        return mCommandQueue.Get();
    }
    UINT GetCurrentFrameIndex() const
    {
        return mCurrentFrameIndex;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE GetCurrentRTV() const;

    DXGI_FORMAT GetBackBufferFormat() const
    {
        return DXGI_FORMAT_R8G8B8A8_UNORM;
    }

private:
    void CreateDevice();
    void CreateCommandObjects();
    void CreateSwapChain(HWND hwnd, UINT width, UINT height);
    void CreateRenderTargets();

    void WaitForFrame(FrameResource &frame);
    void WaitForGPU();

private:
    // DXGI
    Microsoft::WRL::ComPtr<IDXGIFactory6> mFactory;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> mAdapter;

    // D3D12 core
    Microsoft::WRL::ComPtr<ID3D12Device> mDevice;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> mCommandQueue;

    // Swap chain
    Microsoft::WRL::ComPtr<IDXGISwapChain3> mSwapChain;

    // Back buffers
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mRtvHeap;

    Microsoft::WRL::ComPtr<ID3D12Resource>
        mRenderTargets[FrameCount];

    UINT mRtvDescriptorSize = 0;
    UINT mCurrentFrameIndex = 0;

    // Command recording
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList>
        mCommandList;

    // Per-frame resources
    std::array<FrameResource, FrameCount> mFrames;

    // GPU synchronization
    Microsoft::WRL::ComPtr<ID3D12Fence> mFence;

    UINT64 mFenceValue = 0;
    HANDLE mFenceEvent = nullptr;
};
