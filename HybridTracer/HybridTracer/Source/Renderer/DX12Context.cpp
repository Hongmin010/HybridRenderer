#include "DX12Context.h"

#include <stdexcept>

using Microsoft::WRL::ComPtr;

namespace
{
    void CheckHR(HRESULT hr)
    {
        if (FAILED(hr))
        {
            throw std::runtime_error("DirectX 12 call failed.");
        }
    }
}

DX12Context::~DX12Context()
{
    if (mCommandQueue && mFence && mFenceEvent)
    {
        WaitForGPU();
    }

    if (mFenceEvent)
    {
        CloseHandle(mFenceEvent);
        mFenceEvent = nullptr;
    }
}

void DX12Context::Initialize(
    HWND hwnd,
    UINT width,
    UINT height)
{
    InitializeHeadless();
    CreateSwapChain(hwnd, width, height);
    CreateRenderTargets();
}

void DX12Context::InitializeHeadless()
{
    CreateDevice();
    CreateCommandObjects();
}

void DX12Context::CreateDevice()
{
    CheckHR(CreateDXGIFactory2(
        0,
        IID_PPV_ARGS(&mFactory)));

    for (UINT adapterIndex = 0;; ++adapterIndex)
    {
        ComPtr<IDXGIAdapter1> adapter;

        HRESULT hr =
            mFactory->EnumAdapterByGpuPreference(
                adapterIndex,
                DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                IID_PPV_ARGS(&adapter));

        if (hr == DXGI_ERROR_NOT_FOUND)
        {
            break;
        }

        CheckHR(hr);

        DXGI_ADAPTER_DESC1 desc = {};
        CheckHR(adapter->GetDesc1(&desc));

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
        {
            continue;
        }

        if (SUCCEEDED(D3D12CreateDevice(
                adapter.Get(),
                D3D_FEATURE_LEVEL_12_0,
                __uuidof(ID3D12Device),
                nullptr)))
        {
            mAdapter = adapter;
            break;
        }
    }

    if (!mAdapter)
    {
        throw std::runtime_error(
            "No suitable DX12 GPU found.");
    }

    CheckHR(D3D12CreateDevice(
        mAdapter.Get(),
        D3D_FEATURE_LEVEL_12_0,
        IID_PPV_ARGS(&mDevice)));
}

void DX12Context::CreateCommandObjects()
{
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Priority =
        D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queueDesc.Flags =
        D3D12_COMMAND_QUEUE_FLAG_NONE;

    CheckHR(mDevice->CreateCommandQueue(
        &queueDesc,
        IID_PPV_ARGS(&mCommandQueue)));

    for (auto &frame : mFrames)
    {
        CheckHR(mDevice->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&frame.CommandAllocator)));
    }

    CheckHR(mDevice->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        mFrames[0].CommandAllocator.Get(),
        nullptr,
        IID_PPV_ARGS(&mCommandList)));

    CheckHR(mCommandList->Close());

    CheckHR(mDevice->CreateFence(
        0,
        D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&mFence)));

    mFenceEvent = CreateEventW(
        nullptr,
        FALSE,
        FALSE,
        nullptr);

    if (!mFenceEvent)
    {
        throw std::runtime_error(
            "Failed to create fence event.");
    }
}

void DX12Context::CreateSwapChain(
    HWND hwnd,
    UINT width,
    UINT height)
{
    DXGI_SWAP_CHAIN_DESC1 swapDesc = {};

    swapDesc.Width = width;
    swapDesc.Height = height;
    swapDesc.Format =
        DXGI_FORMAT_R8G8B8A8_UNORM;

    swapDesc.SampleDesc.Count = 1;
    swapDesc.SampleDesc.Quality = 0;

    swapDesc.BufferUsage =
        DXGI_USAGE_RENDER_TARGET_OUTPUT;

    swapDesc.BufferCount = FrameCount;

    swapDesc.SwapEffect =
        DXGI_SWAP_EFFECT_FLIP_DISCARD;

    swapDesc.Scaling =
        DXGI_SCALING_STRETCH;

    swapDesc.AlphaMode =
        DXGI_ALPHA_MODE_UNSPECIFIED;

    ComPtr<IDXGISwapChain1> swapChain1;

    CheckHR(mFactory->CreateSwapChainForHwnd(
        mCommandQueue.Get(),
        hwnd,
        &swapDesc,
        nullptr,
        nullptr,
        &swapChain1));

    CheckHR(mFactory->MakeWindowAssociation(
        hwnd,
        DXGI_MWA_NO_ALT_ENTER));

    CheckHR(swapChain1.As(&mSwapChain));
}

void DX12Context::CreateRenderTargets()
{
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};

    heapDesc.Type =
        D3D12_DESCRIPTOR_HEAP_TYPE_RTV;

    heapDesc.NumDescriptors = FrameCount;
    heapDesc.Flags =
        D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    CheckHR(mDevice->CreateDescriptorHeap(
        &heapDesc,
        IID_PPV_ARGS(&mRtvHeap)));

    mRtvDescriptorSize =
        mDevice->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle =
        mRtvHeap->GetCPUDescriptorHandleForHeapStart();

    for (UINT i = 0; i < FrameCount; ++i)
    {
        CheckHR(mSwapChain->GetBuffer(
            i,
            IID_PPV_ARGS(&mRenderTargets[i])));

        mDevice->CreateRenderTargetView(
            mRenderTargets[i].Get(),
            nullptr,
            rtvHandle);

        rtvHandle.ptr += mRtvDescriptorSize;
    }
}

ID3D12GraphicsCommandList *
DX12Context::BeginFrame()
{
    mCurrentFrameIndex =
        mSwapChain->GetCurrentBackBufferIndex();

    FrameResource &frame =
        mFrames[mCurrentFrameIndex];

    WaitForFrame(frame);

    CheckHR(
        frame.CommandAllocator->Reset());

    CheckHR(mCommandList->Reset(
        frame.CommandAllocator.Get(),
        nullptr));

    D3D12_RESOURCE_BARRIER barrier = {};

    barrier.Type =
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;

    barrier.Transition.pResource =
        mRenderTargets[mCurrentFrameIndex].Get();

    barrier.Transition.StateBefore =
        D3D12_RESOURCE_STATE_PRESENT;

    barrier.Transition.StateAfter =
        D3D12_RESOURCE_STATE_RENDER_TARGET;

    barrier.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    mCommandList->ResourceBarrier(
        1,
        &barrier);

    return mCommandList.Get();
}

void DX12Context::EndFrame()
{
    D3D12_RESOURCE_BARRIER barrier = {};

    barrier.Type =
        D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;

    barrier.Transition.pResource =
        mRenderTargets[mCurrentFrameIndex].Get();

    barrier.Transition.StateBefore =
        D3D12_RESOURCE_STATE_RENDER_TARGET;

    barrier.Transition.StateAfter =
        D3D12_RESOURCE_STATE_PRESENT;

    barrier.Transition.Subresource =
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    mCommandList->ResourceBarrier(
        1,
        &barrier);

    CheckHR(mCommandList->Close());

    ID3D12CommandList *commandLists[] =
        {
            mCommandList.Get()};

    mCommandQueue->ExecuteCommandLists(
        1,
        commandLists);

    CheckHR(mSwapChain->Present(
        1,
        0));

    ++mFenceValue;

    CheckHR(mCommandQueue->Signal(
        mFence.Get(),
        mFenceValue));

    mFrames[mCurrentFrameIndex].FenceValue =
        mFenceValue;
}

ID3D12GraphicsCommandList *DX12Context::SubmitAndContinue()
{
    CheckHR(mCommandList->Close());
    ID3D12CommandList *lists[] = {mCommandList.Get()};
    mCommandQueue->ExecuteCommandLists(1,lists);
    // Command-list Reset is legal while prior execution runs. Allocator Reset is deferred
    // to BeginFrame, after its fence; never reset an allocator at this interop boundary.
    CheckHR(mCommandList->Reset(mFrames[mCurrentFrameIndex].CommandAllocator.Get(),nullptr));
    return mCommandList.Get();
}

void DX12Context::Flush()
{
    if (mCommandQueue && mFence && mFenceEvent) WaitForGPU();
}

D3D12_CPU_DESCRIPTOR_HANDLE
DX12Context::GetCurrentRTV() const
{
    D3D12_CPU_DESCRIPTOR_HANDLE handle =
        mRtvHeap->GetCPUDescriptorHandleForHeapStart();

    handle.ptr +=
        static_cast<SIZE_T>(mCurrentFrameIndex) *
        mRtvDescriptorSize;

    return handle;
}

void DX12Context::WaitForFrame(
    FrameResource &frame)
{
    if (frame.FenceValue == 0)
    {
        return;
    }

    if (mFence->GetCompletedValue() < frame.FenceValue)
    {
        CheckHR(mFence->SetEventOnCompletion(
            frame.FenceValue,
            mFenceEvent));

        WaitForSingleObject(
            mFenceEvent,
            INFINITE);
    }
}

void DX12Context::WaitForGPU()
{
    ++mFenceValue;

    CheckHR(mCommandQueue->Signal(
        mFence.Get(),
        mFenceValue));

    if (mFence->GetCompletedValue() < mFenceValue)
    {
        CheckHR(mFence->SetEventOnCompletion(
            mFenceValue,
            mFenceEvent));

        WaitForSingleObject(
            mFenceEvent,
            INFINITE);
    }
}
