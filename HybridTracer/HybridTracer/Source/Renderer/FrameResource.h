#pragma once

#include <d3d12.h>
#include <wrl.h>

struct FrameResource
{
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> CommandAllocator;
    UINT64 FenceValue = 0;
};