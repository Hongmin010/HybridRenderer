#pragma once

#include <Windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <cuda.h>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace HybridInterop
{
    using Microsoft::WRL::ComPtr;

    inline void CheckHR(HRESULT result, const char *operation)
    {
        if (FAILED(result))
        {
            std::ostringstream message;
            message << operation << " failed: HRESULT 0x" << std::hex
                    << static_cast<unsigned long>(result);
            throw std::runtime_error(message.str());
        }
    }

    inline void CheckCuda(CUresult result, const char *operation)
    {
        if (result != CUDA_SUCCESS)
        {
            const char *name = nullptr;
            const char *description = nullptr;
            cuGetErrorName(result, &name);
            cuGetErrorString(result, &description);
            throw std::runtime_error(std::string(operation) + ": " +
                (name ? name : "CUDA error") + " - " +
                (description ? description : "unknown"));
        }
    }

    struct WinHandle
    {
        HANDLE Value = nullptr;
        ~WinHandle() { if (Value) CloseHandle(Value); }
        WinHandle() = default;
        WinHandle(const WinHandle &) = delete;
        WinHandle &operator=(const WinHandle &) = delete;
    };

    inline void PrintLuid(const char *label, const LUID &luid)
    {
        std::cout << label << std::hex << std::setfill('0') << std::setw(8)
                  << static_cast<uint32_t>(luid.HighPart) << ':'
                  << std::setw(8) << luid.LowPart << std::dec << '\n';
    }

    inline CUdevice FindCudaDevice(ID3D12Device *device)
    {
        if (device->GetNodeCount() != 1)
            throw std::runtime_error("The probe requires a single-node DX12 adapter.");

        const LUID dxLuid = device->GetAdapterLuid();
        PrintLuid("DX12 LUID: ", dxLuid);
        ComPtr<IDXGIFactory4> factory;
        CheckHR(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
        ComPtr<IDXGIAdapter1> adapter;
        CheckHR(factory->EnumAdapterByLuid(dxLuid, IID_PPV_ARGS(&adapter)), "EnumAdapterByLuid");
        DXGI_ADAPTER_DESC1 description = {};
        CheckHR(adapter->GetDesc1(&description), "GetDesc1");
        char adapterName[256] = {};
        WideCharToMultiByte(CP_UTF8, 0, description.Description, -1,
            adapterName, sizeof(adapterName), nullptr, nullptr);
        std::cout << "DX12 adapter: " << adapterName << '\n';

        CheckCuda(cuInit(0), "cuInit");
        int count = 0;
        CheckCuda(cuDeviceGetCount(&count), "cuDeviceGetCount");
        for (int index = 0; index < count; ++index)
        {
            CUdevice candidate;
            CheckCuda(cuDeviceGet(&candidate, index), "cuDeviceGet");
            char name[256] = {};
            CheckCuda(cuDeviceGetName(name, sizeof(name), candidate), "cuDeviceGetName");
            LUID cudaLuid = {};
            unsigned int nodeMask = 0;
            CheckCuda(cuDeviceGetLuid(reinterpret_cast<char *>(&cudaLuid),
                &nodeMask, candidate), "cuDeviceGetLuid");
            std::cout << "CUDA device " << index << ": " << name << '\n';
            PrintLuid("CUDA LUID: ", cudaLuid);
            if (std::memcmp(&dxLuid, &cudaLuid, sizeof(LUID)) == 0 && nodeMask == 1)
            {
                std::cout << "PASS: DX12 and CUDA refer to the same GPU.\n";
                return candidate;
            }
        }
        throw std::runtime_error("No CUDA device matches the renderer's DX12 adapter LUID.");
    }

    inline void Transition(ID3D12GraphicsCommandList *list, ID3D12Resource *resource,
        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        list->ResourceBarrier(1, &barrier);
    }

    inline CUexternalSemaphore ImportFence(ID3D12Device *device, ID3D12Fence *fence)
    {
        WinHandle handle;
        CheckHR(device->CreateSharedHandle(fence, nullptr, GENERIC_ALL, nullptr,
            &handle.Value), "CreateSharedHandle(fence)");
        CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC description = {};
        description.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
        description.handle.win32.handle = handle.Value;
        CUexternalSemaphore semaphore = nullptr;
        CheckCuda(cuImportExternalSemaphore(&semaphore, &description), "cuImportExternalSemaphore");
        return semaphore; // NT handle can be closed after successful import.
    }

}
