#include "Renderer/DX12Context.h"
#include "InteropCommon.h"

#include <cuda.h>
#include <d3d12sdklayers.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

using Microsoft::WRL::ComPtr;

namespace
{
    using namespace HybridInterop;

    constexpr size_t HalfCount = 256;
    constexpr size_t WordCount = HalfCount * 2;
    constexpr size_t ByteCount = WordCount * sizeof(uint32_t);

    // Release CUDA mappings before the D3D12 resources they reference.
    struct CudaResources
    {
        CUcontext Context = nullptr;
        CUstream Stream = nullptr;
        CUexternalMemory Memory = nullptr;
        CUdeviceptr Buffer = 0;
        CUexternalSemaphore DxReady = nullptr;
        CUexternalSemaphore CudaReady = nullptr;
        uint32_t *Observed = nullptr;

        ~CudaResources()
        {
            if (Stream) cuStreamSynchronize(Stream);
            if (Observed) cuMemFreeHost(Observed);
            if (Buffer) cuMemFree(Buffer);
            if (Memory) cuDestroyExternalMemory(Memory);
            if (DxReady) cuDestroyExternalSemaphore(DxReady);
            if (CudaReady) cuDestroyExternalSemaphore(CudaReady);
            if (Stream) cuStreamDestroy(Stream);
            if (Context) cuCtxDestroy(Context);
        }
    };

    ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device *device, D3D12_HEAP_TYPE type,
        D3D12_HEAP_FLAGS flags, D3D12_RESOURCE_STATES state)
    {
        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = type;
        heap.CreationNodeMask = heap.VisibleNodeMask = 1;
        D3D12_RESOURCE_DESC description = {};
        description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        description.Width = ByteCount;
        description.Height = description.DepthOrArraySize = description.MipLevels = 1;
        description.SampleDesc.Count = 1;
        description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> resource;
        CheckHR(device->CreateCommittedResource(&heap, flags, &description,
            state, nullptr, IID_PPV_ARGS(&resource)), "CreateCommittedResource");
        return resource;
    }

    uint32_t InputValue(uint32_t iteration, size_t index)
    {
        return 0x12340000u ^ (iteration * 0x00010001u) ^ static_cast<uint32_t>(index * 17);
    }

    uint32_t OutputValue(uint32_t iteration)
    {
        return 0xca000000u | iteration;
    }

    void RequireEqual(uint32_t actual, uint32_t expected, uint32_t iteration,
        size_t index, const char *path)
    {
        if (actual == expected) return;
        std::ostringstream message;
        message << path << " mismatch, iteration " << iteration << ", word " << index
                << ": expected 0x" << std::hex << expected << ", actual 0x" << actual;
        throw std::runtime_error(message.str());
    }

    void RunProbe(DX12Context &dx12, uint32_t iterations)
    {
        auto *device = dx12.GetDevice();
        auto *queue = dx12.GetCommandQueue();
        const CUdevice cudaDevice = FindCudaDevice(device);

        auto shared = CreateBuffer(device, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_HEAP_FLAG_SHARED, D3D12_RESOURCE_STATE_COMMON);
        auto upload = CreateBuffer(device, D3D12_HEAP_TYPE_UPLOAD,
            D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        auto readback = CreateBuffer(device, D3D12_HEAP_TYPE_READBACK,
            D3D12_HEAP_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

        // Each shared fence has exactly one signaler; values never rewind.
        ComPtr<ID3D12Fence> dxReady, cudaReady, readbackDone;
        CheckHR(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&dxReady)), "CreateFence(dxReady)");
        CheckHR(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED,
            IID_PPV_ARGS(&cudaReady)), "CreateFence(cudaReady)");
        CheckHR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&readbackDone)), "CreateFence(readbackDone)");
        WinHandle completed;
        completed.Value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!completed.Value) throw std::runtime_error("CreateEventW failed.");

        std::array<ComPtr<ID3D12CommandAllocator>, 2> allocators;
        std::array<ComPtr<ID3D12GraphicsCommandList>, 2> lists;
        for (size_t i = 0; i < lists.size(); ++i)
        {
            CheckHR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&allocators[i])), "CreateCommandAllocator");
            CheckHR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                allocators[i].Get(), nullptr, IID_PPV_ARGS(&lists[i])), "CreateCommandList");
            CheckHR(lists[i]->Close(), "Close(initial command list)");
        }

        CudaResources cuda;
        CheckCuda(cuCtxCreate(&cuda.Context, CU_CTX_SCHED_AUTO, cudaDevice), "cuCtxCreate");
        CheckCuda(cuStreamCreate(&cuda.Stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        CheckCuda(cuMemAllocHost(reinterpret_cast<void **>(&cuda.Observed), ByteCount), "cuMemAllocHost");
        cuda.DxReady = ImportFence(device, dxReady.Get());
        cuda.CudaReady = ImportFence(device, cudaReady.Get());

        WinHandle memoryHandle;
        CheckHR(device->CreateSharedHandle(shared.Get(), nullptr, GENERIC_ALL,
            nullptr, &memoryHandle.Value), "CreateSharedHandle(buffer)");
        const auto bufferDescription = shared->GetDesc();
        const auto allocation = device->GetResourceAllocationInfo(0, 1, &bufferDescription);
        CUDA_EXTERNAL_MEMORY_HANDLE_DESC memoryDescription = {};
        memoryDescription.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
        memoryDescription.handle.win32.handle = memoryHandle.Value;
        memoryDescription.size = allocation.SizeInBytes;
        memoryDescription.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
        CheckCuda(cuImportExternalMemory(&cuda.Memory, &memoryDescription), "cuImportExternalMemory");
        CloseHandle(memoryHandle.Value);
        memoryHandle.Value = nullptr;
        CUDA_EXTERNAL_MEMORY_BUFFER_DESC mapping = {};
        mapping.size = ByteCount;
        CheckCuda(cuExternalMemoryGetMappedBuffer(&cuda.Buffer, cuda.Memory, &mapping),
            "cuExternalMemoryGetMappedBuffer");
        std::cout << "Shared buffer: " << ByteCount << " bytes, allocation: "
                  << allocation.SizeInBytes << " bytes\n";

        for (uint32_t iteration = 1; iteration <= iterations; ++iteration)
        {
            // Previous readbackDone also protects upload data, allocators and buffer reuse.
            uint32_t *input = nullptr;
            const D3D12_RANGE noRead = {0, 0};
            CheckHR(upload->Map(0, &noRead, reinterpret_cast<void **>(&input)), "Map(upload)");
            for (size_t i = 0; i < WordCount; ++i) input[i] = InputValue(iteration, i);
            const D3D12_RANGE allBytes = {0, ByteCount};
            upload->Unmap(0, &allBytes);

            for (size_t i = 0; i < lists.size(); ++i)
            {
                CheckHR(allocators[i]->Reset(), "Reset(allocator)");
                CheckHR(lists[i]->Reset(allocators[i].Get(), nullptr), "Reset(command list)");
            }

            // 1. DX12 writes, returns resource to COMMON, then signals dxReady.
            auto *writeList = lists[0].Get();
            Transition(writeList, shared.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            writeList->CopyBufferRegion(shared.Get(), 0, upload.Get(), 0, ByteCount);
            Transition(writeList, shared.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
            CheckHR(writeList->Close(), "Close(write list)");
            ID3D12CommandList *writeSubmission[] = {writeList};
            queue->ExecuteCommandLists(1, writeSubmission);
            CheckHR(queue->Signal(dxReady.Get(), iteration), "Signal(dxReady)");

            // 2. CUDA waits on DX12, reads every input word and overwrites the second half.
            // No CPU wait is inserted between APIs. All CUDA operations share one stream.
            CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait = {};
            wait.params.fence.value = iteration;
            CheckCuda(cuWaitExternalSemaphoresAsync(&cuda.DxReady, &wait, 1, cuda.Stream),
                "cuWaitExternalSemaphoresAsync");
            CheckCuda(cuMemcpyDtoHAsync(cuda.Observed, cuda.Buffer, ByteCount, cuda.Stream), "cuMemcpyDtoHAsync");
            CheckCuda(cuMemsetD32Async(cuda.Buffer + HalfCount * sizeof(uint32_t),
                OutputValue(iteration), HalfCount, cuda.Stream), "cuMemsetD32Async");
            CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal = {};
            signal.params.fence.value = iteration;
            CheckCuda(cuSignalExternalSemaphoresAsync(&cuda.CudaReady, &signal, 1, cuda.Stream),
                "cuSignalExternalSemaphoresAsync");

            // 3. DX12 waits on CUDA before copying the shared buffer to readback.
            CheckHR(queue->Wait(cudaReady.Get(), iteration), "Wait(cudaReady)");
            auto *readList = lists[1].Get();
            Transition(readList, shared.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            readList->CopyBufferRegion(readback.Get(), 0, shared.Get(), 0, ByteCount);
            Transition(readList, shared.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            CheckHR(readList->Close(), "Close(read list)");
            ID3D12CommandList *readSubmission[] = {readList};
            queue->ExecuteCommandLists(1, readSubmission);
            CheckHR(queue->Signal(readbackDone.Get(), iteration), "Signal(readbackDone)");

            // CPU waits only to inspect values and safely reuse this one small buffer.
            CheckHR(readbackDone->SetEventOnCompletion(iteration, completed.Value), "SetEventOnCompletion");
            if (WaitForSingleObject(completed.Value, 10000) != WAIT_OBJECT_0)
                throw std::runtime_error("DX12/CUDA readback timed out after 10 seconds.");
            CheckHR(device->GetDeviceRemovedReason(), "GetDeviceRemovedReason");
            CheckCuda(cuStreamSynchronize(cuda.Stream), "cuStreamSynchronize(validation)");
            for (size_t i = 0; i < WordCount; ++i)
                RequireEqual(cuda.Observed[i], InputValue(iteration, i), iteration, i, "DX12 -> CUDA");

            uint32_t *output = nullptr;
            CheckHR(readback->Map(0, &allBytes, reinterpret_cast<void **>(&output)), "Map(readback)");
            // Copy out before checking so an exception cannot leave the resource mapped.
            std::array<uint32_t, WordCount> observed;
            std::memcpy(observed.data(), output, ByteCount);
            readback->Unmap(0, &noRead);
            for (size_t i = 0; i < WordCount; ++i)
                RequireEqual(observed[i], i < HalfCount ? InputValue(iteration, i) : OutputValue(iteration),
                    iteration, i, "CUDA -> DX12");

            if (iteration == 1 || iteration == iterations)
                std::cout << "Iteration " << iteration << ": input[0]=0x" << std::hex
                          << cuda.Observed[0] << ", DX12 readback[256]=0x" << observed[HalfCount]
                          << std::dec << " PASS\n";
        }
        std::cout << "PASS: " << iterations << " round trips; " << WordCount
                  << " input + " << WordCount << " returned words checked per iteration.\n";
    }
}

int main(int argc, char **argv)
{
    try
    {
        uint32_t iterations = 256;
        if (argc != 1)
        {
            if (argc != 3 || std::string(argv[1]) != "--iterations")
                throw std::runtime_error("Usage: HybridTracerInteropTest [--iterations 1..100000]");
            const std::string text = argv[2];
            size_t consumed = 0;
            const auto value = std::stoul(text, &consumed);
            if (consumed != text.size() || value == 0 || value > 100000)
                throw std::runtime_error("Iterations must be an integer from 1 to 100000.");
            iterations = static_cast<uint32_t>(value);
        }

        // Enable validation if the optional Windows Graphics Tools component is installed.
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
        {
            debug->EnableDebugLayer();
            std::cout << "D3D12 debug layer: enabled\n";
        }
        else
            std::cout << "D3D12 debug layer: unavailable\n";

        DX12Context dx12;
        dx12.InitializeHeadless(); // The renderer's actual adapter selection, without a window.
        ComPtr<ID3D12InfoQueue> messages;
        dx12.GetDevice()->QueryInterface(IID_PPV_ARGS(&messages));
        RunProbe(dx12, iterations);
        if (messages)
        {
            const auto count = messages->GetNumStoredMessagesAllowedByRetrievalFilter();
            for (UINT64 i = 0; i < count; ++i)
            {
                SIZE_T size = 0;
                CheckHR(messages->GetMessage(i, nullptr, &size), "GetMessage(size)");
                // D3D12_MESSAGE requires pointer alignment.
                auto storage = std::make_unique<unsigned char[]>(size);
                auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.get());
                CheckHR(messages->GetMessage(i, message, &size), "GetMessage");
                if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
                    throw std::runtime_error(std::string("D3D12 validation: ") + message->pDescription);
            }
            std::cout << "PASS: D3D12 debug layer reported no warnings or errors.\n";
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
