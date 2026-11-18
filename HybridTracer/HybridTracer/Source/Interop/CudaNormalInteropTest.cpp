#include "CudaNormalReader.h"
#include "CudaDepthPositionReader.h"
#include "DepthPositionValidation.h"
#include "InteropCommon.h"
#include "Renderer/DX12Context.h"
#include "Renderer/RasterGBufferPass.h"
#include "Renderer/DebugViewPass.h"
#include "Scene/Camera.h"
#include "Scene/Scene.h"
#ifdef HYBRIDTRACER_HAS_OPTIX_GAS
#include "Optix/OptixSceneGAS.h"
#include "Optix/OptixTracePass.h"
#include "Optix/ShadowValidation.h"
#endif

#include <d3d12sdklayers.h>
#include <DirectXPackedVector.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

using namespace HybridInterop;
using namespace DirectX;

namespace
{
    struct Options
    {
        UINT Width = 321;
        UINT Height = 181;
        UINT Frames = 8;
        bool Samples = false;
        bool DepthPosition = false;
        bool SurfaceInputs = false;
        bool OptixGas = false;
        bool OptixShadow = false;
    };

    Options ParseOptions(int argc, char **argv)
    {
        Options options;
        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if (argument == "--full-resolution")
            {
                options.Width = 1280;
                options.Height = 720;
            }
            else if (argument == "--samples") options.Samples = true;
            else if (argument == "--depth-position") options.DepthPosition = true;
            else if (argument == "--surface-inputs") options.SurfaceInputs = options.DepthPosition = true;
            else if (argument == "--optix-gas" || argument == "--optix-shadow")
            {
#ifdef HYBRIDTRACER_HAS_OPTIX_GAS
                options.OptixGas = options.SurfaceInputs = options.DepthPosition = true;
                if (argument == "--optix-shadow") options.OptixShadow = true;
#else
                throw std::runtime_error("Rebuild with HYBRIDTRACER_BUILD_OPTIX_GAS_TEST=ON and set HYBRIDTRACER_OPTIX_ROOT.");
#endif
            }
            else if (argument == "--frames" && i + 1 < argc)
            {
                const std::string text = argv[++i];
                size_t consumed = 0;
                const auto count = std::stoul(text, &consumed);
                if (consumed != text.size() || count < 1 || count > 256)
                    throw std::runtime_error("Frames must be an integer from 1 to 256.");
                options.Frames = static_cast<UINT>(count);
            }
            else throw std::runtime_error("Usage: HybridTracerNormalInteropTest [--frames 1..256] [--full-resolution] [--samples] [--depth-position|--surface-inputs|--optix-gas|--optix-shadow]");
        }
        return options;
    }

    ComPtr<ID3D12Resource> CreateResource(ID3D12Device *device,
        D3D12_HEAP_TYPE type, const D3D12_RESOURCE_DESC &description,
        D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE *clear = nullptr)
    {
        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = type;
        heap.CreationNodeMask = heap.VisibleNodeMask = 1;
        ComPtr<ID3D12Resource> resource;
        CheckHR(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
            &description, state, clear, IID_PPV_ARGS(&resource)), "CreateCommittedResource(test)");
        return resource;
    }

    std::array<float, 4> DecodeNormal(const uint16_t *pixel)
    {
        std::array<float, 4> result;
        for (size_t c = 0; c < 4; ++c)
            result[c] = PackedVector::XMConvertHalfToFloat(pixel[c]);
        return result;
    }

    XMFLOAT3 ExpectedTiltedNormal(const Scene &scene)
    {
        for (const auto &object : scene.GetObjects())
        {
            // This test intentionally uses the green plane in the existing scene fixture.
            if (!object.MaterialData || object.MaterialData->Albedo.y < 0.7f) continue;
            const XMMATRIX world = object.ObjectTransform.GetWorldMatrix();
            const XMVECTOR p0 = XMVector3TransformCoord(XMVectorSet(-1, 0, -1, 1), world);
            const XMVECTOR p1 = XMVector3TransformCoord(XMVectorSet(-1, 0, 1, 1), world);
            const XMVECTOR p2 = XMVector3TransformCoord(XMVectorSet(1, 0, 1, 1), world);
            XMFLOAT3 normal;
            // Independent geometric reference: cross product of transformed triangle edges.
            XMStoreFloat3(&normal, XMVector3Normalize(XMVector3Cross(p1 - p0, p2 - p0)));
            return normal;
        }
        throw std::runtime_error("The green plane is missing from the validation scene.");
    }

    struct Sample
    {
        bool Found = false;
        UINT X = 0;
        UINT Y = 0;
        size_t Offset = 0;
    };

    uint64_t ValidatePixels(const uint16_t *cuda, const std::vector<uint16_t> &dx,
        const Options &options, UINT frame, const XMFLOAT3 &tilted, std::ofstream &csv)
    {
        const std::array<const char *, 4> labels = {"background", "floor", "tilted_plane", "other_geometry"};
        std::array<Sample, 4> samples;
        std::array<size_t, 4> counts = {};
        float maxLengthError = 0;
        uint64_t hash = 14695981039346656037ull;
        for (UINT y = 0; y < options.Height; ++y)
        {
            for (UINT x = 0; x < options.Width; ++x)
            {
                const size_t offset = (static_cast<size_t>(y) * options.Width + x) * 4;
                for (size_t c = 0; c < 4; ++c)
                {
                    if (cuda[offset + c] != dx[offset + c])
                    {
                        std::ostringstream message;
                        message << "Frame " << frame << ", pixel (" << x << ',' << y << "), channel " << c
                                << ": CUDA half=0x" << std::hex << cuda[offset + c]
                                << ", DX12 half=0x" << dx[offset + c];
                        throw std::runtime_error(message.str());
                    }
                    hash = (hash ^ cuda[offset + c]) * 1099511628211ull;
                }
                const auto n = DecodeNormal(cuda + offset);
                for (const auto component : n)
                    if (!std::isfinite(component)) throw std::runtime_error("Non-finite normal component.");
                size_t category = 0;
                if (n[3] == 0.0f)
                {
                    if (n[0] != 0.0f || n[1] != 0.0f || n[2] != 0.0f)
                        throw std::runtime_error("Background normal must be (0,0,0,0).");
                }
                else
                {
                    if (n[3] != 1.0f) throw std::runtime_error("Surface normal alpha must equal one.");
                    const float length = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                    maxLengthError = std::max(maxLengthError, std::abs(length - 1.0f));
                    if (maxLengthError > 0.002f) throw std::runtime_error("World normal is not unit length within half-float precision.");
                    const auto closeEnough = [](float a, float b) { return std::abs(a-b) < 0.001f; };
                    if (closeEnough(n[0], 0) && closeEnough(n[1], 1) && closeEnough(n[2], 0)) category = 1;
                    else if (closeEnough(n[0], tilted.x) && closeEnough(n[1], tilted.y) && closeEnough(n[2], tilted.z)) category = 2;
                    else category = 3;
                }
                ++counts[category];
                if (!samples[category].Found) samples[category] = {true, x, y, offset};
            }
        }
        for (size_t category = 0; category < samples.size(); ++category)
        {
            if (!samples[category].Found)
                throw std::runtime_error(std::string("Required scene sample is missing: ") + labels[category]);
            const auto &sample = samples[category];
            const auto n = DecodeNormal(cuda + sample.Offset);
            if (frame == 1)
            {
                std::cout << labels[category] << " pixel(" << sample.X << ',' << sample.Y
                          << "): CUDA=DX12=(" << n[0] << ',' << n[1] << ',' << n[2] << ',' << n[3] << ")\n";
            }
            if (csv.is_open())
            {
                csv << frame << ',' << labels[category] << ',' << sample.X << ',' << sample.Y;
                for (const auto value : n) csv << ',' << value;
                for (const auto value : DecodeNormal(dx.data() + sample.Offset)) csv << ',' << value;
                for (size_t c = 0; c < 4; ++c) csv << ',' << cuda[sample.Offset + c];
                for (size_t c = 0; c < 4; ++c) csv << ',' << dx[sample.Offset + c];
                csv << '\n';
            }
        }
        std::cout << "Frame " << frame << ": " << options.Width * options.Height
                  << " pixels bit-exact; background=" << counts[0] << ", floor=" << counts[1]
                  << ", tilted=" << counts[2] << ", other=" << counts[3]
                  << ", max |length-1|=" << maxLengthError << '\n';
        return hash;
    }

    void RunNormalProbe(DX12Context &dx12, const Options &options)
    {
        auto *device = dx12.GetDevice();
        auto *queue = dx12.GetCommandQueue();
        Scene scene;
        scene.Initialize(device);
        RasterGBufferPass raster;
        raster.Initialize(device, DXGI_FORMAT_R8G8B8A8_UNORM, options.Width, options.Height,
            DX12Context::FrameCount, true, options.DepthPosition, options.SurfaceInputs);
        DebugViewPass debugView;
        debugView.Initialize(device, DXGI_FORMAT_R8G8B8A8_UNORM, raster.GetAlbedoBuffer(),
            raster.GetNormalBuffer(), raster.GetRasterDepthBuffer());

        D3D12_RESOURCE_DESC colorDescription = raster.GetAlbedoBuffer()->GetDesc();
        const D3D12_CLEAR_VALUE colorClear = {DXGI_FORMAT_R8G8B8A8_UNORM, {0.08f, 0.12f, 0.18f, 1.0f}};
        auto color = CreateResource(device, D3D12_HEAP_TYPE_DEFAULT, colorDescription,
            D3D12_RESOURCE_STATE_RENDER_TARGET, &colorClear);
        ComPtr<ID3D12DescriptorHeap> rtvHeap;
        D3D12_DESCRIPTOR_HEAP_DESC heap = {};
        heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heap.NumDescriptors = 1;
        CheckHR(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&rtvHeap)), "CreateDescriptorHeap(test RTV)");
        const auto rtv = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(color.Get(), nullptr, rtv);

        auto *normal = raster.GetNormalBuffer();
        const auto normalDescription = normal->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint = {};
        UINT64 readbackBytes = 0;
        device->GetCopyableFootprints(&normalDescription, 0, 1, 0, &footprint, nullptr, nullptr, &readbackBytes);
        D3D12_RESOURCE_DESC buffer = {};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = readbackBytes;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        auto readback = CreateResource(device, D3D12_HEAP_TYPE_READBACK, buffer, D3D12_RESOURCE_STATE_COPY_DEST);
        const size_t rowBytes = static_cast<size_t>(options.Width) * 8;
        std::cout << "Normal: " << options.Width << 'x' << options.Height << " RGBA16F; rowBytes="
                  << rowBytes << ", DX12 rowPitch=" << footprint.Footprint.RowPitch << '\n';

        auto *depth = raster.GetRasterDepthBuffer();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT depthFootprint = {};
        UINT64 depthReadbackBytes = 0;
        ComPtr<ID3D12Resource> depthReadback;
        if (options.DepthPosition)
        {
            const auto desc = depth->GetDesc();
            device->GetCopyableFootprints(&desc, 0, 1, 0, &depthFootprint, nullptr, nullptr, &depthReadbackBytes);
            buffer.Width = depthReadbackBytes;
            depthReadback = CreateResource(device, D3D12_HEAP_TYPE_READBACK, buffer, D3D12_RESOURCE_STATE_COPY_DEST);
            std::cout << "Depth: R32_FLOAT; rowBytes=" << options.Width*sizeof(float)
                      << ", DX12 rowPitch=" << depthFootprint.Footprint.RowPitch << '\n';
        }

        auto *albedo = raster.GetAlbedoBuffer();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT albedoFootprint = {};
        UINT64 albedoReadbackBytes = 0;
        ComPtr<ID3D12Resource> albedoReadback;
        if (options.SurfaceInputs)
        {
            const auto desc = albedo->GetDesc();
            device->GetCopyableFootprints(&desc, 0, 1, 0, &albedoFootprint, nullptr, nullptr, &albedoReadbackBytes);
            buffer.Width = albedoReadbackBytes;
            albedoReadback = CreateResource(device, D3D12_HEAP_TYPE_READBACK, buffer, D3D12_RESOURCE_STATE_COPY_DEST);
            std::cout << "Albedo: RGBA8_UNORM; rowBytes=" << options.Width*4
                      << ", DX12 rowPitch=" << albedoFootprint.Footprint.RowPitch << '\n';
        }

        std::array<ComPtr<ID3D12CommandAllocator>, 2> allocators;
        std::array<ComPtr<ID3D12GraphicsCommandList>, 2> lists;
        for (size_t i = 0; i < lists.size(); ++i)
        {
            CheckHR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&allocators[i])), "CreateCommandAllocator");
            CheckHR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                allocators[i].Get(), nullptr, IID_PPV_ARGS(&lists[i])), "CreateCommandList");
            CheckHR(lists[i]->Close(), "Close(initial list)");
        }
        ComPtr<ID3D12Fence> completed;
        CheckHR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&completed)), "CreateFence(completed)");
        WinHandle completedEvent;
        completedEvent.Value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!completedEvent.Value) throw std::runtime_error("CreateEventW failed.");
        CudaNormalReader reader;
        reader.Initialize(device, normal);
        std::unique_ptr<CudaDepthPositionReader> positionReader;
        if (options.DepthPosition)
        {
            positionReader = std::make_unique<CudaDepthPositionReader>();
            positionReader->Initialize(device, depth,
                options.SurfaceInputs ? normal : nullptr, options.SurfaceInputs ? albedo : nullptr);
        }

#ifdef HYBRIDTRACER_HAS_OPTIX_GAS
        // Declared after positionReader: GAS and OptiX context die before their CUDA owner.
        std::unique_ptr<OptixSceneGAS> gas;
        if (options.OptixGas)
        {
            gas = std::make_unique<OptixSceneGAS>();
            gas->Initialize(positionReader->GetContext(), positionReader->GetStream());
            gas->Build(scene);
            gas->ValidateAgainstScene(scene);
        }
        // Destroy trace programs/output before their borrowed OptiX context and CUDA owner.
        std::unique_ptr<OptixTracePass> trace;
        std::vector<ShadowResult> shadowResults;
        std::ofstream shadowCsv;
        if (options.OptixShadow)
        {
            trace = std::make_unique<OptixTracePass>();
            trace->Initialize(positionReader->GetContext(), positionReader->GetStream(), *gas, options.Width, options.Height);
            shadowResults.resize(static_cast<size_t>(options.Width)*options.Height);
            if (options.Samples)
            {
                shadowCsv.open(std::filesystem::current_path() / L"OptixShadowSamples.csv");
                if (!shadowCsv) throw std::runtime_error("Cannot write OptixShadowSamples.csv.");
                shadowCsv << "frame,x,y,valid,visibility,blocker_object_plus_one,gpu_distance,cpu_distance,light_x,light_y,light_z\n";
            }
        }
#endif

        Camera camera;
        camera.SetLens(XM_PIDIV4, static_cast<float>(options.Width) / options.Height, 0.1f, 100.0f);
        const XMFLOAT3 tilted = ExpectedTiltedNormal(scene);
        std::cout << std::setprecision(9) << "CPU geometric tilted normal: ("
                  << tilted.x << ',' << tilted.y << ',' << tilted.z << ")\n";
        std::ofstream csv;
        if (options.Samples)
        {
            csv.open(std::filesystem::current_path() /
                (options.OptixGas ? L"OptixNormalSamples.csv" : options.SurfaceInputs ? L"PackedSurfaceNormalSamples.csv" :
                    options.DepthPosition ? L"SurfaceNormalSamples.csv" : L"NormalInteropSamples.csv"));
            if (!csv) throw std::runtime_error("Cannot write normal sample CSV.");
            csv << std::setprecision(9)
                << "frame,category,x,y,cuda_x,cuda_y,cuda_z,cuda_a,dx12_x,dx12_y,dx12_z,dx12_a,"
                << "cuda_half_x,cuda_half_y,cuda_half_z,cuda_half_a,dx12_half_x,dx12_half_y,dx12_half_z,dx12_half_a\n";
        }
        std::vector<uint16_t> dxPixels(static_cast<size_t>(options.Width) * options.Height * 4);
        std::vector<float> dxDepth(static_cast<size_t>(options.Width) * options.Height);
        std::vector<uint8_t> dxAlbedo;
        if (options.SurfaceInputs) dxAlbedo.resize(static_cast<size_t>(options.Width)*options.Height*4);
        std::ofstream positionCsv;
        if (options.Samples && options.DepthPosition)
        {
            positionCsv.open(std::filesystem::current_path() /
                (options.OptixGas ? L"OptixPositionSamples.csv" : options.SurfaceInputs ? L"SurfacePositionSamples.csv" : L"DepthPositionSamples.csv"));
            if (!positionCsv) throw std::runtime_error("Cannot write position sample CSV.");
            positionCsv << std::setprecision(9)
                << "frame,object,x,y,depth,gpu_x,gpu_y,gpu_z,cpu_hit_x,cpu_hit_y,cpu_hit_z,world_error\n";
        }
        std::ofstream surfaceCsv;
        if (options.Samples && options.SurfaceInputs)
        {
            surfaceCsv.open(std::filesystem::current_path() /
                (options.OptixGas ? L"OptixSurfaceInputSamples.csv" : L"SurfaceInputSamples.csv"));
            if (!surfaceCsv) throw std::runtime_error("Cannot write SurfaceInputSamples.csv.");
            surfaceCsv << std::setprecision(9)
                << "frame,object,x,y,position_x,position_y,position_z,normal_x,normal_y,normal_z,"
                << "albedo_r,albedo_g,albedo_b,material_r,material_g,material_b,world_error\n";
        }
        uint64_t previousHash = 0;
        for (UINT frame = 1; frame <= options.Frames; ++frame)
        {
#ifdef HYBRIDTRACER_HAS_OPTIX_GAS
            ShadowSettings shadowSettings = {};
            if (trace)
            {
                // Alternate light as well as camera to catch stale launch parameters/output.
                const XMFLOAT3 source = frame % 2 ? scene.GetDirectionalLight().Direction : XMFLOAT3{0.7f,1.0f,0.2f};
                XMFLOAT3 direction;
                XMStoreFloat3(&direction, XMVector3Normalize(XMLoadFloat3(&source)));
                shadowSettings = {direction.x,direction.y,direction.z,0.001f,0.0001f,1e16f};
            }
#endif
            // Alternate view positions: different coverage detects a stale previous frame.
            const float eyeX = frame % 2 ? -0.45f : 0.45f;
            camera.LookAt({eyeX, 2.6f, -7.0f}, {0, 0, 0.5f}, {0, 1, 0});
            if (options.DepthPosition)
                camera.SetLens(XM_PIDIV4, static_cast<float>(options.Width)/options.Height,
                    frame % 2 ? 0.1f : 0.25f, frame % 2 ? 100.0f : 40.0f);
            for (size_t i = 0; i < lists.size(); ++i)
            {
                CheckHR(allocators[i]->Reset(), "Reset(allocator)");
                CheckHR(lists[i]->Reset(allocators[i].Get(), nullptr), "Reset(list)");
            }
            auto *renderList = lists[0].Get();
            raster.Render(renderList, rtv, (frame-1) % DX12Context::FrameCount, camera, scene);
            renderList->OMSetRenderTargets(0, nullptr, FALSE, nullptr);
            Transition(renderList, normal, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION source = {};
            source.pResource = normal;
            source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION destination = {};
            destination.pResource = readback.Get();
            destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            destination.PlacedFootprint = footprint;
            renderList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
            Transition(renderList, normal, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            if (options.DepthPosition)
            {
                Transition(renderList, depth, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
                source.pResource = depth;
                destination.pResource = depthReadback.Get();
                destination.PlacedFootprint = depthFootprint;
                renderList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
                Transition(renderList, depth, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            }
            if (options.SurfaceInputs)
            {
                Transition(renderList, albedo, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
                source.pResource = albedo;
                destination.pResource = albedoReadback.Get();
                destination.PlacedFootprint = albedoFootprint;
                renderList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
                Transition(renderList, albedo, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
            }
            CheckHR(renderList->Close(), "Close(raster and reference copy)");
            ID3D12CommandList *rasterSubmission[] = {renderList};
            queue->ExecuteCommandLists(1, rasterSubmission);

            reader.EnqueueRead(queue, frame); // GPU-only handoff in both directions.
            if (positionReader)
            {
                XMFLOAT4X4 inverseViewProjection;
                XMStoreFloat4x4(&inverseViewProjection,
                    XMMatrixInverse(nullptr, camera.GetView()*camera.GetProjection()));
                positionReader->EnqueueRead(queue, frame, &inverseViewProjection._11);
            }
#ifdef HYBRIDTRACER_HAS_OPTIX_GAS
            if (trace) trace->Launch(positionReader->GetDeviceSurfaceInputs(), shadowSettings);
#endif

            auto *resumeList = lists[1].Get();
            Transition(resumeList, normal, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
            if (positionReader)
                Transition(resumeList, depth, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
            if (options.SurfaceInputs)
                Transition(resumeList, albedo, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
            const UINT debugMode = (frame-1) % 4;
            if (debugMode) debugView.Render(resumeList, rtv, debugMode);
            CheckHR(resumeList->Close(), "Close(resume and debug view)");
            ID3D12CommandList *resumeSubmission[] = {resumeList};
            queue->ExecuteCommandLists(1, resumeSubmission);
            CheckHR(queue->Signal(completed.Get(), frame), "Signal(completed)");
            CheckHR(completed->SetEventOnCompletion(frame, completedEvent.Value), "SetEventOnCompletion");
            if (WaitForSingleObject(completedEvent.Value, 10000) != WAIT_OBJECT_0)
                throw std::runtime_error("Normal interop completion timed out after 10 seconds.");
            CheckHR(device->GetDeviceRemovedReason(), "GetDeviceRemovedReason");
            reader.WaitForReadback();
            if (positionReader) positionReader->WaitForReadback();

            const D3D12_RANGE readRange = {0, static_cast<SIZE_T>(readbackBytes)};
            uint8_t *mapped = nullptr;
            CheckHR(readback->Map(0, &readRange, reinterpret_cast<void **>(&mapped)), "Map(normal readback)");
            for (UINT y = 0; y < options.Height; ++y)
                std::memcpy(dxPixels.data() + static_cast<size_t>(y) * options.Width * 4,
                    mapped + footprint.Offset + static_cast<size_t>(y) * footprint.Footprint.RowPitch, rowBytes);
            const D3D12_RANGE noWrite = {0, 0};
            readback->Unmap(0, &noWrite);
            const auto hash = ValidatePixels(reader.GetPixels(), dxPixels, options, frame, tilted, csv);
            if (frame > 1 && hash == previousHash)
                throw std::runtime_error("Normal image did not change after moving the camera.");
            previousHash = hash;
            if (positionReader)
            {
                const D3D12_RANGE depthRange = {0, static_cast<SIZE_T>(depthReadbackBytes)};
                CheckHR(depthReadback->Map(0, &depthRange, reinterpret_cast<void **>(&mapped)), "Map(depth readback)");
                for (UINT y = 0; y < options.Height; ++y)
                    std::memcpy(dxDepth.data() + static_cast<size_t>(y)*options.Width,
                        mapped + depthFootprint.Offset + static_cast<size_t>(y)*depthFootprint.Footprint.RowPitch,
                        static_cast<size_t>(options.Width)*sizeof(float));
                depthReadback->Unmap(0, &noWrite);
                if (options.SurfaceInputs)
                {
                    const D3D12_RANGE albedoRange = {0, static_cast<SIZE_T>(albedoReadbackBytes)};
                    CheckHR(albedoReadback->Map(0, &albedoRange, reinterpret_cast<void **>(&mapped)), "Map(albedo readback)");
                    for (UINT y = 0; y < options.Height; ++y)
                        std::memcpy(dxAlbedo.data()+static_cast<size_t>(y)*options.Width*4,
                            mapped+albedoFootprint.Offset+static_cast<size_t>(y)*albedoFootprint.Footprint.RowPitch,
                            static_cast<size_t>(options.Width)*4);
                    albedoReadback->Unmap(0, &noWrite);
                    ValidateSurfaceInputs(positionReader->GetSurfaceInputs(), positionReader->GetPositions(),
                        reader.GetPixels(), positionReader->GetAlbedoPixels(), dxAlbedo,
                        options.Width, options.Height, frame);
                }
                ValidateDepthPositions(positionReader->GetDepthPixels(), positionReader->GetPositions(),
                    reader.GetPixels(), dxDepth, camera, scene, options.Width, options.Height, frame,
                    positionCsv.is_open() ? &positionCsv : nullptr,
                    options.SurfaceInputs ? positionReader->GetSurfaceInputs() : nullptr,
                    surfaceCsv.is_open() ? &surfaceCsv : nullptr);
            }
#ifdef HYBRIDTRACER_HAS_OPTIX_GAS
            if (trace)
            {
                // Diagnostic readback only. Launch itself consumes GPU SurfaceInput directly.
                CheckCuda(cuCtxSetCurrent(positionReader->GetContext()), "cuCtxSetCurrent(shadow readback)");
                CheckCuda(cuMemcpyDtoH(shadowResults.data(), trace->GetDeviceOutput(), trace->GetOutputBytes()), "cuMemcpyDtoH(shadow results)");
                ValidateShadowResults(shadowResults.data(), positionReader->GetSurfaceInputs(), scene,
                    shadowSettings, options.Width, options.Height, frame, shadowCsv.is_open() ? &shadowCsv : nullptr);
            }
#endif
        }
        if (csv.is_open())
        {
            csv.flush();
            if (!csv) throw std::runtime_error("Failed to write normal sample CSV.");
            std::cout << "Saved " << (options.OptixGas ? "OptixNormalSamples.csv" : options.SurfaceInputs ? "PackedSurfaceNormalSamples.csv" :
                    options.DepthPosition ? "SurfaceNormalSamples.csv" : "NormalInteropSamples.csv")
                      << " (four representative pixels per frame).\n";
        }
        if (positionCsv.is_open())
        {
            positionCsv.flush();
            if (!positionCsv) throw std::runtime_error("Failed to write position sample CSV.");
            std::cout << "Saved " << (options.OptixGas ? "OptixPositionSamples.csv" : options.SurfaceInputs ? "SurfacePositionSamples.csv" : "DepthPositionSamples.csv")
                      << " (one geometric reference per scene object and frame).\n";
        }
        if (surfaceCsv.is_open())
        {
            surfaceCsv.flush();
            if (!surfaceCsv) throw std::runtime_error("Failed to write surface input CSV.");
            std::cout << "Saved " << (options.OptixGas ? "OptixSurfaceInputSamples.csv" : "SurfaceInputSamples.csv")
                      << " (coherent position/normal/albedo samples).\n";
        }
#ifdef HYBRIDTRACER_HAS_OPTIX_GAS
        if (shadowCsv.is_open())
        {
            shadowCsv.flush();
            if (!shadowCsv) throw std::runtime_error("Failed to write shadow samples.");
            std::cout << "Saved OptixShadowSamples.csv (background, miss and per-blocker samples).\n";
        }
        if (gas)
        {
            // Revalidate after per-frame work/context switching to verify resource lifetime.
            std::ofstream gasReport;
            if (options.Samples)
            {
                gasReport.open(std::filesystem::current_path() / L"OptixGASReport.csv");
                if (!gasReport) throw std::runtime_error("Cannot write OptixGASReport.csv.");
            }
            gas->ValidateAgainstScene(scene, gasReport.is_open() ? &gasReport : nullptr);
            if (gasReport.is_open())
            {
                gasReport.flush();
                if (!gasReport) throw std::runtime_error("Failed to write GAS report CSV.");
                std::cout << "Saved OptixGASReport.csv (four geometry inputs and GAS root bounds).\n";
            }
        }
#endif
        std::cout << "Completed: actual raster normal G-buffer read through CUDA; "
                  << options.Frames << " changing frames; DebugView modes 0..3 exercised with 4+ frames.\n";
    }
}

int main(int argc, char **argv)
{
    try
    {
        const auto options = ParseOptions(argc, argv);
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))
        {
            debug->EnableDebugLayer();
            std::cout << "D3D12 debug layer: enabled\n";
        }
        else std::cout << "D3D12 debug layer: unavailable\n";
        DX12Context dx12;
        dx12.InitializeHeadless();
        ComPtr<ID3D12InfoQueue> messages;
        dx12.GetDevice()->QueryInterface(IID_PPV_ARGS(&messages));
        RunNormalProbe(dx12, options);
        if (messages)
        {
            const auto count = messages->GetNumStoredMessagesAllowedByRetrievalFilter();
            for (UINT64 i = 0; i < count; ++i)
            {
                SIZE_T size = 0;
                CheckHR(messages->GetMessage(i, nullptr, &size), "GetMessage(size)");
                auto storage = std::make_unique<unsigned char[]>(size);
                auto *message = reinterpret_cast<D3D12_MESSAGE *>(storage.get());
                CheckHR(messages->GetMessage(i, message, &size), "GetMessage");
                if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
                    throw std::runtime_error(std::string("D3D12 validation: ") + message->pDescription);
            }
            std::cout << "PASS: D3D12 debug layer reported no warnings or errors.\n";
        }
        std::cout << "PASS: all requested interop/OptiX checks passed.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
