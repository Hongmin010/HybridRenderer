#include "Renderer/HybridFramePipeline.h"
#include "Renderer/IndirectBounceOptions.h"
#include "Scene/SceneFile.h"
#include "Renderer/DX12Context.h"
#include "Optix/ShadowValidation.h"
#include "InteropCommon.h"
#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include <d3d12sdklayers.h>
#include <DirectXPackedVector.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <type_traits>
#include <vector>
using namespace HybridInterop;
using namespace DirectX;

namespace
{
    ComPtr<ID3D12Resource> Buffer(ID3D12Device *device, size_t bytes)
    {
        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> resource;
        CheckHR(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
            D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&resource)),"CreateCommittedResource(readback)");
        return resource;
    }
    struct TextureReadback
    {
        ID3D12Resource *Source = nullptr;
        ComPtr<ID3D12Resource> Target;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT Footprint = {};
        UINT Rows = 0;
        UINT64 RowBytes = 0, TotalBytes = 0;
        void Initialize(ID3D12Device *device, ID3D12Resource *source)
        {
            Source = source;
            const auto desc = source->GetDesc();
            device->GetCopyableFootprints(&desc,0,1,0,&Footprint,&Rows,&RowBytes,&TotalBytes);
            Target = Buffer(device,static_cast<size_t>(TotalBytes));
        }
        void Copy(ID3D12GraphicsCommandList *list)
        {
            Transition(list,Source,D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION from = {}, to = {};
            from.pResource = Source;
            from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            to.pResource = Target.Get();
            to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            to.PlacedFootprint = Footprint;
            list->CopyTextureRegion(&to,0,0,0,&from,nullptr);
            Transition(list,Source,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_RENDER_TARGET);
        }
        template<class T> std::vector<T> Read()
        {
            std::vector<T> result(static_cast<size_t>(RowBytes)*Rows/sizeof(T));
            unsigned char *data = nullptr;
            const D3D12_RANGE all = {0,static_cast<size_t>(TotalBytes)}, none = {0,0};
            CheckHR(Target->Map(0,&all,reinterpret_cast<void **>(&data)),"Map(texture readback)");
            for (UINT y = 0; y < Rows; ++y)
                std::memcpy(reinterpret_cast<unsigned char *>(result.data())+y*RowBytes,
                    data+Footprint.Offset+static_cast<size_t>(y)*Footprint.Footprint.RowPitch,static_cast<size_t>(RowBytes));
            Target->Unmap(0,&none);
            return result;
        }
    };
    void SaveBMP(const std::filesystem::path &path, const std::vector<uint8_t> &rgba, UINT width, UINT height)
    {
        BITMAPFILEHEADER file = {};
        BITMAPINFOHEADER info = {};
        file.bfType = 0x4d42;
        file.bfOffBits = sizeof(file)+sizeof(info);
        file.bfSize = file.bfOffBits+width*height*4;
        info.biSize = sizeof(info);
        info.biWidth = static_cast<LONG>(width);
        info.biHeight = -static_cast<LONG>(height);
        info.biPlanes = 1;
        info.biBitCount = 32;
        info.biCompression = BI_RGB;
        std::vector<uint8_t> bgra = rgba;
        for (size_t i = 0; i < bgra.size(); i += 4) std::swap(bgra[i],bgra[i+2]);
        std::ofstream out(path,std::ios::binary);
        out.write(reinterpret_cast<const char *>(&file),sizeof(file));
        out.write(reinterpret_cast<const char *>(&info),sizeof(info));
        out.write(reinterpret_cast<const char *>(bgra.data()),bgra.size());
        if (!out) throw std::runtime_error("Cannot write renderer capture.");
    }
    void SaveLinearComparison(UINT width, UINT height, UINT spp, UINT seed,
        const std::vector<float> &depth, const std::vector<uint16_t> &direct,
        const std::vector<uint16_t> &indirect, const std::vector<uint16_t> &both)
    {
        // Little-endian header, then tightly packed top-down R32F depth and three RGBA16F images.
        std::ofstream out("CompositeLinear.bin",std::ios::binary);
        out.write("HTHDR001",8);
        const uint32_t header[] = {width,height,spp,seed};
        out.write(reinterpret_cast<const char *>(header),sizeof(header));
        const auto writeImage = [&out](const auto &image)
        {
            out.write(reinterpret_cast<const char *>(image.data()),image.size()*sizeof(image[0]));
        };
        writeImage(depth);
        writeImage(direct);
        writeImage(indirect);
        writeImage(both);
        out.close();
        if (!out) throw std::runtime_error("Cannot write CompositeLinear.bin.");
    }
    void WriteVector(std::ostream &out, const XMFLOAT3 &value)
    {
        out << '[' << value.x << ',' << value.y << ',' << value.z << ']';
    }
    void WriteMatrix(std::ostream &out, FXMMATRIX value)
    {
        XMFLOAT4X4 matrix;
        XMStoreFloat4x4(&matrix,value);
        out << '[';
        for (UINT row = 0; row < 4; ++row)
            for (UINT column = 0; column < 4; ++column)
            {
                if (row != 0 || column != 0) out << ',';
                out << matrix.m[row][column];
            }
        out << ']';
    }
    void WriteLight(std::ostream &out, const SceneLight &light)
    {
        std::visit([&](const auto &source)
        {
            using Type = std::decay_t<decltype(source)>;
            float strength;
            if constexpr (std::is_same_v<Type,DirectionalLight>)
            {
                out << "{\"type\":\"directional\",\"direction_to_light\":";
                WriteVector(out,source.Direction);
                out << ",\"irradiance\":";
                strength = source.Intensity;
            }
            else if constexpr (std::is_same_v<Type,PointLight>)
            {
                out << "{\"type\":\"point\",\"position\":";
                WriteVector(out,source.Position);
                out << ",\"intensity\":";
                strength = source.Intensity;
            }
            else
            {
                out << "{\"type\":\"rectangle\",\"center\":";
                WriteVector(out,source.Center);
                out << ",\"half_u\":";
                WriteVector(out,source.HalfU);
                out << ",\"half_v\":";
                WriteVector(out,source.HalfV);
                out << ",\"radiance\":";
                strength = source.Radiance;
            }
            const auto &color = source.Color;
            WriteVector(out,{color.x*strength,color.y*strength,color.z*strength});
            out << '}';
        },light);
    }
    void SaveComparisonScene(UINT width, UINT height, UINT spp, UINT seed, UINT maxIndirectBounces,
        const Scene &scene, const Camera &camera, const DirectionalLight &light,
        const std::vector<float> &depth, const std::vector<uint8_t> &albedo)
    {
        std::ofstream out("CompositeScene.json");
        out << std::setprecision(std::numeric_limits<float>::max_digits10)
            << "{\n  \"width\":" << width << ", \"height\":" << height
            << ", \"spp\":" << spp << ", \"seed\":" << seed;
        // The original scene format implied one indirect bounce.
        if (maxIndirectBounces != DefaultIndirectBounces)
            out << ", \"max_indirect_bounces\":" << maxIndirectBounces;
        out << ",\n  \"inverse_view_projection\":";
        WriteMatrix(out,XMMatrixInverse(nullptr,camera.GetView()*camera.GetProjection()));
        out << ",\n  \"direction_to_light\":";
        WriteVector(out,light.Direction);
        out << ",\n  \"irradiance\":";
        WriteVector(out,{light.Color.x*light.Intensity,light.Color.y*light.Intensity,light.Color.z*light.Intensity});
        if (!scene.GetAdditionalLights().empty())
        {
            out << ",\n  \"frame_accumulation\":false,\n  \"lights\":[";
            bool separate = light.Intensity > 0;
            if (separate) WriteLight(out,light);
            for (const auto &extra : scene.GetAdditionalLights())
            {
                if (separate) out << ',';
                WriteLight(out,extra);
                separate = true;
            }
            out << ']';
        }
        std::set<std::array<uint8_t,3>> visibleAlbedos;
        for (size_t pixel = 0; pixel < depth.size(); ++pixel)
            if (depth[pixel] < 1.0f)
                visibleAlbedos.insert({albedo[pixel*4],albedo[pixel*4+1],albedo[pixel*4+2]});
        out << ",\n  \"gbuffer_albedos_rgb8\":[";
        bool firstAlbedo = true;
        for (const auto &rgb : visibleAlbedos)
        {
            if (!firstAlbedo) out << ',';
            firstAlbedo = false;
            out << '[' << unsigned(rgb[0]) << ',' << unsigned(rgb[1]) << ',' << unsigned(rgb[2]) << ']';
        }
        out << ']';
        out << ",\n  \"objects\":[";
        bool first = true;
        for (const auto &object : scene.GetObjects())
        {
            if (!first) out << ',';
            first = false;
            out << "\n    {\"world\":";
            WriteMatrix(out,object.ObjectTransform.GetWorldMatrix());
            out << ", \"reflectance\":";
            WriteVector(out,object.MaterialData->Albedo);
            if (object.MaterialData->TwoSided) out << ", \"two_sided\":true";
            if (object.MaterialData->EmitterLight) out << ", \"emitter_light\":" << object.MaterialData->EmitterLight;
            const auto &vertices = object.MeshData->GetVertices();
            const auto writeAttribute = [&](const char *name, XMFLOAT3 Vertex::*attribute)
            {
                out << ", \"" << name << "\":[";
                for (size_t i = 0; i < vertices.size(); ++i)
                {
                    if (i) out << ',';
                    WriteVector(out,vertices[i].*attribute);
                }
                out << ']';
            };
            writeAttribute("positions",&Vertex::Position);
            writeAttribute("normals",&Vertex::Normal);
            out << ", \"indices\":[";
            const auto &indices = object.MeshData->GetIndices();
            for (size_t i = 0; i < indices.size(); ++i)
            {
                if (i) out << ',';
                out << indices[i];
            }
            out << "]}";
        }
        out << "\n  ]\n}\n";
        out.close();
        if (!out) throw std::runtime_error("Cannot write CompositeScene.json.");
    }
    double SRGB(double x) { return x <= 0.0031308 ? 12.92*x : 1.055*std::pow(x,1.0/2.4)-0.055; }
    void ValidateColor(const std::vector<uint8_t> &albedo, const std::vector<uint16_t> &normal,
        const std::vector<float> &depth, const std::vector<uint16_t> &hdr, const std::vector<uint8_t> &display,
        const std::vector<ShadowResult> &shadow, const std::vector<IndirectLighting> &indirect, const DirectionalLight &light, HybridView view,
        float exposure, UINT width, UINT height, UINT frame, UINT maxIndirectBounces, bool checkFixture)
    {
        double maxHDR = 0, maxHDRError = 0;
        int maxByteError = 0;
        size_t darkened = 0, lowSRGB = 0, indirectLit = 0, additionalLit = 0;
        const bool additional = indirect.size() == depth.size()*AdditionalLightingPlanes;
        for (size_t i = 0; i < depth.size(); ++i)
        {
            const bool background = depth[i] >= 1;
            const double rawN[] = {PackedVector::XMConvertHalfToFloat(normal[i*4]),
                PackedVector::XMConvertHalfToFloat(normal[i*4+1]),PackedVector::XMConvertHalfToFloat(normal[i*4+2])};
            const double length = std::sqrt(rawN[0]*rawN[0]+rawN[1]*rawN[1]+rawN[2]*rawN[2]);
            const double cosine = background ? 0 : std::max(0.0,(rawN[0]*light.Direction.x+rawN[1]*light.Direction.y+rawN[2]*light.Direction.z)/length);
            const double visibility = view == HybridView::Unshadowed ? 1 : shadow[i].Visibility;
            const double colors[] = {light.Color.x,light.Color.y,light.Color.z};
            const double bg[] = {0.08,0.12,0.18};
            if (!background && shadow[i].Visibility == 0 && cosine > 0.1) ++darkened;
            const auto &bounce = indirect[i];
            if (bounce.Valid != (background ? 0u : 1u) || !std::isfinite(bounce.R) || !std::isfinite(bounce.G) ||
                !std::isfinite(bounce.B) || bounce.R < 0 || bounce.G < 0 || bounce.B < 0 ||
                (background && (bounce.R != 0 || bounce.G != 0 || bounce.B != 0)))
                throw std::runtime_error("Invalid indirect radiance/background contract.");
            if (bounce.R+bounce.G+bounce.B > 0) ++indirectLit;
            const float gi[] = {bounce.R,bounce.G,bounce.B};
            double extra[3] = {}, visibilityView = shadow[i].Visibility;
            if (additional)
            {
                const auto &d = indirect[i+AdditionalDirectPlane*depth.size()];
                const auto &u = indirect[i+AdditionalUnshadowedPlane*depth.size()];
                const double visibleColors[] = {d.R,d.G,d.B}, openColors[] = {u.R,u.G,u.B};
                double visibleSum = 0, openSum = 0;
                for (size_t c = 0; c < 3; ++c)
                {
                    if (!std::isfinite(openColors[c]) || !std::isfinite(visibleColors[c]) ||
                        visibleColors[c] < 0 || visibleColors[c] > openColors[c])
                        throw std::runtime_error("Invalid additional direct radiance or shadow coverage.");
                    const double directional = (double(albedo[i*4+c])/255)*colors[c]*cosine*light.Intensity/3.14159265358979323846;
                    visibleSum += directional*shadow[i].Visibility+visibleColors[c];
                    openSum += directional+openColors[c];
                    extra[c] = view == HybridView::Unshadowed ? openColors[c] : visibleColors[c];
                }
                if (d.Valid != bounce.Valid || u.Valid != bounce.Valid || (background && openSum != 0))
                    throw std::runtime_error("Additional light did not preserve background validity.");
                if (d.R+d.G+d.B > 0) ++additionalLit;
                visibilityView = openSum > 0 ? visibleSum/openSum : 0;
            }
            for (size_t c = 0; c < 3; ++c)
            {
                const double direct = (double(albedo[i*4+c])/255)*colors[c]*cosine*light.Intensity*visibility/3.14159265358979323846+extra[c];
                double expectedHDR = background ? (view == HybridView::Indirect ? 0 : bg[c]) :
                    view == HybridView::Shadow ? visibilityView : view == HybridView::Direct ? direct :
                    view == HybridView::Indirect ? gi[c] : direct+gi[c];
                const double actualHDR = PackedVector::XMConvertHalfToFloat(hdr[i*4+c]);
                const double error = std::abs(expectedHDR-actualHDR);
                maxHDRError = std::max(maxHDRError,error);
                maxHDR = std::max(maxHDR,actualHDR);
                if (!std::isfinite(actualHDR) || error > 0.002*std::max(1.0,expectedHDR))
                    throw std::runtime_error("Composite HDR differs from independent CPU lighting at frame " + std::to_string(frame)+" pixel "+std::to_string(i));
                double expectedDisplay = 0;
                if (view >= HybridView::Albedo && view <= HybridView::Depth)
                {
                    if (background) expectedDisplay = bg[c];
                    else if (view == HybridView::Albedo) expectedDisplay = double(albedo[i*4+c])/255;
                    else if (view == HybridView::Normal) expectedDisplay = rawN[c]*0.5+0.5;
                    else expectedDisplay = std::clamp((1-double(depth[i]))*20,0.0,1.0);
                }
                else if (view == HybridView::Shadow) expectedDisplay = actualHDR;
                else
                {
                    const double exposed = actualHDR*exposure;
                    const double mapped = exposed/(1+exposed);
                    if (mapped <= 0.0031308) ++lowSRGB;
                    expectedDisplay = SRGB(mapped);
                }
                const int expectedByte = static_cast<int>(std::lround(std::clamp(expectedDisplay,0.0,1.0)*255));
                const int byteError = std::abs(int(display[i*4+c])-expectedByte);
                maxByteError = std::max(maxByteError,byteError);
                if (byteError > 1) throw std::runtime_error("Display color/gamma differs from CPU reference at frame "+std::to_string(frame)+" pixel "+std::to_string(i));
            }
            if (hdr[i*4+3] != 0x3c00 || display[i*4+3] != 255) throw std::runtime_error("Output alpha must be 1.");
        }
        if (checkFixture)
        {
            if ((indirectLit != 0) != (maxIndirectBounces != 0))
                throw std::runtime_error("Indirect illumination disagrees with the enabled bounce limit.");
            if (additional && !additionalLit) throw std::runtime_error("The scene received no additional direct light.");
            if (!darkened) throw std::runtime_error("Fixture has no directly lit surface that should be darkened by a shadow.");
            if (frame == 6 && maxHDR <= 1) throw std::runtime_error("Bright fixture did not exercise HDR values above 1.");
            if (frame == 7 && lowSRGB == 0) throw std::runtime_error("Dim fixture did not exercise linear sRGB segment.");
        }
        std::cout << "PASS: Composite/display frame " << frame << " view=" << static_cast<UINT>(view)
            << ": ALL " << static_cast<size_t>(width)*height << " pixels; max HDR=" << maxHDR
            << ", HDR error=" << maxHDRError << ", display error=" << maxByteError
            << " byte; nonzero-indirect pixels=" << indirectLit << "; shadow-darkened candidates=" << darkened << ", low-sRGB channels=" << lowSRGB << ".\n";
    }
    void CheckMessages(ID3D12Device *device)
    {
        ComPtr<ID3D12InfoQueue> messages;
        CheckHR(device->QueryInterface(IID_PPV_ARGS(&messages)),"QueryInterface(info queue)");
        for (UINT64 i = 0; i < messages->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i)
        {
            SIZE_T bytes = 0;
            CheckHR(messages->GetMessage(i,nullptr,&bytes),"GetMessage(size)");
            std::vector<unsigned char> data(bytes);
            auto *message = reinterpret_cast<D3D12_MESSAGE *>(data.data());
            CheckHR(messages->GetMessage(i,message,&bytes),"GetMessage");
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) throw std::runtime_error(message->pDescription);
        }
    }
    void Run(bool fullResolution, bool captures, bool profile, bool linearComparison, UINT maxIndirectBounces,
        const std::filesystem::path &sceneFile)
    {
        DX12Context dx;
        dx.InitializeHeadless();
        auto *device = dx.GetDevice();
        auto *queue = dx.GetCommandQueue();
        Scene scene;
        std::optional<SceneCamera> sceneCamera;
        if (sceneFile.empty()) scene.Initialize(device);
        else
        {
            const auto description = ReadSceneFile(sceneFile);
            description.Apply(scene,device);
            sceneCamera = description.Camera;
        }
        const UINT width = sceneCamera ? sceneCamera->Width : fullResolution ? 1280 : 321;
        const UINT height = sceneCamera ? sceneCamera->Height : fullResolution ? 720 : 181;
        const bool additional = !scene.GetAdditionalLights().empty();
        HybridFramePipeline pipeline;
        pipeline.Initialize(device,queue,DXGI_FORMAT_R8G8B8A8_UNORM,width,height,2,scene,profile,additional && !linearComparison);
        D3D12_HEAP_PROPERTIES heap = {};
        heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE clear = {};
        clear.Format = desc.Format;
        clear.Color[0] = 0.08f;
        clear.Color[1] = 0.12f;
        clear.Color[2] = 0.18f;
        clear.Color[3] = 1.0f;
        ComPtr<ID3D12Resource> output;
        CheckHR(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_RENDER_TARGET,
            &clear,IID_PPV_ARGS(&output)),"CreateCommittedResource(output)");
        D3D12_DESCRIPTOR_HEAP_DESC rtvDesc = {};
        rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtvDesc.NumDescriptors = 1;
        ComPtr<ID3D12DescriptorHeap> rtvHeap;
        CheckHR(device->CreateDescriptorHeap(&rtvDesc,IID_PPV_ARGS(&rtvHeap)),"CreateDescriptorHeap(output)");
        const auto rtv = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(output.Get(),nullptr,rtv);
        std::array<ComPtr<ID3D12CommandAllocator>,2> allocators;
        std::array<ComPtr<ID3D12GraphicsCommandList>,2> lists;
        for (size_t i = 0; i < lists.size(); ++i)
        {
            CheckHR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocators[i])),"CreateCommandAllocator");
            CheckHR(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocators[i].Get(),nullptr,IID_PPV_ARGS(&lists[i])),"CreateCommandList");
            CheckHR(lists[i]->Close(),"Close(initial)");
        }
        std::array<TextureReadback,5> readbacks;
        ID3D12Resource *textures[] = {pipeline.GetAlbedo(),pipeline.GetNormal(),pipeline.GetDepth(),pipeline.GetHDR(),output.Get()};
        for (size_t i = 0; i < readbacks.size(); ++i) readbacks[i].Initialize(device,textures[i]);
        const size_t pixels = static_cast<size_t>(width)*height;
        const size_t shadowBytes = pixels*sizeof(ShadowResult);
        auto shadowReadback = Buffer(device,shadowBytes);
        const size_t lightingRecords = pixels*(additional ? AdditionalLightingPlanes : 1u);
        const size_t indirectBytes = lightingRecords*sizeof(IndirectLighting);
        auto indirectReadback = Buffer(device,indirectBytes);
        std::vector<IndirectLighting> cudaIndirect(lightingRecords), dxIndirect(lightingRecords);
        std::vector<ShadowResult> cudaShadow(pixels), dxShadow(pixels);
        std::vector<SurfaceInput> surfaces(pixels);
        std::vector<uint8_t> firstLit, firstUnshadowed;
        Camera camera;
        const HybridView views[] = {HybridView::Lit,HybridView::Unshadowed,HybridView::Shadow,HybridView::Albedo,
            HybridView::Normal,HybridView::Depth,HybridView::Lit,HybridView::Lit,HybridView::Direct,HybridView::Indirect};
        constexpr UINT seed = 12345;
        const UINT spp = captures || linearComparison ? 32u : 4u;
        const bool captureAndDebugViews = !linearComparison || captures;
        const bool checkFixture = captureAndDebugViews && !sceneCamera;
        const UINT frameCount = static_cast<UINT>(std::size(views));
        for (UINT frame = captureAndDebugViews ? 0 : frameCount-1; frame < frameCount; ++frame)
        {
            const bool alternate = !sceneCamera && frame >= 3 && frame <= 7 && frame%2;
            if (sceneCamera) sceneCamera->Apply(camera,width,height);
            else
            {
                camera.SetLens(XM_PIDIV4,float(width)/height,alternate ? 0.25f : 0.1f,alternate ? 40.0f : 100.0f);
                camera.LookAt({alternate ? 0.45f : -0.45f,2.6f,-7.0f},{0,0,0.5f},{0,1,0});
            }
            auto light = scene.GetDirectionalLight();
            if (alternate) light.Direction = {0.7f,1.0f,0.2f};
            if (!sceneCamera && frame == 6) { light.Intensity = 4*XM_PI; light.Color = {1.2f,0.5f,0.3f}; }
            if (!sceneCamera && frame == 7) { light.Intensity = 0.001f; }
            const float exposure = sceneCamera ? 1.0f : frame == 6 ? 0.35f : frame == 7 ? 2.0f : 1.0f;
            for (size_t i = 0; i < lists.size(); ++i)
            {
                CheckHR(allocators[i]->Reset(),"Reset(allocator)");
                CheckHR(lists[i]->Reset(allocators[i].Get(),nullptr),"Reset(list)");
            }
            pipeline.RecordRaster(lists[0].Get(),rtv,frame%2,camera,scene);
            CheckHR(lists[0]->Close(),"Close(raster)");
            ID3D12CommandList *raster[] = {lists[0].Get()};
            queue->ExecuteCommandLists(1,raster);
            pipeline.EnqueueTrace(camera,light,spp,seed,maxIndirectBounces);
            pipeline.RecordComposite(lists[1].Get(),rtv,views[frame],exposure);
            lists[1]->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
            for (auto &readback : readbacks) readback.Copy(lists[1].Get());
            Transition(lists[1].Get(),pipeline.GetSharedShadow(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
            lists[1]->CopyBufferRegion(shadowReadback.Get(),0,pipeline.GetSharedShadow(),0,shadowBytes);
            Transition(lists[1].Get(),pipeline.GetSharedShadow(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
            Transition(lists[1].Get(),pipeline.GetSharedIndirect(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
            lists[1]->CopyBufferRegion(indirectReadback.Get(),0,pipeline.GetSharedIndirect(),0,indirectBytes);
            Transition(lists[1].Get(),pipeline.GetSharedIndirect(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
            CheckHR(lists[1]->Close(),"Close(composite/readback)");
            ID3D12CommandList *compose[] = {lists[1].Get()};
            queue->ExecuteCommandLists(1,compose);
            pipeline.WaitForIdle();
            if (profile)
            {
                pipeline.CollectTimings();
                const auto timing=pipeline.TakeCompletedTiming();
                if (!timing || timing->Frame!=UINT64(frame)+1 || timing->Spp!=spp ||
                    timing->View!=static_cast<UINT>(views[frame]) || timing->Width!=width || timing->Height!=height)
                    throw std::runtime_error("Missing/stale profiling frame or wrong metadata.");
                const auto &r=*timing;
                if (r.At(FrameMetric::Raster)<=0 || r.At(FrameMetric::OptixTrace)<=0 || r.At(FrameMetric::GpuSpan)<=0 ||
                    std::abs(r.At(FrameMetric::SurfaceInput)-r.At(FrameMetric::SurfaceCopies)-r.At(FrameMetric::SurfaceKernels))>0.00001 ||
                    r.At(FrameMetric::GpuSpan)+1e-8 < r.At(FrameMetric::Raster)+r.At(FrameMetric::Composite)+r.At(FrameMetric::ToneMap)+r.At(FrameMetric::DebugView))
                    throw std::runtime_error("GPU timestamp durations/partitions are inconsistent.");
                std::cout<<"PASS: timing frame "<<r.Frame<<": GPU span="<<r.At(FrameMetric::GpuSpan)
                    <<" ms, trace="<<r.At(FrameMetric::OptixTrace)<<" ms; timestamps and metadata valid.\n";
            }
            CheckCuda(cuCtxSetCurrent(pipeline.GetCudaContext()),"cuCtxSetCurrent(validation)");
            CheckCuda(cuMemcpyDtoH(cudaShadow.data(),pipeline.GetDeviceShadow(),shadowBytes),"cuMemcpyDtoH(shadow diagnostic)");
            CheckCuda(cuMemcpyDtoH(surfaces.data(),pipeline.GetDeviceSurfaces(),pixels*sizeof(SurfaceInput)),"cuMemcpyDtoH(surface diagnostic)");
            const D3D12_RANGE range = {0,shadowBytes}, none = {0,0};
            void *mapped = nullptr;
            CheckHR(shadowReadback->Map(0,&range,&mapped),"Map(shared shadow readback)");
            std::memcpy(dxShadow.data(),mapped,shadowBytes);
            shadowReadback->Unmap(0,&none);
            if (std::memcmp(dxShadow.data(),cudaShadow.data(),shadowBytes) != 0)
                throw std::runtime_error("CUDA->DX12 shadow bytes differ.");
            CheckCuda(cuMemcpyDtoH(cudaIndirect.data(),pipeline.GetDeviceIndirect(),indirectBytes),"cuMemcpyDtoH(indirect diagnostic)");
            const D3D12_RANGE indirectRange = {0,indirectBytes};
            CheckHR(indirectReadback->Map(0,&indirectRange,&mapped),"Map(shared indirect readback)");
            std::memcpy(dxIndirect.data(),mapped,indirectBytes);
            indirectReadback->Unmap(0,&none);
            if (std::memcmp(dxIndirect.data(),cudaIndirect.data(),indirectBytes) != 0)
                throw std::runtime_error("CUDA->DX12 indirect bytes differ.");
            ValidateShadowResults(dxShadow.data(),surfaces.data(),scene,pipeline.GetShadowSettings(),width,height,frame,nullptr,!sceneCamera);
            const auto albedo = readbacks[0].Read<uint8_t>();
            const auto normal = readbacks[1].Read<uint16_t>();
            const auto depth = readbacks[2].Read<float>();
            const auto hdr = readbacks[3].Read<uint16_t>();
            const auto display = readbacks[4].Read<uint8_t>();
            ValidateColor(albedo,normal,depth,hdr,display,dxShadow,dxIndirect,pipeline.GetFrameLight(),
                views[frame],exposure,width,height,frame,maxIndirectBounces,checkFixture);
            if (frame == 0) firstLit = display;
            if (frame == 1) firstUnshadowed = display;
            if (captures && (frame < 3 || frame >= 8))
            {
                const wchar_t *names[] = {L"HybridGI.bmp",L"HybridUnshadowed.bmp",L"HybridVisibility.bmp",L"HybridDirect.bmp",L"HybridIndirect.bmp"};
                SaveBMP(std::filesystem::current_path()/names[frame < 3 ? frame : frame-5],display,width,height);
            }
            if (linearComparison && frame+1 == std::size(views))
            {
                // The final frame is Indirect. Reuse its raster and OptiX outputs for both draws.
                const auto captureView = [&](HybridView view)
                {
                    CheckHR(allocators[1]->Reset(),"Reset(linear capture allocator)");
                    auto *list = lists[1].Get();
                    CheckHR(list->Reset(allocators[1].Get(),nullptr),"Reset(linear capture list)");
                    for (auto *source : {pipeline.GetAlbedo(),pipeline.GetNormal(),pipeline.GetDepth()})
                        Transition(list,source,D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_COMMON);
                    pipeline.RecordComposite(list,rtv,view,exposure);
                    list->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
                    readbacks[3].Copy(list);
                    CheckHR(list->Close(),"Close(linear capture)");
                    ID3D12CommandList *commands[] = {list};
                    queue->ExecuteCommandLists(1,commands);
                    pipeline.WaitForIdle();
                    return readbacks[3].Read<uint16_t>();
                };
                const auto direct = captureView(HybridView::Direct);
                const auto both = captureView(HybridView::Lit);
                CheckMessages(device);
                SaveLinearComparison(width,height,spp,seed,depth,direct,hdr,both);
                SaveComparisonScene(width,height,spp,seed,maxIndirectBounces,scene,camera,pipeline.GetFrameLight(),depth,albedo);
                std::cout << "Saved CompositeLinear.bin and CompositeScene.json for the same frame.\n";
            }
        }
        if (!checkFixture)
        {
            pipeline.ValidateGAS(scene);
            CheckMessages(device);
            return;
        }
        size_t changed = 0;
        for (size_t i = 0; i < firstLit.size(); i += 4)
        {
            bool darker = false;
            for (size_t c = 0; c < 3; ++c)
            {
                if (firstLit[i+c] > firstUnshadowed[i+c]) throw std::runtime_error("Applying visibility brightened a pixel.");
                darker = darker || firstLit[i+c] < firstUnshadowed[i+c];
            }
            if (darker) ++changed;
        }
        if (!changed) throw std::runtime_error("Shadowed/unshadowed displays are identical.");
        pipeline.ValidateGAS(scene);
        CheckMessages(device);
        std::cout << "PASS: " << changed << " pixels darkened versus identical-camera unshadowed capture; "
            << "CUDA->DX12 shadow + indirect bytes exact, HDR/composite/display and all eight views checked; D3D12 validation clean.\n";
    }
}

int wmain(int argc, wchar_t **argv)
{
    try
    {
        bool full = false, captures = false, profile = false, linearComparison = false;
        UINT maxIndirectBounces = DefaultIndirectBounces;
        std::filesystem::path sceneFile;
        for (int i = 1; i < argc; ++i)
        {
            const std::wstring argument = argv[i];
            if (argument == L"--full-resolution") full = true;
            else if (argument == L"--captures") captures = true;
            else if (argument == L"--profile") profile = true;
            else if (argument == L"--linear-comparison") linearComparison = true;
            else if (argument == L"--bounces") maxIndirectBounces = ParseIndirectBounceCount(++i < argc ? argv[i] : nullptr);
            else if (argument == L"--scene")
            {
                if (++i >= argc || !*argv[i] || argv[i][0]==L'-' || !sceneFile.empty())
                    throw std::runtime_error("--scene expects one scene file path.");
                sceneFile=argv[i];
            }
            else throw std::runtime_error("Usage: HybridTracerCompositeTest [--scene path.xml] [--full-resolution] [--captures] [--profile | --linear-comparison] [--bounces 0..8]");
        }
        if (profile && linearComparison)
            throw std::runtime_error("Linear comparison and profiling require separate runs.");
        ComPtr<ID3D12Debug> debug;
        CheckHR(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),"D3D12GetDebugInterface");
        debug->EnableDebugLayer();
        Run(full,captures,profile,linearComparison,maxIndirectBounces,sceneFile);
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

