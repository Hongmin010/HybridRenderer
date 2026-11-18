#include "Renderer/HybridFramePipeline.h"
#include "Renderer/DX12Context.h"
#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include <d3d12sdklayers.h>
#include <DirectXPackedVector.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>
using namespace HybridInterop;
using namespace DirectX;
namespace
{
    void Require(bool ok,const char *message) { if (!ok) throw std::runtime_error(message); }
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

    struct FrameFixture
    {
        UINT Width,Height,Frame=0;
        DX12Context DX;
        Scene World;
        HybridFramePipeline Pipeline;
        ComPtr<ID3D12Resource> Output,SharedRead;
        ComPtr<ID3D12DescriptorHeap> RTV;
        std::array<ComPtr<ID3D12CommandAllocator>,2> Allocators;
        std::array<ComPtr<ID3D12GraphicsCommandList>,2> Lists;
        TextureReadback HDRRead,DisplayRead;
        std::vector<IndirectLighting> Raw,Mean,Shared;
        std::vector<uint16_t> HDR;
        std::vector<uint8_t> Display;
        std::vector<unsigned int> Counts;
        FrameTimingLog Timings;
        FrameFixture(UINT width,UINT height):Width(width),Height(height),Timings(L"TemporalFrames.csv")
        {
            DX.InitializeHeadless(); auto *device=DX.GetDevice(); World.Initialize(device);
            Pipeline.Initialize(device,DX.GetCommandQueue(),DXGI_FORMAT_R8G8B8A8_UNORM,width,height,2,World,true,true);
            D3D12_HEAP_PROPERTIES heap={}; heap.Type=D3D12_HEAP_TYPE_DEFAULT;
            D3D12_RESOURCE_DESC desc={}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width=width; desc.Height=height; desc.DepthOrArraySize=desc.MipLevels=1;
            desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM; desc.SampleDesc.Count=1; desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            D3D12_CLEAR_VALUE clear={}; clear.Format=desc.Format;
            clear.Color[0]=0.08f; clear.Color[1]=0.12f; clear.Color[2]=0.18f; clear.Color[3]=1;
            CheckHR(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_RENDER_TARGET,&clear,IID_PPV_ARGS(&Output)),"CreateCommittedResource(temporal frame)");
            D3D12_DESCRIPTOR_HEAP_DESC rtv={}; rtv.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV; rtv.NumDescriptors=1;
            CheckHR(device->CreateDescriptorHeap(&rtv,IID_PPV_ARGS(&RTV)),"CreateDescriptorHeap(temporal frame)");
            device->CreateRenderTargetView(Output.Get(),nullptr,RTV->GetCPUDescriptorHandleForHeapStart());
            for (size_t i=0;i<2;++i)
            {
                CheckHR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&Allocators[i])),"CreateCommandAllocator(temporal frame)");
                CheckHR(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,Allocators[i].Get(),nullptr,IID_PPV_ARGS(&Lists[i])),"CreateCommandList(temporal frame)");
                CheckHR(Lists[i]->Close(),"Close(initial temporal frame)");
            }
            const size_t pixels=size_t(width)*height;
            Raw.resize(pixels); Mean.resize(pixels); Shared.resize(pixels); Counts.resize(pixels);
            SharedRead=Buffer(device,pixels*sizeof(IndirectLighting));
            HDRRead.Initialize(device,Pipeline.GetHDR()); DisplayRead.Initialize(device,Output.Get());
        }
        void Render(const Camera &camera,const DirectionalLight &light,UINT spp,UINT seed,
            bool read=false,HybridView view=HybridView::Indirect,float exposure=1,
            UINT maxIndirectBounces=DefaultIndirectBounces)
        {
            for (size_t i=0;i<2;++i)
            {
                CheckHR(Allocators[i]->Reset(),"Reset(temporal frame allocator)");
                CheckHR(Lists[i]->Reset(Allocators[i].Get(),nullptr),"Reset(temporal frame list)");
            }
            const auto rtv=RTV->GetCPUDescriptorHandleForHeapStart();
            Pipeline.RecordRaster(Lists[0].Get(),rtv,Frame%2,camera,World);
            CheckHR(Lists[0]->Close(),"Close(temporal raster)");
            ID3D12CommandList *raster[]={Lists[0].Get()}; DX.GetCommandQueue()->ExecuteCommandLists(1,raster);
            Pipeline.EnqueueTrace(camera,light,spp,seed,maxIndirectBounces);
            Pipeline.RecordComposite(Lists[1].Get(),rtv,view,exposure);
            if (read)
            {
                Lists[1]->OMSetRenderTargets(0,nullptr,FALSE,nullptr);
                HDRRead.Copy(Lists[1].Get()); DisplayRead.Copy(Lists[1].Get());
                Transition(Lists[1].Get(),Pipeline.GetSharedIndirect(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE);
                Lists[1]->CopyBufferRegion(SharedRead.Get(),0,Pipeline.GetSharedIndirect(),0,Raw.size()*sizeof(IndirectLighting));
                Transition(Lists[1].Get(),Pipeline.GetSharedIndirect(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON);
            }
            CheckHR(Lists[1]->Close(),"Close(temporal composite)");
            ID3D12CommandList *composite[]={Lists[1].Get()}; DX.GetCommandQueue()->ExecuteCommandLists(1,composite);
            Pipeline.WaitForIdle(); Pipeline.CollectTimings();
            const auto timing=Pipeline.TakeCompletedTiming(); const auto &info=Pipeline.GetAccumulationInfo();
            Require(timing && timing->Frame==UINT64(Frame)+1 && timing->HistoryFrames==info.Frames &&
                timing->HistorySamples==info.SamplesPerPixel && timing->HistoryResetMask==info.ResetMask &&
                timing->AccumulationEnabled==(info.Enabled ? 1u:0u),"Stale history profiling metadata.");
            if (info.Enabled) Require(timing->At(FrameMetric::Accumulation)>0,"Accumulation event span missing.");
            Timings.Add(*timing); ++Frame;
            if (!read) return;
            const size_t bytes=Raw.size()*sizeof(IndirectLighting);
            CheckCuda(cuCtxSetCurrent(Pipeline.GetCudaContext()),"cuCtxSetCurrent(temporal validation)");
            CheckCuda(cuMemcpyDtoH(Raw.data(),Pipeline.GetDeviceRawIndirect(),bytes),"cuMemcpyDtoH(raw indirect)");
            CheckCuda(cuMemcpyDtoH(Mean.data(),Pipeline.GetDeviceIndirect(),bytes),"cuMemcpyDtoH(accumulated indirect)");
            if (info.Enabled) CheckCuda(cuMemcpyDtoH(Counts.data(),Pipeline.GetHistoryPixelCounts(),Counts.size()*sizeof(unsigned int)),"cuMemcpyDtoH(temporal counts)");
            const D3D12_RANGE range={0,bytes},none={0,0}; void *mapped=nullptr;
            CheckHR(SharedRead->Map(0,&range,&mapped),"Map(temporal shared readback)");
            std::memcpy(Shared.data(),mapped,bytes); SharedRead->Unmap(0,&none);
            Require(std::memcmp(Shared.data(),Mean.data(),bytes)==0,"Accumulated CUDA->DX12 bytes differ.");
            if (!info.Enabled || info.Frames==1)
                Require(std::memcmp(Raw.data(),Mean.data(),bytes)==0,"Reset/disabled output must equal this frame's raw estimate.");
            HDR=HDRRead.Read<uint16_t>(); Display=DisplayRead.Read<uint8_t>();
            for (size_t i=0;i<Mean.size();++i)
            {
                const auto &v=Mean[i]; const float gi[]={v.R,v.G,v.B};
                Require(v.Valid==Raw[i].Valid,"Accumulation changed surface validity.");
                Require(!info.Enabled || Counts[i]==(v.Valid ? info.Frames:0u),"Wrong static-scene per-pixel count.");
                for (size_t c=0;c<3;++c)
                {
                    const double actual=PackedVector::XMConvertHalfToFloat(HDR[i*4+c]);
                    Require(std::isfinite(actual) && actual>=0,"Invalid temporal HDR color.");
                    if (view==HybridView::Indirect) Require(std::abs(actual-gi[c])<0.002*std::max(1.0,double(gi[c])),"HDR indirect view differs from linear accumulated radiance.");
                    const double x=actual*exposure/(1+actual*exposure);
                    const double srgb=x<=0.0031308 ? 12.92*x:1.055*std::pow(x,1.0/2.4)-0.055;
                    const int expected=int(std::lround(std::clamp(srgb,0.0,1.0)*255));
                    Require(std::abs(int(Display[i*4+c])-expected)<=1,"Temporal display ToneMap/sRGB mismatch.");
                }
            }
        }
        void Expect(UINT frames,HistoryReset reason=HistoryReset::None)
        {
            const auto &info=Pipeline.GetAccumulationInfo();
            Require(info.Frames==frames,"Wrong integration history frame count.");
            Require(info.ResetMask==static_cast<UINT>(reason),"Wrong integration reset reason.");
        }
    };
    double MSE(const std::vector<IndirectLighting> &image,const std::vector<IndirectLighting> &reference)
    {
        double sum=0;size_t channels=0;
        for (size_t i=0;i<image.size();++i) if (reference[i].Valid)
        {
            const double differences[]={double(image[i].R)-reference[i].R,double(image[i].G)-reference[i].G,double(image[i].B)-reference[i].B};
            for (double d:differences) { sum+=d*d; ++channels; }
        }
        Require(channels>0,"No foreground pixels for temporal error metric."); return sum/channels;
    }
    void Run(bool captures,bool full)
    {
        FrameFixture f(full ? 1280:321,full ? 720:181);
        Camera camera; camera.SetLens(XM_PIDIV4,float(f.Width)/f.Height,0.1f,100);
        camera.LookAt({-0.45f,2.6f,-7},{0,0,0.5f},{0,1,0});
        auto light=f.World.GetDirectionalLight();
        std::vector<std::array<double,3>> sum(f.Raw.size());
        std::vector<IndirectLighting> first,accumulated;
        double maxError=0;
        for (UINT frame=0;frame<64;++frame)
        {
            f.Render(camera,light,4,10000+frame,true);
            f.Expect(frame+1,frame==0 ? HistoryReset::FirstFrame:HistoryReset::None);
            for (size_t i=0;i<f.Raw.size();++i)
            {
                const float raw[]={f.Raw[i].R,f.Raw[i].G,f.Raw[i].B},mean[]={f.Mean[i].R,f.Mean[i].G,f.Mean[i].B};
                for (size_t c=0;c<3;++c)
                {
                    sum[i][c]+=raw[c]; const double error=std::abs(mean[c]-sum[i][c]/(frame+1));
                    maxError=std::max(maxError,error);
                    Require(error<0.000002,"Accumulated GI differs from CPU sum of actual raw frame estimates.");
                }
            }
            if (frame==0) { first=f.Mean; if (captures) SaveBMP(L"Indirect_1frame.bmp",f.Display,f.Width,f.Height); }
            if (frame==63) { accumulated=f.Mean; if (captures) SaveBMP(L"Indirect_64frames.bmp",f.Display,f.Width,f.Height); }
        }
        if (captures)
        {
            f.Render(camera,light,4,20000,true,HybridView::Lit);
            SaveBMP(L"Lit_65frames.bmp",f.Display,f.Width,f.Height);
        }
        // Independent seed stream, 256 frames * 16 spp = 4096 spp reference estimate.
        // Many short launches avoid a single long GPU kernel on Windows.
        for (UINT frame=0;frame<256;++frame) f.Render(camera,light,16,1000000+frame,frame==255);
        f.Expect(256);
        const auto reference=f.Mean;
        const double rawMSE=MSE(first,reference),meanMSE=MSE(accumulated,reference);
        Require(rawMSE>1e-7 && meanMSE<rawMSE*0.1,"Temporal accumulation did not sufficiently reduce noise against independent reference samples.");
        if (captures) SaveBMP(L"Indirect_Reference4096spp.bmp",f.Display,f.Width,f.Height);
        std::ofstream report("TemporalQuality.csv");
        report<<"width,height,frame_spp,accumulated_frames,accumulated_spp,reference_spp,raw_mse,accumulated_mse,rmse_ratio,max_mean_error\n"
            <<std::setprecision(12)<<f.Width<<','<<f.Height<<",4,64,256,4096,"<<rawMSE<<','<<meanMSE<<','<<std::sqrt(meanMSE/rawMSE)<<','<<maxError<<'\n';
        Require(bool(report),"Cannot write temporal quality report.");
        f.Render(camera,light,16,2000000,true,HybridView::Direct,2); f.Expect(257); // View/exposure/seed preserve history.
        camera.SetLens(XM_PIDIV4,float(f.Width)/f.Height,0.2f,100);
        f.Render(camera,light,16,2000001,true); f.Expect(1,HistoryReset::Camera);
        camera.LookAt({-0.25f,2.6f,-7},{0,0,0.5f},{0,1,0});
        f.Render(camera,light,16,2000002,true); f.Expect(1,HistoryReset::Camera);
        f.Render(camera,light,16,2000003,true); f.Expect(2);
        light.Direction={0.7f,1,0.2f};
        f.Render(camera,light,16,2000004,true); f.Expect(1,HistoryReset::Light);
        light.Intensity*=0.75f;
        f.Render(camera,light,16,2000005,true); f.Expect(1,HistoryReset::Light);
        light.Color.z=0.5f;
        f.Render(camera,light,16,2000006,true); f.Expect(1,HistoryReset::Light);
        f.Render(camera,light,8,2000007,true); f.Expect(1,HistoryReset::Sampling);
        f.Pipeline.ResetAccumulation(); f.Render(camera,light,8,2000008,true); f.Expect(1,HistoryReset::Manual);
        f.Pipeline.SetAccumulationEnabled(false); f.Render(camera,light,8,2000009,true); f.Expect(0,HistoryReset::Toggle);
        f.Render(camera,light,8,2000010,true); f.Expect(0);
        f.Pipeline.SetAccumulationEnabled(true); f.Render(camera,light,8,2000011,true); f.Expect(1,HistoryReset::Toggle);
        f.Render(camera,light,8,2000012,true); f.Expect(2);
        f.Render(camera,light,8,2000013,true,HybridView::Indirect,1,2); f.Expect(1,HistoryReset::Sampling);
        f.Render(camera,light,8,2000014,true,HybridView::Indirect,1,2); f.Expect(2);
        f.Pipeline.ValidateGAS(f.World); CheckMessages(f.DX.GetDevice());
        std::cout<<"PASS: temporal frame integration "<<f.Width<<'x'<<f.Height<<", 64x4 spp vs independent 4096 spp; raw MSE="<<rawMSE
            <<", accumulated MSE="<<meanMSE<<", RMSE ratio="<<std::sqrt(meanMSE/rawMSE)<<", max CPU mean error="<<maxError
            <<"; resets, bypass, shared bytes, HDR/sRGB, profiling metadata verified.\n";
    }
}
int main(int argc,char **argv)
{
    try
    {
        bool captures=false,full=false;
        for (int i=1;i<argc;++i)
        {
            const std::string flag=argv[i];
            if (flag=="--captures") captures=true; else if (flag=="--full-resolution") full=true;
            else throw std::runtime_error("Usage: HybridTracerTemporalFrameTest [--captures] [--full-resolution]");
        }
        ComPtr<ID3D12Debug> debug; CheckHR(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),"D3D12GetDebugInterface"); debug->EnableDebugLayer();
        Run(captures,full); return 0;
    }
    catch (const std::exception &error) { std::cerr<<"FAIL: "<<error.what()<<'\n'; return 1; }
}
