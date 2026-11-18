#include "Renderer/HybridFramePipeline.h"
#include "Renderer/IndirectBounceOptions.h"
#include "Renderer/DX12Context.h"
#include "Scene/Camera.h"
#include "Scene/SceneFile.h"
#include <d3d12sdklayers.h>
#include <algorithm>
#include <cmath>
#include <numeric>
using namespace HybridInterop;
using namespace DirectX;
namespace
{
    struct Options
    {
        UINT Width=1280,Height=720,Frames=120,Warmup=30,Spp=4;
        UINT MaxIndirectBounces=DefaultIndirectBounces;
        std::filesystem::path SceneFile;
        bool Check=false,Moving=false,Profile=true,Accumulation=true;
        std::filesystem::path Csv=L"GpuTimings.csv";
    };
    UINT Number(const wchar_t *text,UINT maximum)
    {
        size_t used=0; const auto value=std::stoul(text,&used);
        if (used!=std::wcslen(text) || value>maximum) throw std::runtime_error("Invalid benchmark integer.");
        return static_cast<UINT>(value);
    }
    Options Parse(int argc,wchar_t **argv)
    {
        Options o;
        for (int i=1;i<argc;++i)
        {
            const std::wstring flag=argv[i];
            if (flag==L"--check") { o.Check=true; continue; }
            if (flag==L"--moving-camera") { o.Moving=true; continue; }
            if (flag==L"--no-accumulation") { o.Accumulation=false; continue; }
            if (flag==L"--no-profile") { o.Profile=false; continue; }
            if (++i>=argc) throw std::runtime_error("Benchmark flag needs a value.");
            if (flag==L"--width") o.Width=Number(argv[i],4096);
            else if (flag==L"--height") o.Height=Number(argv[i],4096);
            else if (flag==L"--frames") o.Frames=Number(argv[i],100000);
            else if (flag==L"--warmup") o.Warmup=Number(argv[i],10000);
            else if (flag==L"--spp") o.Spp=Number(argv[i],MaxIndirectSamples);
            else if (flag==L"--bounces") o.MaxIndirectBounces=ParseIndirectBounceCount(argv[i]);
            else if (flag==L"--scene")
            {
                if (!*argv[i] || argv[i][0]==L'-' || !o.SceneFile.empty())
                    throw std::runtime_error("--scene expects one scene file path.");
                o.SceneFile=argv[i];
            }
            else if (flag==L"--csv") o.Csv=argv[i];
            else throw std::runtime_error("Unknown benchmark flag.");
        }
        if (!o.Width || !o.Height || !o.Frames || !o.Spp) throw std::runtime_error("Dimensions, frames and spp must be positive.");
        return o;
    }
    void CheckMessages(ID3D12Device *device)
    {
        ComPtr<ID3D12InfoQueue> messages;
        CheckHR(device->QueryInterface(IID_PPV_ARGS(&messages)),"QueryInterface(benchmark validation)");
        for (UINT64 i=0;i<messages->GetNumStoredMessagesAllowedByRetrievalFilter();++i)
        {
            SIZE_T bytes=0; CheckHR(messages->GetMessage(i,nullptr,&bytes),"GetMessage(size)");
            std::vector<unsigned char> storage(bytes); auto *message=reinterpret_cast<D3D12_MESSAGE *>(storage.data());
            CheckHR(messages->GetMessage(i,message,&bytes),"GetMessage(benchmark)");
            if (message->Severity<=D3D12_MESSAGE_SEVERITY_WARNING) throw std::runtime_error(message->pDescription);
        }
    }
    void Run(const Options &o)
    {
        const auto scenePath = o.SceneFile.empty() ? DefaultSceneFile() : o.SceneFile;
        const auto description = ReadSceneFile(scenePath);
        std::cout << "Scene: " << scenePath.u8string() << " (" << description.Objects.size() << " objects, "
            << description.Lights.size() << " lights)\n";
        DX12Context dx; dx.InitializeHeadless(); auto *device=dx.GetDevice(); auto *queue=dx.GetCommandQueue();
        Scene scene;
        description.Apply(scene,device);
        HybridFramePipeline pipeline;
        pipeline.Initialize(device,queue,DXGI_FORMAT_R8G8B8A8_UNORM,o.Width,o.Height,2,scene,o.Profile,o.Accumulation);
        D3D12_HEAP_PROPERTIES heap={}; heap.Type=D3D12_HEAP_TYPE_DEFAULT; heap.CreationNodeMask=heap.VisibleNodeMask=1;
        D3D12_RESOURCE_DESC desc={}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width=o.Width; desc.Height=o.Height;
        desc.DepthOrArraySize=desc.MipLevels=1; desc.SampleDesc.Count=1; desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE clear={}; clear.Format=desc.Format;
        clear.Color[0]=0.08f; clear.Color[1]=0.12f; clear.Color[2]=0.18f; clear.Color[3]=1;
        ComPtr<ID3D12Resource> output;
        CheckHR(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_RENDER_TARGET,
            &clear,IID_PPV_ARGS(&output)),"CreateCommittedResource(benchmark target)");
        D3D12_DESCRIPTOR_HEAP_DESC rtvDescription={}; rtvDescription.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV; rtvDescription.NumDescriptors=1;
        ComPtr<ID3D12DescriptorHeap> rtvHeap;
        CheckHR(device->CreateDescriptorHeap(&rtvDescription,IID_PPV_ARGS(&rtvHeap)),"CreateDescriptorHeap(benchmark)");
        const auto rtv=rtvHeap->GetCPUDescriptorHandleForHeapStart(); device->CreateRenderTargetView(output.Get(),nullptr,rtv);
        std::array<ComPtr<ID3D12CommandAllocator>,2> allocators;
        std::array<ComPtr<ID3D12GraphicsCommandList>,2> lists;
        for (size_t i=0;i<2;++i)
        {
            CheckHR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocators[i])),"CreateCommandAllocator(benchmark)");
            CheckHR(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocators[i].Get(),nullptr,IID_PPV_ARGS(&lists[i])),"CreateCommandList(benchmark)");
            CheckHR(lists[i]->Close(),"Close(initial benchmark)");
        }
        Camera camera; camera.SetLens(XM_PIDIV4,float(o.Width)/o.Height,0.1f,100);
        std::unique_ptr<FrameTimingLog> log;
        if (o.Profile) log=std::make_unique<FrameTimingLog>(o.Csv);
        CUdevice cudaDevice; CheckCuda(cuCtxGetDevice(&cudaDevice),"cuCtxGetDevice(benchmark)");
        char gpu[256]={}; CheckCuda(cuDeviceGetName(gpu,sizeof(gpu),cudaDevice),"cuDeviceGetName(benchmark)");
        int driver=0; CheckCuda(cuDriverGetVersion(&driver),"cuDriverGetVersion(benchmark)");
#ifdef NDEBUG
        const char *configuration="Release";
#else
        const char *configuration="Debug";
#endif
        std::cout<<"BENCHMARK: "<<gpu<<"; CUDA driver API="<<driver<<"; "<<configuration
            <<"; OptiX validation=ALL; D3D debug="<<o.Check<<"; headless/no Present; profiling="<<o.Profile
            <<"; accumulation="<<o.Accumulation<<"; max_indirect_bounces="<<o.MaxIndirectBounces
            <<"; warmup="<<o.Warmup<<", samples="<<o.Frames<<", camera="<<(o.Moving ? "moving":"static")<<".\n";
        std::vector<double> wall;
        ProfileClock::time_point measuredBegin;
        for (UINT frame=0;frame<o.Warmup+o.Frames;++frame)
        {
            if (frame==o.Warmup) measuredBegin=ProfileClock::now();
            camera.LookAt({-0.45f+(o.Moving ? 0.3f*std::sin(frame*0.025f):0),2.6f,-7},{0,0,0.5f},{0,1,0});
            const auto begin=ProfileClock::now();
            for (size_t i=0;i<2;++i)
            {
                CheckHR(allocators[i]->Reset(),"Reset(benchmark allocator)");
                CheckHR(lists[i]->Reset(allocators[i].Get(),nullptr),"Reset(benchmark list)");
            }
            pipeline.RecordRaster(lists[0].Get(),rtv,frame%2,camera,scene);
            CheckHR(lists[0]->Close(),"Close(benchmark raster)");
            ID3D12CommandList *raster[]={lists[0].Get()}; queue->ExecuteCommandLists(1,raster);
            pipeline.EnqueueTrace(camera,scene.GetDirectionalLight(),o.Spp,12345+frame,o.MaxIndirectBounces);
            pipeline.RecordComposite(lists[1].Get(),rtv,HybridView::Lit);
            CheckHR(lists[1]->Close(),"Close(benchmark composite)");
            ID3D12CommandList *composite[]={lists[1].Get()}; queue->ExecuteCommandLists(1,composite);
            const double submitMs=ProfileMilliseconds(begin);
            const auto wait=ProfileClock::now(); pipeline.WaitForIdle(); const double waitMs=ProfileMilliseconds(wait);
            const double wallMs=ProfileMilliseconds(begin);
            pipeline.SetCpuTiming(submitMs,0,waitMs); pipeline.CollectTimings();
            auto timing=pipeline.TakeCompletedTiming();
            const auto &history=pipeline.GetAccumulationInfo();
            const UINT expectedHistory=o.Accumulation ? (o.Moving ? 1u:frame+1):0u;
            if (history.Frames!=expectedHistory || history.SamplesPerPixel!=UINT64(expectedHistory)*o.Spp)
                throw std::runtime_error("Benchmark accumulation state/reset is incorrect.");
            if (o.Profile)
            {
                if (!timing || timing->Frame!=UINT64(frame)+1 || timing->Spp!=o.Spp || timing->View!=0 ||
                    timing->HistoryFrames!=expectedHistory || timing->HistorySamples!=UINT64(expectedHistory)*o.Spp ||
                    timing->At(FrameMetric::OptixTrace)<=0 || timing->At(FrameMetric::GpuSpan)<=0)
                    throw std::runtime_error("Benchmark missing valid frame timing/metadata.");
                if (frame>=o.Warmup) log->Add(*timing);
            }
            if (frame>=o.Warmup) wall.push_back(wallMs);
        }
        const double loopMeanMs=ProfileMilliseconds(measuredBegin)/o.Frames;
        if (log)
        {
            if (log->Size()!=o.Frames) throw std::runtime_error("Warmup filtering lost measured frames.");
            log->Summary(std::cout);
            std::cout<<"TIMING CSV: "<<o.Csv.u8string()<<'\n';
        }
        std::sort(wall.begin(),wall.end());
        const double mean=std::accumulate(wall.begin(),wall.end(),0.0)/wall.size();
        const double median=wall.size()%2 ? wall[wall.size()/2] : (wall[wall.size()/2-1]+wall[wall.size()/2])*0.5;
        std::cout<<"CPU completed-frame wall: mean="<<mean<<" ms; median="<<median
            <<" ms; p95="<<wall[static_cast<size_t>(std::ceil(0.95*wall.size()))-1]<<" ms.\n";
        std::cout<<"CPU measured-loop mean (includes timing collection/CSV): "<<loopMeanMs<<" ms/frame.\n";
        pipeline.ValidateGAS(scene); if (o.Check) CheckMessages(device);
        std::cout<<"PASS: benchmark completed; timestamp readback only, no image readback.\n";
    }
}
int wmain(int argc,wchar_t **argv)
{
    try
    {
        const auto options=Parse(argc,argv);
        if (options.Check) { ComPtr<ID3D12Debug> debug; CheckHR(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),"D3D12GetDebugInterface"); debug->EnableDebugLayer(); }
        Run(options); return 0;
    }
    catch (const std::exception &error) { std::cerr<<"FAIL: "<<error.what()<<'\n'; return 1; }
}
