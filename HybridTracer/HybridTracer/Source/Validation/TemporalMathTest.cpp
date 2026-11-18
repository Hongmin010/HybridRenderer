#include "Renderer/TemporalAccumulationPass.h"
#include "Renderer/DX12Context.h"
#include <cmath>
#include <algorithm>
#include <vector>
using namespace HybridInterop;
namespace
{
    constexpr unsigned int Count=51;
    void Require(bool condition,const char *message) { if (!condition) throw std::runtime_error(message); }
    struct Resources
    {
        CUcontext Context=nullptr; CUstream Stream=nullptr; CUdeviceptr Input=0; IndirectLighting *Host=nullptr;
        ~Resources()
        {
            if (Context) cuCtxSetCurrent(Context);
            if (Stream) cuStreamSynchronize(Stream);
            if (Input) cuMemFree(Input); if (Host) cuMemFreeHost(Host);
            if (Stream) cuStreamDestroy(Stream); if (Context) cuCtxDestroy(Context);
        }
    };
    void Run(unsigned int planes)
    {
        DX12Context dx; dx.InitializeHeadless(); Resources cuda;
        CheckCuda(cuCtxCreate(&cuda.Context,CU_CTX_SCHED_AUTO,FindCudaDevice(dx.GetDevice())),"cuCtxCreate(temporal math)");
        CheckCuda(cuStreamCreate(&cuda.Stream,CU_STREAM_NON_BLOCKING),"cuStreamCreate(temporal math)");
        const unsigned int records=Count*planes;
        const size_t bytes=records*sizeof(IndirectLighting);
        CheckCuda(cuMemAlloc(&cuda.Input,bytes),"cuMemAlloc(temporal fixture)");
        CheckCuda(cuMemAllocHost(reinterpret_cast<void **>(&cuda.Host),bytes),"cuMemAllocHost(temporal fixture)");
        TemporalAccumulationPass pass; pass.Initialize(cuda.Context,cuda.Stream,17,3,planes);
        AccumulationState state; state.Spp=4; state.Camera[0]=state.Camera[5]=state.Camera[10]=state.Camera[15]=1;
        state.Light={0,1,0,1,1,1,3.14159265f};
        std::vector<IndirectLighting> output(records); std::vector<unsigned int> counts(Count);
        std::vector<std::array<double,3>> sums(records); std::vector<unsigned int> expectedCount(records,0);
        unsigned int epochFrames=0; double maxRelativeError=0;
        for (unsigned int frame=0;frame<140;++frame)
        {
            for (unsigned int i=0;i<Count;++i)
            {
                cuda.Host[i]={float((frame%5)*0.25+i*0.01),float((frame+1)*0.2),float(1000+i*0.125+frame*0.5),1};
                if ((i+frame)%7==0) cuda.Host[i].Valid=0; // Poison RGB must not leak from background.
                if (i==1) cuda.Host[i]={0,0,0,1}; // Black is valid and must count.
                if (i==2) cuda.Host[i].Valid=1; // Long, uninterrupted HDR history.
                for (unsigned int plane=1; plane<planes; ++plane)
                {
                    const auto &base=cuda.Host[i];
                    cuda.Host[plane*Count+i]={base.R*(plane+1),base.G/(plane+1),
                        base.B+plane*(frame+1),base.Valid};
                }
            }
            unsigned int expectedMask=frame==0 ? unsigned(HistoryReset::FirstFrame):0;
            if (frame==128) { state.Camera[31]=0.25f; expectedMask=unsigned(HistoryReset::Camera); }
            if (frame==130) { state.Light[6]=6; expectedMask=unsigned(HistoryReset::Light); }
            if (frame==132) { state.Spp=8; expectedMask=unsigned(HistoryReset::Sampling); }
            if (frame==134) { pass.Reset(); expectedMask=unsigned(HistoryReset::Manual); }
            if (frame==136) { pass.SetEnabled(false); expectedMask=unsigned(HistoryReset::Toggle); }
            if (frame==138) { pass.SetEnabled(true); expectedMask=unsigned(HistoryReset::Toggle); }
            CheckCuda(cuMemcpyHtoDAsync(cuda.Input,cuda.Host,bytes,cuda.Stream),"cuMemcpyHtoDAsync(temporal fixture)");
            pass.Execute(cuda.Input,state); CheckCuda(cuStreamSynchronize(cuda.Stream),"cuStreamSynchronize(temporal fixture)");
            const auto &info=pass.GetInfo();
            Require(info.ResetMask==expectedMask,"Wrong history reset reason.");
            CheckCuda(cuMemcpyDtoH(output.data(),pass.GetOutput(),bytes),"cuMemcpyDtoH(temporal mean)");
            if (!info.Enabled)
            {
                Require(info.Frames==0 && info.SamplesPerPixel==0 && pass.GetOutput()==cuda.Input,"Disabled accumulation must bypass raw input.");
                Require(std::memcmp(output.data(),cuda.Host,bytes)==0,"Raw bypass must be byte exact.");
                epochFrames=0; continue;
            }
            if (expectedMask) { epochFrames=0; std::fill(sums.begin(),sums.end(),std::array<double,3>{}); std::fill(expectedCount.begin(),expectedCount.end(),0); }
            ++epochFrames;
            Require(info.Frames==epochFrames && info.SamplesPerPixel==UINT64(epochFrames)*state.Spp,"Wrong progressive frame/sample count.");
            CheckCuda(cuMemcpyDtoH(counts.data(),pass.GetPixelCounts(),Count*sizeof(unsigned int)),"cuMemcpyDtoH(pixel counts)");
            for (unsigned int i=0;i<records;++i)
            {
                const auto &value=cuda.Host[i]; const float values[]={value.R,value.G,value.B};
                if (!value.Valid) { sums[i]={}; expectedCount[i]=0; }
                else { for (size_t c=0;c<3;++c) sums[i][c]+=values[c]; ++expectedCount[i]; }
                Require(counts[i%Count]==expectedCount[i] && output[i].Valid==(value.Valid ? 1u:0u),"Per-pixel validity/count mismatch.");
                const float actual[]={output[i].R,output[i].G,output[i].B};
                for (size_t c=0;c<3;++c)
                {
                    const double expected=expectedCount[i] ? sums[i][c]/expectedCount[i]:0;
                    const double error=std::abs(actual[c]-expected)/std::max(1.0,expected);
                    maxRelativeError=std::max(maxRelativeError,error);
                    Require(std::isfinite(actual[c]) && error<0.000003,"GPU progressive mean differs from independent double sum/count.");
                }
            }
        }
        bool rejected=false; state.Spp=0;
        try { pass.Execute(cuda.Input,state); } catch (const std::exception &) { rejected=true; }
        Require(rejected,"Zero spp must be rejected before dispatch.");
        std::cout<<"PASS: "<<planes<<" lighting planes, temporal arithmetic, odd dimensions, HDR, black/background, per-pixel validity changes, camera/light/spp/manual/toggle resets; max relative error="<<maxRelativeError<<".\n";
    }
}
int main()
{
    try { Run(1); Run(AdditionalLightingPlanes); return 0; }
    catch (const std::exception &error) { std::cerr<<"FAIL: "<<error.what()<<'\n'; return 1; }
}
