#include "Interop/InteropCommon.h"
#include "Renderer/DX12Context.h"
#include "../../Shaders/DiffuseBSDFProbeData.h"
#include <nvrtc.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <vector>
using namespace HybridInterop;
using namespace HybridBSDF;

namespace
{
    constexpr double Pi = 3.1415926535897932384626433832795;
    constexpr unsigned int SampleCount = 131072;
    struct Point
    {
        double X,Y,Z;
        Point operator+(Point b) const { return {X+b.X,Y+b.Y,Z+b.Z}; }
        Point operator-(Point b) const { return {X-b.X,Y-b.Y,Z-b.Z}; }
        Point operator*(double s) const { return {X*s,Y*s,Z*s}; }
    };
    Point From(Vector3 v) { return {v.X,v.Y,v.Z}; }
    Vector3 To(Point v) { return {static_cast<float>(v.X),static_cast<float>(v.Y),static_cast<float>(v.Z)}; }
    double Dot(Point a, Point b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
    Point Cross(Point a, Point b) { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
    Point Normalize(Point p) { return p*(1/std::sqrt(Dot(p,p))); }
    Point Tangent(Point n)
    {
        // Independent Gram-Schmidt basis, different axis/algorithm from the device sampler.
        const Point reference = std::abs(n.X) < 0.8 ? Point{1,0,0} : Point{0,1,0};
        return Normalize(reference-n*Dot(reference,n));
    }
    std::array<double,3> Components(Vector3 v) { return {v.X,v.Y,v.Z}; }
    void Require(bool condition, const std::string &reason)
    { if (!condition) throw std::runtime_error(reason); }
    void Require(bool condition, const char *reason)
    { if (!condition) throw std::runtime_error(reason); }
    void CheckNvrtc(nvrtcResult result, const char *operation)
    {
        if (result != NVRTC_SUCCESS) throw std::runtime_error(std::string(operation)+": "+nvrtcGetErrorString(result));
    }
    std::string ReadText(const std::filesystem::path &path)
    {
        std::ifstream input(path,std::ios::binary);
        if (!input) throw std::runtime_error("Cannot read BSDF device source/header: "+path.u8string());
        return {(std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>()};
    }
    class Probe
    {
    public:
        ~Probe()
        {
            if (mContext) cuCtxSetCurrent(mContext);
            if (mStream) cuStreamSynchronize(mStream);
            if (mStart) cuEventDestroy(mStart);
            if (mStop) cuEventDestroy(mStop);
            if (mInput) cuMemFree(mInput);
            if (mOutput) cuMemFree(mOutput);
            if (mHostInput) cuMemFreeHost(mHostInput);
            if (mModule) cuModuleUnload(mModule);
            if (mStream) cuStreamDestroy(mStream);
            if (mContext) cuCtxDestroy(mContext);
        }
        void Initialize(CUdevice device)
        {
            CheckCuda(cuCtxCreate(&mContext,CU_CTX_SCHED_AUTO,device),"cuCtxCreate(diffuse)");
            CheckCuda(cuStreamCreate(&mStream,CU_STREAM_NON_BLOCKING),"cuStreamCreate(diffuse)");
            CheckCuda(cuEventCreate(&mStart,CU_EVENT_DEFAULT),"cuEventCreate(start)");
            CheckCuda(cuEventCreate(&mStop,CU_EVENT_DEFAULT),"cuEventCreate(stop)");
            CheckCuda(cuMemAlloc(&mInput,SampleCount*sizeof(DiffuseProbeInput)),"cuMemAlloc(diffuse input)");
            CheckCuda(cuMemAlloc(&mOutput,SampleCount*sizeof(DiffuseProbeOutput)),"cuMemAlloc(diffuse output)");
            CheckCuda(cuMemAllocHost(&mHostInput,SampleCount*sizeof(DiffuseProbeInput)),"cuMemAllocHost(diffuse input)");
            wchar_t executable[MAX_PATH] = {};
            GetModuleFileNameW(nullptr,executable,MAX_PATH);
            const auto folder = std::filesystem::path(executable).parent_path()/L"Shaders";
            const auto source = ReadText(folder/L"DiffuseBSDFProbe.cu");
            const std::array<std::string,3> headerSources = {ReadText(folder/L"DiffuseBSDFTypes.h"),
                ReadText(folder/L"DiffuseBSDF.cuh"),ReadText(folder/L"DiffuseBSDFProbeData.h")};
            const char *headers[] = {headerSources[0].c_str(),headerSources[1].c_str(),headerSources[2].c_str()};
            const char *names[] = {"DiffuseBSDFTypes.h","DiffuseBSDF.cuh","DiffuseBSDFProbeData.h"};
            struct Program { nvrtcProgram Handle = nullptr; ~Program() { if (Handle) nvrtcDestroyProgram(&Handle); } } program;
            CheckNvrtc(nvrtcCreateProgram(&program.Handle,source.c_str(),"DiffuseBSDFProbe.cu",3,headers,names),"nvrtcCreateProgram(diffuse)");
            int major = 0, minor = 0;
            CheckCuda(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device),"cuDeviceGetAttribute(major)");
            CheckCuda(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device),"cuDeviceGetAttribute(minor)");
            const std::string architecture = "--gpu-architecture=compute_"+std::to_string(major*10+minor);
            const std::string include = std::string("--include-path=")+HYBRIDTRACER_CUDA_INCLUDE_DIR;
            const char *options[] = {architecture.c_str(),"--std=c++14","--fmad=false",include.c_str()};
            const auto result = nvrtcCompileProgram(program.Handle,4,options);
            size_t bytes = 0;
            CheckNvrtc(nvrtcGetProgramLogSize(program.Handle,&bytes),"nvrtcGetProgramLogSize(diffuse)");
            std::vector<char> log(bytes);
            CheckNvrtc(nvrtcGetProgramLog(program.Handle,log.data()),"nvrtcGetProgramLog(diffuse)");
            if (result != NVRTC_SUCCESS) throw std::runtime_error(std::string("NVRTC diffuse compile failed: ")+log.data());
            if (bytes > 1) std::cout << "NVRTC diffuse log: " << log.data();
            CheckNvrtc(nvrtcGetPTXSize(program.Handle,&bytes),"nvrtcGetPTXSize(diffuse)");
            std::vector<char> ptx(bytes);
            CheckNvrtc(nvrtcGetPTX(program.Handle,ptx.data()),"nvrtcGetPTX(diffuse)");
            CheckCuda(cuModuleLoadData(&mModule,ptx.data()),"cuModuleLoadData(diffuse)");
            CheckCuda(cuModuleGetFunction(&mSamples,mModule,"ProbeDiffuseSamples"),"cuModuleGetFunction(samples)");
            CheckCuda(cuModuleGetFunction(&mEval,mModule,"ProbeDiffuseEval"),"cuModuleGetFunction(eval)");
            std::cout << "Diffuse BSDF device functions compiled with NVRTC for compute_" << major << minor << ".\n";
        }
        template<class Input, class Output> std::vector<Output> Run(const std::vector<Input> &input, bool eval, float &milliseconds)
        {
            Require(!input.empty() && input.size() <= SampleCount,"Unsupported diffuse probe count.");
            // Pageable synchronous HtoD can return after staging, while DMA still runs on
            // the default stream. Pin and enqueue on the kernel's NON_BLOCKING stream.
            std::memcpy(mHostInput,input.data(),input.size()*sizeof(Input));
            CheckCuda(cuMemcpyHtoDAsync(mInput,mHostInput,input.size()*sizeof(Input),mStream),"cuMemcpyHtoDAsync(diffuse)");
            unsigned int count = static_cast<unsigned int>(input.size());
            void *arguments[] = {&mInput,&mOutput,&count};
            const CUfunction kernel = eval ? mEval : mSamples;
            if (!mWarmed)
            {
                CheckCuda(cuLaunchKernel(kernel,(count+255)/256,1,1,256,1,1,0,mStream,arguments,nullptr),"cuLaunchKernel(warmup)");
                CheckCuda(cuStreamSynchronize(mStream),"cuStreamSynchronize(warmup)");
                mWarmed = true;
            }
            CheckCuda(cuEventRecord(mStart,mStream),"cuEventRecord(start)");
            CheckCuda(cuLaunchKernel(kernel,(count+255)/256,1,1,256,1,1,0,mStream,arguments,nullptr),"cuLaunchKernel(diffuse)");
            CheckCuda(cuEventRecord(mStop,mStream),"cuEventRecord(stop)");
            CheckCuda(cuEventSynchronize(mStop),"cuEventSynchronize(diffuse)");
            CheckCuda(cuEventElapsedTime(&milliseconds,mStart,mStop),"cuEventElapsedTime(diffuse)");
            std::vector<Output> output(input.size());
            CheckCuda(cuMemcpyDtoH(output.data(),mOutput,output.size()*sizeof(Output)),"cuMemcpyDtoH(diffuse)");
            return output;
        }
    private:
        CUcontext mContext = nullptr;
        CUstream mStream = nullptr;
        CUmodule mModule = nullptr;
        CUfunction mSamples = nullptr, mEval = nullptr;
        CUdeviceptr mInput = 0, mOutput = 0;
        CUevent mStart = nullptr, mStop = nullptr;
        void *mHostInput = nullptr;
        bool mWarmed = false;
    };

    struct Fixture { const char *Name; Vector3 Normal,Albedo,Environment; bool GrazingOutgoing = false; };
    void ValidateDistribution(Probe &probe, const Fixture &fixture, unsigned int seed, std::ostream *report, std::ostream *samples)
    {
        const auto normal = To(Normalize(From(fixture.Normal)));
        const Point n = Normalize(From(normal)), t = Tangent(n), b = Cross(n,t);
        const Vector3 outgoing = fixture.GrazingOutgoing ? To(Normalize(n*0.001+t*0.999)) : normal;
        std::mt19937 generator(seed);
        const auto uniform = [&]() { return static_cast<float>(generator() >> 8)*(1.0f/16777216.0f); };
        std::vector<DiffuseProbeInput> input(SampleCount);
        for (auto &query : input) query = {normal,uniform(),outgoing,uniform(),fixture.Albedo,0,fixture.Environment,0};
        float milliseconds = 0;
        const auto output = probe.Run<DiffuseProbeInput,DiffuseProbeOutput>(input,false,milliseconds);
        const auto rho = Components(fixture.Albedo), environment = Components(fixture.Environment);
        const double expectedMean[] = {2*n.X/3,2*n.Y/3,2*n.Z/3};
        const double nc[] = {n.X,n.Y,n.Z};
        const double expectedAngular[] = {rho[0]*(2+n.X/3),rho[1]*(1+0.2*n.Y),rho[2]*(0.5+(1+n.Z*n.Z)/16)};
        double mean[3] = {}, second[3] = {}, constant[3] = {}, angular[3] = {};
        double maxPdfError = 0, maxWeightError = 0, maxConstantError = 0;
        std::array<unsigned int,64> bins = {};
        for (size_t i = 0; i < output.size(); ++i)
        {
            const auto &value = output[i];
            const auto &sample = value.Sample;
            const Point direction = From(sample.Direction);
            const double cosine = Dot(n,direction);
            if (!(sample.Valid == 1 && std::isfinite(cosine) && cosine > 0 && std::abs(Dot(direction,direction)-1) < 0.000003))
            {
                std::ostringstream message;
                message << std::setprecision(10) << fixture.Name << " sample " << i << ": invalid hemisphere/unit sample; valid="
                    << sample.Valid << ", direction=(" << direction.X << ',' << direction.Y << ',' << direction.Z << "), length2="
                    << Dot(direction,direction) << ", cosine=" << cosine << ", U=(" << input[i].U1 << ',' << input[i].U2 << ").";
                throw std::runtime_error(message.str());
            }
            const double pdfError = std::abs(sample.Pdf-cosine/Pi);
            maxPdfError = std::max(maxPdfError,pdfError);
            if (!(pdfError < 0.0000003 && sample.Pdf > 0 && std::abs(value.EvaluatedPdf-sample.Pdf) < 0.0000003))
                throw std::runtime_error(std::string(fixture.Name)+": solid-angle PDF mismatch.");
            Require(value.BacksidePdf == 0 && value.BacksideBRDFMax == 0,"Backside reflection/PDF must be zero.");
            Require(std::abs(value.CosTheta-cosine) < 0.000001,"Cosine differs from independent double dot product.");
            const auto d = Components(sample.Direction), weight = Components(sample.Weight), eval = Components(value.BRDF);
            const auto explicitWeight = Components(value.ExplicitWeight), radiance = Components(value.ConstantRadiance);
            const auto explicitRadiance = Components(value.ExplicitRadiance), directional = Components(value.DirectionalRadiance);
            const double angularFunction[] = {2+0.5*d[0],1+0.3*d[1],0.5+0.25*d[2]*d[2]};
            for (size_t c = 0; c < 3; ++c)
            {
                Require(std::isfinite(weight[c]) && std::abs(weight[c]-rho[c]) < 0.000001,"Diffuse sample weight is not reflectance.");
                Require(std::isfinite(eval[c]) && std::abs(eval[c]-rho[c]/Pi) < 0.0000001,"Eval must return raw rho/pi, without cosine.");
                const double weightError = std::abs(explicitWeight[c]-rho[c]);
                maxWeightError = std::max(maxWeightError,weightError);
                Require(std::isfinite(explicitWeight[c]) && weightError < 0.000002,"Explicit BRDF*cos/PDF differs from Sample weight.");
                const double expectedConstant = rho[c]*environment[c];
                const double error = std::abs(radiance[c]-expectedConstant);
                maxConstantError = std::max(maxConstantError,error);
                Require(std::isfinite(radiance[c]) && error < 0.000002*std::max(1.0,expectedConstant),"Constant environment radiance mismatch.");
                Require(std::isfinite(explicitRadiance[c]) && std::abs(explicitRadiance[c]-expectedConstant) < 0.000003*std::max(1.0,expectedConstant),
                    "Explicit constant environment radiance mismatch.");
                Require(std::isfinite(directional[c]) && std::abs(directional[c]-rho[c]*angularFunction[c]) < 0.000002,
                    "Directional environment contribution mismatch.");
                mean[c] += d[c]; second[c] += d[c]*d[c]; constant[c] += radiance[c]; angular[c] += directional[c];
            }
            double phi = std::atan2(Dot(direction,b),Dot(direction,t));
            if (phi < 0) phi += 2*Pi;
            const unsigned int zBin = std::min(7u,static_cast<unsigned int>(std::clamp(cosine,0.0,1.0)*8));
            const unsigned int phiBin = std::min(7u,static_cast<unsigned int>(phi/(2*Pi)*8));
            ++bins[zBin*8+phiBin];
            if (samples && i < 4)
                *samples << fixture.Name << ',' << seed << ',' << i << ',' << input[i].U1 << ',' << input[i].U2 << ','
                    << d[0] << ',' << d[1] << ',' << d[2] << ',' << sample.Pdf << ',' << weight[0] << ',' << weight[1] << ','
                    << weight[2] << ',' << radiance[0] << ',' << radiance[1] << ',' << radiance[2] << '\n';
        }
        for (size_t c = 0; c < 3; ++c)
        {
            mean[c] /= SampleCount; second[c] /= SampleCount; constant[c] /= SampleCount; angular[c] /= SampleCount;
            const double expectedSecond = (1+nc[c]*nc[c])/4;
            const double expectedFourth = 0.125+nc[c]*nc[c]/4-std::pow(nc[c],4)/24;
            const double meanTolerance = 6*std::sqrt((expectedSecond-expectedMean[c]*expectedMean[c])/SampleCount)+0.000002;
            const double secondTolerance = 6*std::sqrt((expectedFourth-expectedSecond*expectedSecond)/SampleCount)+0.000002;
            Require(std::abs(mean[c]-expectedMean[c]) < meanTolerance,std::string(fixture.Name)+": incorrect analytic first direction moment.");
            Require(std::abs(second[c]-expectedSecond) < secondTolerance,std::string(fixture.Name)+": incorrect analytic second direction moment.");
            const double angularTolerance = c == 2 ? rho[c]*secondTolerance/4 : rho[c]*(c == 0 ? 0.5 : 0.3)*meanTolerance;
            Require(std::abs(angular[c]-expectedAngular[c]) < angularTolerance+0.000002,"Nonconstant environment integral differs from analytic value.");
        }
        double chiSquared = 0;
        for (unsigned int z = 0; z < 8; ++z)
        {
            const double expected = SampleCount*double((z+1)*(z+1)-z*z)/64/8;
            for (unsigned int p = 0; p < 8; ++p)
            {
                const double difference = bins[z*8+p]-expected;
                chiSquared += difference*difference/expected;
            }
        }
        // 64 joint bins / 63 degrees of freedom; conservative rejection threshold.
        Require(chiSquared < 150,std::string(fixture.Name)+": cosine/azimuth joint distribution failed (chi2="+std::to_string(chiSquared)+").");
        std::cout << "PASS: " << fixture.Name << ": " << SampleCount << " GPU samples; constant L=(" << constant[0] << ',' << constant[1]
            << ',' << constant[2] << "), chi2=" << chiSquared << ", max PDF error=" << maxPdfError
            << ", max explicit-weight error=" << maxWeightError << ", kernel=" << milliseconds << " ms.\n";
        if (report)
            *report << fixture.Name << ',' << seed << ',' << SampleCount << ',' << constant[0] << ',' << constant[1] << ',' << constant[2]
                << ',' << angular[0] << ',' << angular[1] << ',' << angular[2] << ',' << expectedAngular[0] << ',' << expectedAngular[1]
                << ',' << expectedAngular[2] << ',' << chiSquared << ',' << maxPdfError << ',' << maxWeightError << ',' << maxConstantError
                << ',' << milliseconds << '\n';
    }

    void ValidateEdgesAndEval(Probe &probe)
    {
        const DiffuseProbeInput baseline = {{0,0,1},0.3f,{0,0,1},0.7f,{0.8f,0.8f,0.8f},0,{5,5,5},0};
        std::vector<DiffuseProbeInput> inputs;
        std::vector<bool> valid;
        const auto add = [&](DiffuseProbeInput query, bool expected) { inputs.push_back(query); valid.push_back(expected); };
        for (const float u1 : {0.0f,0.5f,std::nextafter(1.0f,0.0f)})
            for (const float u2 : {0.0f,0.25f,0.5f,std::nextafter(1.0f,0.0f)})
            { auto q = baseline; q.U1 = u1; q.U2 = u2; add(q,true); }
        auto q = baseline; q.Albedo = {0,0,0}; add(q,true);
        q = baseline; q.Albedo = {1,1,1}; add(q,true);
        for (const float bad : {-0.1f,1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        { q = baseline; q.U1 = bad; add(q,false); q = baseline; q.U2 = bad; add(q,false); }
        for (const Vector3 bad : {Vector3{0,0,0},Vector3{0,0,2},Vector3{0,0,std::numeric_limits<float>::quiet_NaN()}})
        { q = baseline; q.Normal = bad; add(q,false); q = baseline; q.Outgoing = bad; add(q,false); }
        q = baseline; q.Outgoing = {0,0,-1}; add(q,false);
        q = baseline; q.Outgoing = {1,0,0}; add(q,false);
        for (const float bad : {-0.1f,1.1f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()})
        { q = baseline; q.Albedo.Y = bad; add(q,false); }
        float milliseconds = 0;
        const auto results = probe.Run<DiffuseProbeInput,DiffuseProbeOutput>(inputs,false,milliseconds);
        for (size_t i = 0; i < results.size(); ++i)
        {
            if (!valid[i])
            {
                const DiffuseProbeOutput zero = {};
                Require(std::memcmp(&results[i],&zero,sizeof(zero)) == 0,"Invalid sample must return a finite zero record.");
            }
            else
            {
                const auto &r = results[i];
                Require(r.Sample.Valid == 1 && r.Sample.Pdf > 0 && std::isfinite(r.Sample.Pdf),"Valid endpoint/black/white sample rejected.");
                Require(std::abs(Dot(From(r.Sample.Direction),From(r.Sample.Direction))-1) < 0.000003,"Endpoint direction not unit.");
                const auto weight = Components(r.ExplicitWeight), rho = Components(inputs[i].Albedo);
                for (size_t c = 0; c < 3; ++c) Require(std::abs(weight[c]-rho[c]) < 0.000002,"Endpoint explicit weight mismatch.");
            }
        }
        std::vector<DiffuseEvalInput> evalInputs;
        const std::array<Vector3,4> directions = {Vector3{0,0,1},Vector3{0.6f,0,0.8f},Vector3{1,0,0},Vector3{0,0,-1}};
        for (const auto incoming : directions) for (const auto outgoing : directions)
            evalInputs.push_back({{0,0,1},0,incoming,0,outgoing,0,{0.2f,0.5f,1},0});
        const auto evaluations = probe.Run<DiffuseEvalInput,DiffuseEvalOutput>(evalInputs,true,milliseconds);
        for (size_t i = 0; i < evaluations.size(); ++i)
        {
            const auto &input = evalInputs[i];
            const bool front = input.Incoming.Z > 0 && input.Outgoing.Z > 0;
            const double expectedPdf = front ? input.Incoming.Z/Pi : 0;
            const auto rho = Components(input.Albedo), actual = Components(evaluations[i].BRDF);
            Require(std::abs(evaluations[i].Pdf-expectedPdf) < 0.0000001,"Arbitrary-direction PDF evaluated at wrong direction/hemisphere.");
            for (size_t c = 0; c < 3; ++c)
                Require(std::abs(actual[c]-(front ? rho[c]/Pi : 0)) < 0.0000001,"Arbitrary-direction Eval includes cosine or shades backside.");
        }
        std::cout << "PASS: " << inputs.size() << " endpoint/invalid/black/white cases and " << evaluations.size()
            << " independent incoming/outgoing Eval/PDF queries.\n";
    }
}

int main(int argc, char **argv)
{
    try
    {
        const bool writeSamples = argc == 2 && std::string(argv[1]) == "--samples";
        Require(argc == 1 || writeSamples,"Usage: HybridTracerDiffuseBSDFTest [--samples]");
        DX12Context dx;
        dx.InitializeHeadless();
        Probe probe;
        probe.Initialize(FindCudaDevice(dx.GetDevice()));
        std::ofstream report, samples;
        if (writeSamples)
        {
            report.open(std::filesystem::current_path()/L"DiffuseBSDFReport.csv");
            samples.open(std::filesystem::current_path()/L"DiffuseBSDFSamples.csv");
            Require(bool(report) && bool(samples),"Cannot create diffuse reports.");
            report << std::setprecision(10) << "case,seed,count,constant_r,constant_g,constant_b,angular_r,angular_g,angular_b,expected_angular_r,expected_angular_g,expected_angular_b,chi2,max_pdf_error,max_explicit_weight_error,max_constant_error,kernel_ms\n";
            samples << std::setprecision(10) << "case,seed,index,u1,u2,dir_x,dir_y,dir_z,pdf,weight_r,weight_g,weight_b,radiance_r,radiance_g,radiance_b\n";
        }
        const Vector3 gray = {0.8f,0.8f,0.8f}, constant = {5,5,5};
        const Fixture fixtures[] = {
            {"plus_z",{0,0,1},gray,constant}, {"minus_z",{0,0,-1},gray,constant},
            {"plus_y",{0,1,0},gray,constant}, {"minus_y",{0,-1,0},gray,constant},
            {"plus_x",{1,0,0},gray,constant}, {"minus_x",{-1,0,0},gray,constant},
            {"tilted",{-0.358f,0.602f,-0.713f},gray,constant},
            {"near_pole",{0.00001f,0,1},gray,constant},
            {"basis_switch_low",{0.05f,0,0.998f},gray,constant},
            {"basis_switch_high",{0.01f,0,0.99995f},gray,constant},
            {"grazing_outgoing",{1,2,3},gray,constant,true},
            {"colored",{1,2,-3},{0.85f,0.1f,0.08f},{2,0.5f,3}},
            {"black_white_hdr",{-1,2,3},{0,1,0.25f},{0.0001f,5,100}}
        };
        unsigned int seed = 137;
        for (const auto &fixture : fixtures)
        {
            ValidateDistribution(probe,fixture,seed,writeSamples ? &report : nullptr,writeSamples ? &samples : nullptr);
            seed += 7919;
        }
        ValidateEdgesAndEval(probe);
        if (writeSamples)
        {
            report.flush(); samples.flush();
            Require(bool(report) && bool(samples),"Cannot flush diffuse reports.");
            std::cout << "Saved DiffuseBSDFReport.csv and DiffuseBSDFSamples.csv.\n";
        }
        std::cout << "PASS: reusable CUDA diffuse Sample/Eval/PDF, analytic radiance and direction statistics.\n"
            << "NOTE: CUDA event times measure this diagnostic kernel only; uploads, readback, CPU checks and renderer passes are excluded.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
