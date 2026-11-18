#include "Core/Application.h"
#include "Renderer/IndirectBounceOptions.h"
#include <d3d12sdklayers.h>
#include <iostream>
#include <stdexcept>
#include <string>

int wmain(int argc, wchar_t **argv)
{
    try
    {
        bool smoke=false,profile=false,accumulation=true;
        unsigned int maxIndirectBounces=DefaultIndirectBounces;
        std::filesystem::path sceneFile;
#ifndef HYBRIDTRACER_HAS_HYBRID_RENDERER
        bool hasBounceOption=false;
#endif
        for (int i=1;i<argc;++i)
        {
            const std::wstring flag=argv[i];
            if (flag==L"--smoke-test") smoke=true;
            else if (flag==L"--profile") profile=true;
            else if (flag==L"--no-accumulation") accumulation=false;
            else if (flag==L"--scene")
            {
                if (++i >= argc || !*argv[i] || argv[i][0]==L'-' || !sceneFile.empty())
                    throw std::runtime_error("--scene expects one scene file path.");
                sceneFile=argv[i];
            }
            else if (flag==L"--bounces")
            {
                maxIndirectBounces=ParseIndirectBounceCount(++i < argc ? argv[i] : nullptr);
#ifndef HYBRIDTRACER_HAS_HYBRID_RENDERER
                hasBounceOption=true;
#endif
            }
            else throw std::runtime_error("Usage: HybridTracer [--scene path.xml] [--bounces 0..8] [--profile] [--no-accumulation] [--smoke-test]");
        }
#ifndef HYBRIDTRACER_HAS_HYBRID_RENDERER
        if (profile || !accumulation || hasBounceOption)
            throw std::runtime_error("Profiling, accumulation and bounce controls require HYBRIDTRACER_ENABLE_HYBRID_RENDERER=ON.");
#endif
        if (smoke)
        {
            Microsoft::WRL::ComPtr<ID3D12Debug> debug;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
        }
        Application app;
        return app.Run(smoke,profile,accumulation,maxIndirectBounces,sceneFile);
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
