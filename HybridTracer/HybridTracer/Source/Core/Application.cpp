
#include "Application.h"
#include "Scene/SceneFile.h"

#include <Windows.h>
#include <windowsx.h>

#include <DirectXMath.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <d3d12sdklayers.h>
#include <iostream>
#include <iterator>

using namespace DirectX;

namespace
{
    constexpr UINT WindowWidth = 1280;
    constexpr UINT WindowHeight = 720;
    constexpr UINT FrameCount = 2;

    constexpr wchar_t WindowClassName[] =
        L"HybridTracerWindowClass";
}

Application::Application() = default;

Application::~Application()
{
    // Drain DX12 (including its CUDA completion waits) before Scene/pass members die.
    try { mDX12.Flush(); }
    catch (const std::exception &error) { std::cerr << "Application shutdown: " << error.what() << '\n'; }
}

int Application::Run(bool smokeTest, bool profile, bool accumulation, unsigned int maxIndirectBounces,
    const std::filesystem::path &sceneFile)
{
    const auto path = sceneFile.empty() ? DefaultSceneFile() : sceneFile;
    const auto description = ReadSceneFile(path);
    const UINT width = description.Camera ? description.Camera->Width : WindowWidth;
    const UINT height = description.Camera ? description.Camera->Height : WindowHeight;
    const float aspect = float(width)/height;
    const float fovY = description.Camera ? description.Camera->VerticalFov(aspect) : XM_PIDIV4;
    const float nearClip = description.Camera ? description.Camera->Near : 0.1f;
    const float farClip = description.Camera ? description.Camera->Far : 100.0f;
#ifndef HYBRIDTRACER_HAS_HYBRID_RENDERER
    if (description.Lights.size() > 1 ||
        (!description.Lights.empty() && !std::holds_alternative<DirectionalLight>(description.Lights.front())))
        throw std::runtime_error("This scene requires HYBRIDTRACER_ENABLE_HYBRID_RENDERER=ON.");
#endif
    std::cout << "Scene: " << path.u8string() << " (" << description.Objects.size() << " objects, "
        << description.Lights.size() << " lights, " << width << 'x' << height << ")\n";
    HINSTANCE instance = GetModuleHandleW(nullptr);
    WNDCLASSW windowClass = {};
    windowClass.lpfnWndProc = Application::WindowProc;
    windowClass.hInstance = instance;
    windowClass.lpszClassName = WindowClassName;
    windowClass.hCursor = LoadCursor(nullptr,IDC_ARROW);
    RegisterClassW(&windowClass);
    RECT rect = {0,0,LONG(width),LONG(height)};
    AdjustWindowRect(&rect,WS_OVERLAPPEDWINDOW,FALSE);
    mHwnd = CreateWindowExW(0,WindowClassName,L"HybridTracer",WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT,CW_USEDEFAULT,rect.right-rect.left,rect.bottom-rect.top,
        nullptr,nullptr,instance,this);
    if (!mHwnd) throw std::runtime_error("Failed to create application window.");
    mDX12.Initialize(mHwnd,width,height);
    description.Apply(mScene,mDX12.GetDevice());
    if (description.Camera)
    {
        const auto &camera = *description.Camera;
        camera.Apply(mCamera,width,height);
        mOrbitTarget = {camera.Position.x+camera.Forward.x*mRadius,
            camera.Position.y+camera.Forward.y*mRadius,camera.Position.z+camera.Forward.z*mRadius};
        mCameraUp = camera.Up;
        mTheta = std::atan2(-camera.Forward.z,-camera.Forward.x);
        mPhi = std::acos(std::clamp(-camera.Forward.y,-1.0f,1.0f));
    }
    else
    {
        mCamera.SetLens(fovY,aspect,nearClip,farClip);
        UpdateCamera();
    }
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
    mHybrid = std::make_unique<HybridFramePipeline>();
    mHybrid->Initialize(mDX12.GetDevice(),mDX12.GetCommandQueue(),mDX12.GetBackBufferFormat(),
        width,height,FrameCount,mScene,profile,accumulation);
    std::optional<FrameTiming> latestTiming;
    std::unique_ptr<FrameTimingLog> timingLog;
    if (profile) timingLog=std::make_unique<FrameTimingLog>(L"HybridTracerTimings.csv");
    const wchar_t *titles[] = {L"HybridTracer | 1: Direct + indirect",L"HybridTracer | 2: Albedo",
        L"HybridTracer | 3: World normal",L"HybridTracer | 4: Depth",L"HybridTracer | 5: Shadow visibility",
        L"HybridTracer | 6: Primary direct unshadowed + indirect",
        L"HybridTracer | 7: Direct only",L"HybridTracer | 8: Indirect only"};
#else
    mRasterPass.Initialize(mDX12.GetDevice(),mDX12.GetBackBufferFormat(),width,height,FrameCount);
    mDebugPass.Initialize(mDX12.GetDevice(),mDX12.GetBackBufferFormat(),mRasterPass.GetAlbedoBuffer(),
        mRasterPass.GetNormalBuffer(),mRasterPass.GetRasterDepthBuffer());
    const wchar_t *titles[] = {L"HybridTracer | 1: Shaded",L"HybridTracer | 2: Albedo",
        L"HybridTracer | 3: World normal",L"HybridTracer | 4: Depth"};
#endif
    SetWindowTextW(mHwnd,titles[0]);
    if (!smokeTest) { ShowWindow(mHwnd,SW_SHOW); UpdateWindow(mHwnd); }
    MSG message = {};
    UINT rendered = 0;
    while (message.message != WM_QUIT)
    {
        if (PeekMessageW(&message,nullptr,0,0,PM_REMOVE))
        {
            TranslateMessage(&message);
            DispatchMessageW(&message);
            continue;
        }
        if (smokeTest)
        {
            mDebugMode = rendered % static_cast<UINT>(std::size(titles));
            if (rendered==4) { mTheta+=0.05f; mCameraMoved=true; }
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
            if (rendered==6) mHybrid->SetAccumulationEnabled(false);
            if (rendered==7) mHybrid->SetAccumulationEnabled(true);
            if (rendered==9) mHybrid->ResetAccumulation();
#endif
            if (rendered==10) mCamera.SetLens(fovY,aspect,nearClip+(std::min)(nearClip,(farClip-nearClip)/2),farClip);
        }
        else if (GetForegroundWindow() == mHwnd)
        {
            for (UINT i = 0; i < std::size(titles); ++i)
                if (GetAsyncKeyState('1'+static_cast<int>(i)) & 0x8000) mDebugMode = i;
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
            const bool toggle=(GetAsyncKeyState('T') & 0x8000)!=0,reset=(GetAsyncKeyState('R') & 0x8000)!=0;
            if (toggle && !mToggleWasDown) mHybrid->SetAccumulationEnabled(!mHybrid->GetAccumulationInfo().Enabled);
            if (reset && !mResetWasDown) mHybrid->ResetAccumulation();
            mToggleWasDown=toggle; mResetWasDown=reset;
#endif
        }
        else { mToggleWasDown=false; mResetWasDown=false; }
#ifndef HYBRIDTRACER_HAS_HYBRID_RENDERER
        SetWindowTextW(mHwnd,titles[mDebugMode]);
#endif
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
        const auto cpuBegin=profile ? ProfileClock::now() : ProfileClock::time_point{};
#endif
        if (!description.Camera || mCameraMoved) UpdateCamera();
        auto *list = mDX12.BeginFrame();
        const auto output = mDX12.GetCurrentRTV();
        const UINT index = mDX12.GetCurrentFrameIndex();
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
        mHybrid->RecordRaster(list,output,index,mCamera,mScene);
        list = mDX12.SubmitAndContinue();
        mHybrid->EnqueueTrace(mCamera,mScene.GetDirectionalLight(),4,0,maxIndirectBounces);
        mHybrid->RecordComposite(list,output,static_cast<HybridView>(mDebugMode));
#else
        mRasterPass.Render(list,output,index,mCamera,mScene);
        if (mDebugMode) mDebugPass.Render(list,output,mDebugMode);
#endif
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
        const double submitMs=profile ? ProfileMilliseconds(cpuBegin) : 0;
        const auto endBegin=profile ? ProfileClock::now() : ProfileClock::time_point{};
#endif
        mDX12.EndFrame();
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
        if (profile)
        {
            mHybrid->SetCpuTiming(submitMs,ProfileMilliseconds(endBegin));
            if (auto timing=mHybrid->TakeCompletedTiming())
            {
                timingLog->Add(*timing);
                latestTiming=*timing;
            }
        }
        const auto &history=mHybrid->GetAccumulationInfo();
        std::wostringstream title;
        title<<titles[mDebugMode]<<L" | Max indirect bounces "<<maxIndirectBounces<<L" | Accum "<<(history.Enabled ? L"ON ":L"OFF ")
            <<history.Frames<<L"f / "<<history.SamplesPerPixel<<L" spp | T:toggle R:reset";
        if (latestTiming) title<<L" | GPU "<<std::fixed<<std::setprecision(2)
            <<latestTiming->At(FrameMetric::GpuSpan)<<L" ms | Accum "<<latestTiming->At(FrameMetric::Accumulation)<<L" ms";
        SetWindowTextW(mHwnd,title.str().c_str());
        if (smokeTest)
        {
            const UINT expected[]={1,2,3,4,1,2,0,1,2,1,1,2};
            const UINT frames=(!accumulation && rendered<6) ? 0u:expected[rendered];
            if (history.Frames!=frames || history.SamplesPerPixel!=UINT64(frames)*4)
                throw std::runtime_error("Window smoke accumulation/reset/toggle failed.");
        }
#endif
        if (smokeTest && ++rendered == 12)
        {
            mDX12.Flush();
            DestroyWindow(mHwnd);
        }
    }
    mDX12.Flush();
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
    if (profile)
    {
        mHybrid->CollectTimings();
        if (auto timing=mHybrid->TakeCompletedTiming()) timingLog->Add(*timing);
        if (smokeTest && timingLog->Size()!=12) throw std::runtime_error("Profile smoke must collect all 12 frames.");
        timingLog->Summary(std::cout);
        std::cout<<"TIMING CSV: HybridTracerTimings.csv (interactive frames include Present/vsync in cpu_end_frame_ms).\n";
    }
#endif
    if (smokeTest)
    {
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
        mHybrid->ValidateGAS(mScene);
#endif
        Microsoft::WRL::ComPtr<ID3D12InfoQueue> messages;
        if (SUCCEEDED(mDX12.GetDevice()->QueryInterface(IID_PPV_ARGS(&messages))))
        {
            for (UINT64 i = 0; i < messages->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i)
            {
                SIZE_T bytes = 0;
                messages->GetMessage(i,nullptr,&bytes);
                std::vector<unsigned char> storage(bytes);
                auto *entry = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
                if (FAILED(messages->GetMessage(i,entry,&bytes))) throw std::runtime_error("Cannot read D3D12 validation log.");
                if (entry->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) throw std::runtime_error(entry->pDescription);
            }
        }
        std::cout << "PASS: 12 window/swapchain frames, all available views, clean D3D12 validation.\n";
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
        std::cout << "PASS: window history growth, camera/projection reset, toggle and manual reset.\n";
#endif
    }
    return static_cast<int>(message.wParam);
}
void Application::UpdateCamera()
{
    float x =
        mRadius *
        std::sin(mPhi) *
        std::cos(mTheta);

    float y =
        mRadius *
        std::cos(mPhi);

    float z =
        mRadius *
        std::sin(mPhi) *
        std::sin(mTheta);

    mCamera.LookAt(
        XMFLOAT3(mOrbitTarget.x+x,mOrbitTarget.y+y,mOrbitTarget.z+z),
        mOrbitTarget,
        mCameraUp);
    mCameraMoved = false;
}

//
// =========================================================
// Window Procedure
// =========================================================
//

LRESULT CALLBACK Application::WindowProc(
    HWND hwnd,
    UINT message,
    WPARAM wParam,
    LPARAM lParam)
{
    //
    // Application 포인터 연결
    //

    if (message == WM_NCCREATE)
    {
        CREATESTRUCTW *createStruct =
            reinterpret_cast<CREATESTRUCTW *>(
                lParam);

        Application *app =
            static_cast<Application *>(
                createStruct->lpCreateParams);

        SetWindowLongPtrW(
            hwnd,
            GWLP_USERDATA,
            reinterpret_cast<LONG_PTR>(app));

        return TRUE;
    }

    Application *app =
        reinterpret_cast<Application *>(
            GetWindowLongPtrW(
                hwnd,
                GWLP_USERDATA));

    if (!app)
    {
        return DefWindowProcW(
            hwnd,
            message,
            wParam,
            lParam);
    }

    switch (message)
    {
        //
        // -----------------------------------------------------
        // Left Mouse Button Down
        // -----------------------------------------------------
        //

    case WM_LBUTTONDOWN:
    {
        app->mLeftMouseDown = true;

        app->mLastMousePosition.x =
            GET_X_LPARAM(lParam);

        app->mLastMousePosition.y =
            GET_Y_LPARAM(lParam);

        SetCapture(hwnd);

        return 0;
    }

        //
        // -----------------------------------------------------
        // Left Mouse Button Up
        // -----------------------------------------------------
        //

    case WM_LBUTTONUP:
    {
        app->mLeftMouseDown = false;

        ReleaseCapture();

        return 0;
    }

        //
        // -----------------------------------------------------
        // Mouse Move - Orbit Camera
        // -----------------------------------------------------
        //

    case WM_MOUSEMOVE:
    {
        POINT currentPosition =
            {
                GET_X_LPARAM(lParam),
                GET_Y_LPARAM(lParam)};

        if (app->mLeftMouseDown)
        {
            float dx =
                static_cast<float>(
                    currentPosition.x -
                    app->mLastMousePosition.x);

            float dy =
                static_cast<float>(
                    currentPosition.y -
                    app->mLastMousePosition.y);

            app->mTheta += dx * 0.01f;
            app->mPhi += dy * 0.01f;
            app->mCameraMoved = true;

            app->mPhi = std::clamp(
                app->mPhi,
                0.1f,
                XM_PI - 0.1f);
        }

        app->mLastMousePosition =
            currentPosition;

        return 0;
    }

        //
        // -----------------------------------------------------
        // Mouse Wheel - Zoom
        // -----------------------------------------------------
        //

    case WM_MOUSEWHEEL:
    {
        float wheelDelta =
            static_cast<float>(
                GET_WHEEL_DELTA_WPARAM(wParam)) /
            static_cast<float>(WHEEL_DELTA);

        app->mCameraMoved = true;
        app->mRadius -=
            wheelDelta * 0.5f;

        app->mRadius = std::clamp(
            app->mRadius,
            2.5f,
            15.0f);

        return 0;
    }

    case WM_CAPTURECHANGED:
    {
        app->mLeftMouseDown = false;
        return 0;
    }

        //
        // -----------------------------------------------------
        // Window Destroy
        // -----------------------------------------------------
        //

    case WM_DESTROY:
    {
        PostQuitMessage(0);
        return 0;
    }
    }

    return DefWindowProcW(
        hwnd,
        message,
        wParam,
        lParam);
}
