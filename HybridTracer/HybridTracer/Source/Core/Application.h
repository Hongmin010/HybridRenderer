#pragma once

#include <Windows.h>

#include "Renderer/DX12Context.h"
#include "Renderer/RasterGBufferPass.h"
#include "Renderer/DebugViewPass.h"
#include "../../Shaders/IndirectLighting.h"

#include "Scene/Camera.h"
#include "Scene/Scene.h"
#include <filesystem>
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
#include "Renderer/HybridFramePipeline.h"
#include <memory>
#endif

class Application
{
public:
    Application();
    ~Application();

    int Run(bool smokeTest = false, bool profile = false, bool accumulation = true,
        unsigned int maxIndirectBounces = DefaultIndirectBounces, const std::filesystem::path &sceneFile = {});

    HWND GetWindowHandle() const
    {
        return mHwnd;
    }

private:
    bool InitWindow();

    void UpdateCamera();

    void OnMouseDown(
        WPARAM buttonState,
        int x,
        int y);

    void OnMouseUp(
        WPARAM buttonState,
        int x,
        int y);

    void OnMouseMove(
        WPARAM buttonState,
        int x,
        int y);

    void OnMouseWheel(
        short wheelDelta);

    static LRESULT CALLBACK WindowProc(
        HWND hwnd,
        UINT msg,
        WPARAM wParam,
        LPARAM lParam);

private:
    HINSTANCE mInstance = nullptr;
    HWND mHwnd = nullptr;

    int mWidth = 1280;
    int mHeight = 720;

    DX12Context mDX12;

    RasterGBufferPass mRasterPass;
    DebugViewPass mDebugPass;

    Camera mCamera;
    Scene mScene;
#ifdef HYBRIDTRACER_HAS_HYBRID_RENDERER
    std::unique_ptr<HybridFramePipeline> mHybrid;
#endif

    UINT mDebugMode = 0;
    bool mToggleWasDown=false,mResetWasDown=false;

    float mTheta = -1.5707963f;
    float mPhi = 1.22f;
    float mRadius = 7.5f;

    bool mLeftMouseDown = false;

    POINT mLastMousePosition =
        {
            0,
            0};
};
