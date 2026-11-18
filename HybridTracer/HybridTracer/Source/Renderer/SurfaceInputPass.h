#pragma once
#include "Interop/CudaDepthPositionReader.h"

// Production interface to the already validated G-buffer import/reconstruction engine.
// The interop owner handles CUDA resources/fences; this pass exposes only GPU inputs/output.
// Caller completes the previous frame before reusing its single matrix/output allocation.
class SurfaceInputPass
{
public:
    void Initialize(ID3D12Device *device, ID3D12Resource *depth, ID3D12Resource *normal, ID3D12Resource *albedo)
    { mInterop.Initialize(device,depth,normal,albedo,false); }
    void Execute(ID3D12CommandQueue *queue, UINT64 value, const float *inverseVP, const CudaTimingEvents *timing = nullptr)
    { mInterop.EnqueueCompute(queue,value,inverseVP,timing); }
    CUdeviceptr GetOutput() const { return mInterop.GetDeviceSurfaceInputs(); }
    CUcontext GetContext() const { return mInterop.GetContext(); }
    CUstream GetStream() const { return mInterop.GetStream(); }
private:
    CudaDepthPositionReader mInterop;
};
