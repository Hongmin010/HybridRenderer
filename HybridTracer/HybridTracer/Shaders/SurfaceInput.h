#pragma once

// Shared host/CUDA/OptiX ABI; keep this header independent of graphics APIs.
struct WorldPosition
{
    float X, Y, Z, Valid;
};
struct SurfaceInput
{
    WorldPosition Position;
    float NormalX, NormalY, NormalZ, NormalPadding;
    float AlbedoR, AlbedoG, AlbedoB, AlbedoA;
};
static_assert(sizeof(WorldPosition) == 16, "WorldPosition ABI");
static_assert(sizeof(SurfaceInput) == 48, "SurfaceInput ABI");
