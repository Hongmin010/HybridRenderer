#pragma once
#include <optix.h>
#include "SurfaceInput.h"
#include "OptixSceneData.h"
#include "SurfaceHit.h"
#include "IndirectLighting.h"

struct ShadowResult
{
    float Visibility;             // 1 = miss/open, 0 = blocked or invalid background.
    unsigned int Blocker;         // SceneObject index + 1; 0 = no blocker.
    float HitDistance;            // From the offset ray origin; 0 for miss/background.
    unsigned int Valid;           // Copied from surface validity, not inferred from visibility.
};
struct ShadowSettings
{
    float LightX, LightY, LightZ; // Unit world direction FROM surface TO directional light.
    float NormalOffset;
    float TMin, TMax;
};

enum class TraceLightKind : unsigned int { Directional, Point, Rectangle };
constexpr unsigned int MaxAdditionalLights = 16;
struct TraceLight
{
    float PositionOrDirection[3];
    float Emission[3];
    float HalfU[3], HalfV[3];
    float Normal[3];
    float Area;
    TraceLightKind Kind;
};

struct OptixTraceLaunchParams
{
    const SurfaceInput *Surfaces;
    ShadowResult *Output;
    OptixTraversableHandle Gas;
    unsigned int Width, Height;
    ShadowSettings Settings;
    const SurfaceRay *QueryRays;  // Non-null selects independent surface-query mode.
    SurfaceHit *QueryHits;
    IndirectLighting *IndirectOutput;
    IndirectSampleDebug *IndirectDebug;
    IndirectSettings Indirect;
    unsigned int AdditionalLightCount;
    TraceLight AdditionalLights[MaxAdditionalLights];
};
static_assert(sizeof(ShadowResult) == 16, "ShadowResult ABI");
static_assert(sizeof(ShadowSettings) == 24, "ShadowSettings ABI");
static_assert(sizeof(TraceLight) == 68, "TraceLight ABI");
static_assert(sizeof(OptixTraceLaunchParams) == 1216, "OptixTraceLaunchParams ABI");
