#pragma once
#include "SurfaceHit.h"

// Linear outgoing radiance at the rasterized surface, including its BSDF weight.
// Each lighting plane uses the same layout.
struct IndirectLighting { float R,G,B; unsigned int Valid; };
constexpr unsigned int MaxIndirectSamples = 4096;
constexpr unsigned int DefaultIndirectBounces = 1;
constexpr unsigned int MaxIndirectBounces = 8;
// Lighting uses one plane for indirect-only or three planes when additional lights are active.
constexpr unsigned int AdditionalLightingPlanes = 3;
constexpr unsigned int AdditionalDirectPlane = 1;
constexpr unsigned int AdditionalUnshadowedPlane = 2;
struct IndirectSettings
{
    float CameraX,CameraY,CameraZ;
    unsigned int Seed;
    // Directional irradiance on a plane perpendicular to the light direction.
    float IrradianceR,IrradianceG,IrradianceB;
    unsigned int SamplesPerPixel;
    unsigned int MaxBounces;
};
// Optional diagnostics: first bounce of the last sample, including invalid/missed rays.
struct IndirectSampleDebug
{
    SurfaceRay Ray;
    SurfaceHit Hit;
    float WeightR,WeightG,WeightB,Pdf;
    float Visibility;
    unsigned int SampleValid,Padding0,Padding1;
};
static_assert(sizeof(IndirectLighting) == 16, "IndirectLighting ABI");
static_assert(sizeof(IndirectSettings) == 36, "IndirectSettings ABI");
static_assert(sizeof(IndirectSampleDebug) == 160, "IndirectSampleDebug ABI");
