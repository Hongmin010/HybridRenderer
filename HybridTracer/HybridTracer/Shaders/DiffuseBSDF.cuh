#pragma once
#include <cuda_runtime.h>
#include "DiffuseBSDFTypes.h"

// Shared device functions for ordinary CUDA and future OptiX diffuse transport.
// One-sided Lambert reflection. N, incoming and outgoing must be finite unit vectors
// (length-squared tolerance 1e-4); N faces the outgoing direction. Albedo is linear [0,1].
// Random inputs are supplied by the caller, independently uniform in [0,1).
namespace HybridBSDF
{
    constexpr float Pi = 3.14159265358979323846f;
    constexpr float InvPi = 0.31830988618379067154f;

    __device__ __forceinline__ float Dot(Vector3 a, Vector3 b)
    { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
    __device__ __forceinline__ Vector3 Scale(Vector3 a, float b)
    { return {a.X*b,a.Y*b,a.Z*b}; }
    __device__ __forceinline__ Vector3 Cross(Vector3 a, Vector3 b)
    { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
    __device__ __forceinline__ bool IsUnit(Vector3 a)
    {
        return isfinite(a.X) && isfinite(a.Y) && isfinite(a.Z) && fabsf(Dot(a,a)-1.0f) <= 0.0001f;
    }
    __device__ __forceinline__ bool IsReflectance(Vector3 a)
    {
        return isfinite(a.X) && isfinite(a.Y) && isfinite(a.Z) &&
            a.X >= 0 && a.X <= 1 && a.Y >= 0 && a.Y <= 1 && a.Z >= 0 && a.Z <= 1;
    }
    __device__ __forceinline__ Vector3 Normalize(Vector3 a)
    { return Scale(a,rsqrtf(Dot(a,a))); }

    // Raw BRDF only: NO cosine, PDF division, lighting or visibility included.
    __device__ __forceinline__ Vector3 EvalDiffuseBRDF(Vector3 albedo, Vector3 normal,
        Vector3 incoming, Vector3 outgoing)
    {
        if (!IsReflectance(albedo) || !IsUnit(normal) || !IsUnit(incoming) || !IsUnit(outgoing) ||
            Dot(normal,incoming) <= 0 || Dot(normal,outgoing) <= 0) return {};
        return Scale(albedo,InvPi);
    }

    // Density of SampleDiffuse for an arbitrary incoming direction, not just its own sample.
    __device__ __forceinline__ float PdfDiffuse(Vector3 normal, Vector3 incoming, Vector3 outgoing)
    {
        if (!IsUnit(normal) || !IsUnit(incoming) || !IsUnit(outgoing) || Dot(normal,outgoing) <= 0) return 0;
        return fmaxf(Dot(Normalize(normal),incoming),0.0f)*InvPi;
    }

    __device__ __forceinline__ DiffuseSample SampleDiffuse(Vector3 albedo, Vector3 normal,
        Vector3 outgoing, float u1, float u2)
    {
        if (!IsReflectance(albedo) || !IsUnit(normal) || !IsUnit(outgoing) || Dot(normal,outgoing) <= 0 ||
            !(u1 >= 0 && u1 < 1) || !(u2 >= 0 && u2 < 1)) return {};
        const Vector3 n = Normalize(normal);
        const Vector3 axis = fabsf(n.Z) < 0.999f ? Vector3{0,0,1} : Vector3{0,1,0};
        const Vector3 tangent = Normalize(Cross(axis,n));
        const Vector3 bitangent = Cross(n,tangent);
        const float radius = sqrtf(u1), phi = 2.0f*Pi*u2;
        const float x = radius*cosf(phi), y = radius*sinf(phi), z = sqrtf(1.0f-u1);
        const Vector3 direction = Normalize({tangent.X*x+bitangent.X*y+n.X*z,
            tangent.Y*x+bitangent.Y*y+n.Y*z,tangent.Z*x+bitangent.Z*y+n.Z*z});
        const float pdf = PdfDiffuse(n,direction,outgoing);
        if (!(pdf > 0)) return {};
        // (rho/pi)*cos / (cos/pi) = rho for this exact BRDF/proposal pair.
        // Black reflectance remains a valid sample; its throughput is zero.
        return {direction,pdf,albedo,1};
    }
}
