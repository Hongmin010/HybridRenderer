#pragma once
#include <optix.h>
#include "OptixSceneData.h"
#include "SurfaceHit.h"

namespace HybridSurface
{
    __device__ __forceinline__ float3 Subtract(float3 a, float3 b)
    { return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }
    __device__ __forceinline__ float3 Cross(float3 a, float3 b)
    { return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
    __device__ __forceinline__ float3 Normalize(float3 a)
    {
        const float scale = rsqrtf(a.x*a.x+a.y*a.y+a.z*a.z);
        return make_float3(a.x*scale,a.y*scale,a.z*scale);
    }
    __device__ __forceinline__ bool ValidRay(const SurfaceRay &ray)
    {
        const float length2 = ray.DirectionX*ray.DirectionX+ray.DirectionY*ray.DirectionY+ray.DirectionZ*ray.DirectionZ;
        return ray.Valid == 1 && isfinite(ray.OriginX) && isfinite(ray.OriginY) && isfinite(ray.OriginZ) &&
            isfinite(length2) && fabsf(length2-1.0f) <= 0.0001f &&
            isfinite(ray.TMin) && isfinite(ray.TMax) && ray.TMin >= 0 && ray.TMax > ray.TMin;
    }
    // Closest-hit context only. Native triangle attributes and per-object SBT data.
    __device__ __forceinline__ SurfaceHit ReadCurrentHit(const SceneHitData &data)
    {
        SurfaceHit hit = {};
        const unsigned int primitive = optixGetPrimitiveIndex();
        if (primitive >= data.TriangleCount) return hit;
        const GasTriangle triangle = data.Indices[primitive];
        const auto p0 = data.Vertices[triangle.X], p1 = data.Vertices[triangle.Y], p2 = data.Vertices[triangle.Z];
        const auto n0 = data.Normals[triangle.X], n1 = data.Normals[triangle.Y], n2 = data.Normals[triangle.Z];
        const float2 bary = optixGetTriangleBarycentrics();
        const float w = 1.0f-bary.x-bary.y;
        const float3 a = make_float3(p0.X,p0.Y,p0.Z), b = make_float3(p1.X,p1.Y,p1.Z), c = make_float3(p2.X,p2.Y,p2.Z);
        const float3 geometric = Normalize(Cross(Subtract(b,a),Subtract(c,a)));
        float3 shading = make_float3(w*n0.X+bary.x*n1.X+bary.y*n2.X,
            w*n0.Y+bary.x*n1.Y+bary.y*n2.Y,w*n0.Z+bary.x*n1.Z+bary.y*n2.Z);
        const float length2 = shading.x*shading.x+shading.y*shading.y+shading.z*shading.z;
        shading = isfinite(length2) && length2 > 1e-20f ? Normalize(shading) : geometric;
        hit.PositionX = w*p0.X+bary.x*p1.X+bary.y*p2.X;
        hit.PositionY = w*p0.Y+bary.x*p1.Y+bary.y*p2.Y;
        hit.PositionZ = w*p0.Z+bary.x*p1.Z+bary.y*p2.Z;
        hit.Distance = optixGetRayTmax();
        hit.GeometricNormalX = geometric.x; hit.GeometricNormalY = geometric.y; hit.GeometricNormalZ = geometric.z;
        hit.ShadingNormalX = shading.x; hit.ShadingNormalY = shading.y; hit.ShadingNormalZ = shading.z;
        hit.BarycentricU = bary.x; hit.BarycentricV = bary.y;
        hit.AlbedoR = data.AlbedoR; hit.AlbedoG = data.AlbedoG; hit.AlbedoB = data.AlbedoB;
        hit.ObjectIndex = data.ObjectIndex; hit.PrimitiveIndex = primitive;
        hit.TwoSided = data.TwoSided;
        hit.Valid = 1; hit.FrontFace = optixIsFrontFaceHit() ? 1u : 0u;
        return hit;
    }
}
