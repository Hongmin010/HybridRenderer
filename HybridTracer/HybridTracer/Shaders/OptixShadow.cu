#include <optix.h>
#include "OptixTraceShared.h"
#include "OptixSurface.cuh"

extern "C" { __constant__ OptixTraceLaunchParams params; }
#include "OptixIndirect.cuh"

extern "C" __global__ void __raygen__shadow()
{
    const uint3 pixel = optixGetLaunchIndex();
    const unsigned int i = pixel.y * params.Width + pixel.x;
    if (params.QueryRays)
    {
        params.QueryHits[i] = {};
        const auto ray = params.QueryRays[i];
        if (HybridSurface::ValidRay(ray))
        {
            unsigned int blocker = 0, distance = 0;
            optixTrace(params.Gas,make_float3(ray.OriginX,ray.OriginY,ray.OriginZ),
                make_float3(ray.DirectionX,ray.DirectionY,ray.DirectionZ),ray.TMin,ray.TMax,
                0.0f,OptixVisibilityMask(255),OPTIX_RAY_FLAG_DISABLE_ANYHIT,0,1,0,blocker,distance);
        }
        return;
    }
    const SurfaceInput s = params.Surfaces[i];
    ShadowResult result = {};
    if (s.Position.Valid == 1.0f)
    {
        const ShadowSettings settings = params.Settings;
        const float3 origin = make_float3(
            s.Position.X + s.NormalX * settings.NormalOffset,
            s.Position.Y + s.NormalY * settings.NormalOffset,
            s.Position.Z + s.NormalZ * settings.NormalOffset);
        const float3 direction = make_float3(settings.LightX, settings.LightY, settings.LightZ);
        unsigned int blocker = 0, distance = 0;
        // Closest hit is retained for independent distance/SBT validation in this first pass.
        // Both triangle sides are opaque. No primary camera ray or recursive lighting here.
        optixTrace(params.Gas, origin, direction, settings.TMin, settings.TMax,
            0.0f, OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 1, 0, blocker, distance);
        result.Visibility = blocker == 0 ? 1.0f : 0.0f;
        result.Blocker = blocker;
        result.HitDistance = __uint_as_float(distance);
        result.Valid = 1;
    }
    params.Output[i] = result;
    if (params.IndirectOutput)
    {
        if (params.AdditionalLightCount) HybridIndirect::EvaluateAdditionalDirect(i,s);
        HybridIndirect::Evaluate(i,s);
    }
}

extern "C" __global__ void __miss__shadow()
{
    optixSetPayload_0(0);
    optixSetPayload_1(__float_as_uint(0.0f));
}

extern "C" __global__ void __closesthit__shadow()
{
    const auto *data = reinterpret_cast<const SceneHitData *>(optixGetSbtDataPointer());
    optixSetPayload_0(data->ObjectIndex + 1);
    optixSetPayload_1(__float_as_uint(optixGetRayTmax()));
    if (params.QueryHits)
    {
        const auto pixel = optixGetLaunchIndex();
        params.QueryHits[pixel.y*params.Width+pixel.x] = HybridSurface::ReadCurrentHit(*data);
    }
}
