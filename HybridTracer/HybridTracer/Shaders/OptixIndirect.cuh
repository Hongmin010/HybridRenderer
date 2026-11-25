#pragma once
#include "DiffuseBSDF.cuh"

namespace HybridIndirect
{
    using HybridBSDF::Vector3;

    // Integer avalanche mixer; distinct counter dimensions, 24-bit [0,1) conversion.
    __device__ __forceinline__ unsigned int Mix(unsigned int x)
    {
        x ^= x >> 16; x *= 0x7feb352du;
        x ^= x >> 15; x *= 0x846ca68bu;
        return x ^ (x >> 16);
    }
    __device__ __forceinline__ float Uniform(unsigned int pixel, unsigned int sample, unsigned int dimension)
    {
        return float(Mix(Mix(pixel) ^ Mix(params.Indirect.Seed) ^
            Mix(sample*2u+dimension+0x9e3779b9u)) >> 8) * (1.0f/16777216.0f);
    }
    __device__ __forceinline__ float3 Offset(Vector3 position, Vector3 normal, float epsilon)
    {
        return make_float3(position.X+normal.X*epsilon,position.Y+normal.Y*epsilon,
            position.Z+normal.Z*epsilon);
    }

    __device__ __forceinline__ bool VisibleFrom(float3 origin, Vector3 light, float tMin, float tMax)
    {
        unsigned int blocker = 1, distance = 0;
        optixTrace(params.Gas,origin,make_float3(light.X,light.Y,light.Z),tMin,tMax,
            0.0f,OptixVisibilityMask(255),OPTIX_RAY_FLAG_DISABLE_ANYHIT |
            OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,0,1,0,blocker,distance);
        return blocker == 0;
    }

    __device__ __forceinline__ Vector3 ReadVector(const float (&xyz)[3])
    {
        return {xyz[0],xyz[1],xyz[2]};
    }

    __device__ __forceinline__ float LightUniform(unsigned int pixel, unsigned int sample,
        unsigned int vertex, unsigned int lightIndex, unsigned int dimension)
    {
        // Separate counters for light samples and cosine-BSDF samples.
        const auto counter = (vertex*MaxAdditionalLights+lightIndex)*MaxIndirectSamples+sample;
        return float(Mix(Mix(pixel) ^ Mix(params.Indirect.Seed ^ 0x3c6ef372u) ^
            Mix(2u*counter+dimension+0xbb67ae85u)) >> 8) * (1.0f/16777216.0f);
    }

    struct LightContribution
    {
        Vector3 Unoccluded;
        bool Visible;
    };

    __device__ __forceinline__ LightContribution SampleLight(unsigned int pixel, unsigned int sample,
        unsigned int vertex, unsigned int lightIndex, Vector3 position, Vector3 ng, Vector3 ns, Vector3 albedo)
    {
        using namespace HybridBSDF;
        const auto &light = params.AdditionalLights[lightIndex];
        Vector3 direction = ReadVector(light.PositionOrDirection);
        Vector3 target = direction;
        float weight = 1.0f;
        if (light.Kind != TraceLightKind::Directional)
        {
            if (light.Kind == TraceLightKind::Rectangle)
            {
                const auto u = Scale(ReadVector(light.HalfU),2*LightUniform(pixel,sample,vertex,lightIndex,0)-1);
                const auto v = Scale(ReadVector(light.HalfV),2*LightUniform(pixel,sample,vertex,lightIndex,1)-1);
                target = {target.X+u.X+v.X,target.Y+u.Y+v.Y,target.Z+u.Z+v.Z};
            }
            const Vector3 delta = {target.X-position.X,target.Y-position.Y,target.Z-position.Z};
            const float distanceSquared = Dot(delta,delta);
            if (!(distanceSquared > 0) || !isfinite(distanceSquared)) return {};
            direction = Scale(delta,rsqrtf(distanceSquared));
            weight = 1.0f/distanceSquared;
            if (light.Kind == TraceLightKind::Rectangle)
                weight *= light.Area*fmaxf(-Dot(ReadVector(light.Normal),direction),0.0f);
        }
        const float cosine = Dot(ns,direction);
        if (cosine <= 0 || Dot(ng,direction) <= 0 || !(weight > 0) || !isfinite(weight)) return {};
        const float factor = InvPi*cosine*weight;
        const Vector3 unoccluded = {albedo.X*light.Emission[0]*factor,
            albedo.Y*light.Emission[1]*factor,albedo.Z*light.Emission[2]*factor};
        if (unoccluded.X == 0 && unoccluded.Y == 0 && unoccluded.Z == 0) return {};
        const auto &settings = params.Settings;
        const float3 origin = Offset(position,ng,settings.NormalOffset);
        float tMax = settings.TMax;
        if (light.Kind != TraceLightKind::Directional)
        {
            // Aim at the same emitter sample after the origin offset. Geometry behind it cannot shadow it.
            const Vector3 segment = {target.X-origin.x,target.Y-origin.y,target.Z-origin.z};
            const float distance = sqrtf(Dot(segment,segment));
            if (distance <= 2*settings.TMin) return {unoccluded,true};
            direction = Scale(segment,1.0f/distance);
            tMax = fminf(tMax,distance-settings.TMin);
        }
        return {unoccluded,VisibleFrom(origin,direction,settings.TMin,tMax)};
    }

    __device__ __forceinline__ void EvaluateAdditionalDirect(unsigned int pixel, const SurfaceInput &surface)
    {
        using namespace HybridBSDF;
        IndirectLighting direct = {}, unshadowed = {};
        if (surface.Position.Valid == 1.0f)
        {
            direct.Valid = unshadowed.Valid = 1;
            const Vector3 position = {surface.Position.X,surface.Position.Y,surface.Position.Z};
            const Vector3 normal = {surface.NormalX,surface.NormalY,surface.NormalZ};
            const Vector3 albedo = {surface.AlbedoR,surface.AlbedoG,surface.AlbedoB};
            // The first raster hit can see the front side of an emissive rectangle.
            // Secondary diffuse paths use light sampling to account for this emission once.
            const auto emitter = static_cast<unsigned int>(roundf((1.0f-surface.AlbedoA)*255.0f));
            if (emitter && emitter <= params.AdditionalLightCount)
            {
                const auto &radiance = params.AdditionalLights[emitter-1].Emission;
                direct = unshadowed = {radiance[0],radiance[1],radiance[2],1};
            }
            for (unsigned int light = 0; light < params.AdditionalLightCount; ++light)
            {
                const auto samples = params.AdditionalLights[light].Kind == TraceLightKind::Rectangle
                    ? params.Indirect.SamplesPerPixel : 1u;
                const float scale = 1.0f/samples;
                for (unsigned int sample = 0; sample < samples; ++sample)
                {
                    const auto value = SampleLight(pixel,sample,0,light,position,normal,normal,albedo);
                    const auto contribution = Scale(value.Unoccluded,scale);
                    unshadowed.R += contribution.X; unshadowed.G += contribution.Y; unshadowed.B += contribution.Z;
                    if (value.Visible)
                    {
                        direct.R += contribution.X; direct.G += contribution.Y; direct.B += contribution.Z;
                    }
                }
            }
        }
        const auto pixels = params.Width*params.Height;
        params.IndirectOutput[pixel+AdditionalDirectPlane*pixels] = direct;
        params.IndirectOutput[pixel+AdditionalUnshadowedPlane*pixels] = unshadowed;
    }

    __device__ __forceinline__ void Evaluate(unsigned int i, const SurfaceInput &a)
    {
        using namespace HybridBSDF;
        IndirectLighting result = {};
        IndirectSampleDebug debug = {};
        result.Valid = a.Position.Valid == 1.0f ? 1u : 0u;
        if (result.Valid && params.Indirect.MaxBounces != 0)
        {
            result.Valid = 1;
            const auto &settings = params.Settings;
            const Vector3 nA = {a.NormalX,a.NormalY,a.NormalZ};
            const Vector3 rhoA = {a.AlbedoR,a.AlbedoG,a.AlbedoB};
            const Vector3 woA = Normalize({params.Indirect.CameraX-a.Position.X,
                params.Indirect.CameraY-a.Position.Y,params.Indirect.CameraZ-a.Position.Z});
            const Vector3 light = {settings.LightX,settings.LightY,settings.LightZ};
            for (unsigned int sampleIndex = 0; sampleIndex < params.Indirect.SamplesPerPixel; ++sampleIndex)
            {
                debug = {};
                Vector3 position = {a.Position.X,a.Position.Y,a.Position.Z};
                Vector3 ng = nA, ns = nA, wo = woA, rho = rhoA;
                Vector3 throughput = {1,1,1};
                for (unsigned int bounce = 0; bounce < params.Indirect.MaxBounces; ++bounce)
                {
                    // Reserve disjoint random counters per bounce, preserving bounce zero's samples.
                    const auto counter = sampleIndex+bounce*MaxIndirectSamples;
                    const auto sample = SampleDiffuse(rho,ns,wo,Uniform(i,counter,0),Uniform(i,counter,1));
                    if (!sample.Valid) break;
                    const auto d = sample.Direction;
                    if (bounce && Dot(ng,d) <= 0) break;
                    throughput = {throughput.X*sample.Weight.X,throughput.Y*sample.Weight.Y,
                        throughput.Z*sample.Weight.Z};
                    const float3 origin = Offset(position,ng,settings.NormalOffset);
                    const bool record = bounce == 0 && params.IndirectDebug;
                    if (record)
                    {
                        debug.Ray = {origin.x,origin.y,origin.z,settings.TMin,d.X,d.Y,d.Z,settings.TMax,1,0,0,0};
                        debug.WeightR=sample.Weight.X; debug.WeightG=sample.Weight.Y; debug.WeightB=sample.Weight.Z;
                        debug.Pdf=sample.Pdf; debug.SampleValid=1;
                    }
                    params.QueryHits[i] = {};
                    unsigned int blocker = 0, distance = 0;
                    optixTrace(params.Gas,origin,make_float3(d.X,d.Y,d.Z),settings.TMin,settings.TMax,
                        0.0f,OptixVisibilityMask(255),OPTIX_RAY_FLAG_DISABLE_ANYHIT,0,1,0,blocker,distance);
                    const SurfaceHit hit = params.QueryHits[i];
                    if (record) debug.Hit = hit;
                    if (!hit.Valid || (!hit.FrontFace && !hit.TwoSided)) break;
                    position = {hit.PositionX,hit.PositionY,hit.PositionZ};
                    ng = {hit.GeometricNormalX,hit.GeometricNormalY,hit.GeometricNormalZ};
                    if (!hit.FrontFace) ng = Scale(ng,-1);
                    ns = {hit.ShadingNormalX,hit.ShadingNormalY,hit.ShadingNormalZ};
                    if (Dot(ns,ng) < 0) ns = Scale(ns,-1);
                    wo = Scale(d,-1);
                    if (Dot(ng,wo) <= 0) break;
                    rho = {hit.AlbedoR,hit.AlbedoG,hit.AlbedoB};

                    // A surface without a direct contribution can still scatter the path onward.
                    if (Dot(ng,light) > 0)
                    {
                        const auto f = EvalDiffuseBRDF(rho,ns,light,wo);
                        if (f.X != 0 || f.Y != 0 || f.Z != 0)
                        {
                            const float cosine = fmaxf(Dot(ns,light),0.0f);
                            const float visibility = VisibleFrom(Offset(position,ng,settings.NormalOffset),light,settings.TMin,settings.TMax)
                                ? 1.0f : 0.0f;
                            if (record) debug.Visibility = visibility;
                            const float factor = cosine*visibility;
                            result.R += throughput.X*f.X*params.Indirect.IrradianceR*factor;
                            result.G += throughput.Y*f.Y*params.Indirect.IrradianceG*factor;
                            result.B += throughput.Z*f.Z*params.Indirect.IrradianceB*factor;
                        }
                    }
                    if (params.AdditionalLightCount && Dot(ns,wo) > 0)
                    {
                        for (unsigned int lightIndex = 0; lightIndex < params.AdditionalLightCount; ++lightIndex)
                        {
                            const auto extra = SampleLight(i,sampleIndex,bounce+1,lightIndex,position,ng,ns,rho);
                            if (!extra.Visible) continue;
                            result.R += throughput.X*extra.Unoccluded.X;
                            result.G += throughput.Y*extra.Unoccluded.Y;
                            result.B += throughput.Z*extra.Unoccluded.Z;
                        }
                    }
                    if (throughput.X == 0 && throughput.Y == 0 && throughput.Z == 0) break;
                }
            }
            const float invCount = 1.0f/params.Indirect.SamplesPerPixel;
            result.R *= invCount; result.G *= invCount; result.B *= invCount;
        }
        params.IndirectOutput[i] = result;
        if (params.IndirectDebug) params.IndirectDebug[i] = debug;
    }
}
