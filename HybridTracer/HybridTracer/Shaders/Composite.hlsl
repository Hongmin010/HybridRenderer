#include "Fullscreen.hlsli"
Texture2D<float4> Albedo : register(t0);
Texture2D<float4> Normal : register(t1);
Texture2D<float> Depth : register(t2);
struct ShadowResult { float Visibility; uint Blocker; float HitDistance; uint Valid; };
StructuredBuffer<ShadowResult> Shadows : register(t3);
struct IndirectLighting { float R,G,B; uint Valid; };
StructuredBuffer<IndirectLighting> Indirect : register(t4);
cbuffer Lighting : register(b0)
{
    float4 LightDirectionIntensity;
    float4 LightColorUnused;
    uint Width;
    uint UseShadows;
    uint ShowVisibility;
    uint LightingView; // 0=direct+indirect, 1=direct, 2=indirect
#ifdef ADDITIONAL_LIGHTS
    uint PixelCount;
#endif
};
#ifdef ADDITIONAL_LIGHTS
float3 ReadLighting(uint index)
{
    IndirectLighting sample = Indirect[index];
    return sample.Valid ? float3(sample.R,sample.G,sample.B) : 0;
}
#endif

float4 PSMain(VSOutput input) : SV_Target0
{
    int3 pixel = int3(input.Position.xy,0);
    if (Depth.Load(pixel) >= 1.0f) return LightingView == 2 ? float4(0,0,0,1) : float4(0.08f,0.12f,0.18f,1);
    ShadowResult shadow = Shadows[pixel.y * Width + pixel.x];
#ifndef ADDITIONAL_LIGHTS
    if (ShowVisibility) return float4(shadow.Visibility.xxx,1);
#endif
    float visibility = UseShadows ? (shadow.Valid ? shadow.Visibility : 0.0f) : 1.0f;
    float cosine = saturate(dot(normalize(Normal.Load(pixel).xyz),LightDirectionIntensity.xyz));
    // Albedo is linear UNORM material color. Apply it ONCE; do not sample raster's RT0.
    float3 direct = Albedo.Load(pixel).rgb * LightColorUnused.rgb *
        (0.3183098861837907f * cosine * LightDirectionIntensity.w * visibility);
#ifdef ADDITIONAL_LIGHTS
    // Plane order mirrors IndirectLighting.h: indirect, shadowed direct, unshadowed direct.
    const uint index = pixel.y * Width + pixel.x;
    direct += ReadLighting(index + (UseShadows ? 1 : 2)*PixelCount);
    if (ShowVisibility)
    {
        const float3 unshadowed = Albedo.Load(pixel).rgb * LightColorUnused.rgb *
            (0.3183098861837907f * cosine * LightDirectionIntensity.w) + ReadLighting(index + 2*PixelCount);
        const float sum = unshadowed.r + unshadowed.g + unshadowed.b;
        const float fraction = sum > 0 ? saturate((direct.r + direct.g + direct.b)/sum) : 0;
        return float4(fraction.xxx,1);
    }
#endif
    IndirectLighting bounce = Indirect[pixel.y * Width + pixel.x];
    float3 indirect = bounce.Valid ? float3(bounce.R,bounce.G,bounce.B) : 0;
    return float4(LightingView == 1 ? direct : LightingView == 2 ? indirect : direct + indirect,1);
}
