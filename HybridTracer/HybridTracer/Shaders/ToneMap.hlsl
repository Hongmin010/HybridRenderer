#include "Fullscreen.hlsli"
Texture2D<float4> HDR : register(t0);
cbuffer DisplaySettings : register(b0) { float Exposure; uint RawVisibility; };
float3 LinearToSRGB(float3 x)
{
    return float3(x.r <= 0.0031308f ? x.r*12.92f : 1.055f*pow(abs(x.r),1.0f/2.4f)-0.055f,
                  x.g <= 0.0031308f ? x.g*12.92f : 1.055f*pow(abs(x.g),1.0f/2.4f)-0.055f,
                  x.b <= 0.0031308f ? x.b*12.92f : 1.055f*pow(abs(x.b),1.0f/2.4f)-0.055f);
}
float4 PSMain(VSOutput input) : SV_Target0
{
    float3 color = max(HDR.Load(int3(input.Position.xy,0)).rgb,0.0f);
    if (RawVisibility) return float4(saturate(color),1);
    color *= Exposure;
    color = color/(1.0f+color); // Reinhard, fixed exposure by default.
    return float4(LinearToSRGB(color),1); // UNORM RTV, no second sRGB conversion.
}
