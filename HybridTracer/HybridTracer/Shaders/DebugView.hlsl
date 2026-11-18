Texture2D<float4> gAlbedo : register(t0);
Texture2D<float4> gNormal : register(t1);
Texture2D<float>  gDepth  : register(t2);

cbuffer DebugSettings : register(b0)
{
    uint gMode;
};

struct VSOutput
{
    float4 Position : SV_POSITION;
};

VSOutput VSMain(uint vertexID : SV_VertexID)
{
    float2 positions[3] =
    {
        float2(-1.0f, -1.0f),
        float2(-1.0f,  3.0f),
        float2( 3.0f, -1.0f)
    };

    VSOutput output;

    output.Position =
        float4(positions[vertexID], 0.0f, 1.0f);

    return output;
}

float4 PSMain(VSOutput input) : SV_Target0
{
    int3 pixel = int3(input.Position.xy, 0);

    float depth = gDepth.Load(pixel);

    // 배경은 기존 배경색으로 표시
    if (depth >= 1.0f)
    {
        return float4(
            0.08f,
            0.12f,
            0.18f,
            1.0f
        );
    }

    //
    // Mode 1 = Albedo
    //

    if (gMode == 1)
    {
        return gAlbedo.Load(pixel);
    }

    //
    // Mode 2 = World Normal
    //

    if (gMode == 2)
    {
        float3 normal =
            gNormal.Load(pixel).xyz;

        return float4(
            normal * 0.5f + 0.5f,
            1.0f
        );
    }

    //
    // Mode 3 = Depth
    //
    // 원래 Depth를 더 잘 보이게
    // 대비만 확대하는 Debug 표현
    //

    float visualDepth =
        saturate((1.0f - depth) * 20.0f);

    return float4(
        visualDepth,
        visualDepth,
        visualDepth,
        1.0f
    );
}