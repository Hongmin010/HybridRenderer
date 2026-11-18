//
// Object / Material
//

cbuffer ObjectConstants : register(b0)
{
    float4x4 gWorldViewProjection;
    float4x4 gWorldInverseTranspose;

    float4 gAlbedo;
};

//
// Scene / Light
//

cbuffer PassConstants : register(b1)
{
    // xyz = surface -> light
    // w   = intensity
    float4 gLightDirectionIntensity;

    // xyz = light color
    // w   = unused
    float4 gLightColorUnused;
};

struct VSInput
{
    float3 Position : POSITION;
    float3 Normal   : NORMAL;
};

struct PSInput
{
    float4 Position : SV_POSITION;
    float3 NormalW  : NORMAL;
};

struct PSOutput
{
    //
    // RT0
    // 화면 확인용 shaded color
    //

    float4 Color : SV_Target0;

    //
    // RT1
    // Material Albedo
    //

    float4 Albedo : SV_Target1;

    //
    // RT2
    // World Space Normal
    //

    float4 Normal : SV_Target2;

    //
    // RT3
    // D3D depth [0, 1]
    //

    float Depth : SV_Target3;
};

PSInput VSMain(VSInput input)
{
    PSInput output;

    //
    // Local -> Clip
    //

    output.Position =
        mul(
            float4(
                input.Position,
                1.0f
            ),

            gWorldViewProjection
        );

    //
    // Local Normal -> World Normal
    //

    output.NormalW =
        mul(
            input.Normal,

            (float3x3)
                gWorldInverseTranspose
        );

    output.NormalW =
        normalize(
            output.NormalW
        );

    return output;
}

PSOutput PSMain(
    PSInput input)
{
    PSOutput output;

    float3 normal =
        normalize(
            input.NormalW
        );

    float3 lightDirection =
        normalize(
            gLightDirectionIntensity.xyz
        );

    float lightIntensity =
        gLightDirectionIntensity.w;

    float3 lightColor =
        gLightColorUnused.xyz;

    //
    // Lambert lighting
    //

    float diffuse =
        saturate(
            dot(
                normal,
                lightDirection
            )
        );

    float lighting =
        diffuse * lightIntensity * 0.3183098861837907f;

    //
    // RT0
    // 현재 화면에 보이는 결과
    //

    float3 shadedColor =
        gAlbedo.rgb *
        lightColor *
        lighting;

    output.Color =
        float4(
        shadedColor,
        1.0f
    );
    //
    // RT1
    // Albedo
    //

    output.Albedo =
        gAlbedo;

    //
    // RT2
    // World Space Normal
    //
    // R16G16B16A16_FLOAT라
    // -1 ~ +1 그대로 저장 가능
    //

    output.Normal =
        float4(
            normal,
            1.0f
        );

    //
    // RT3
    //
    // Pixel Shader의 SV_Position.z는
    // 현재 D3D depth 값이다.
    //
    // World 거리나 linear depth가 아니다.
    //

    output.Depth =
        input.Position.z;

    return output;
}