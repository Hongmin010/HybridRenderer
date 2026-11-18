#pragma once

#include <DirectXMath.h>
#include <variant>

struct DirectionalLight
{
    DirectX::XMFLOAT3 Direction =
        {
            -0.5f,
            1.0f,
            -0.5f};

    DirectX::XMFLOAT3 Color =
        {
            1.0f,
            1.0f,
            1.0f};

    // Irradiance on a plane perpendicular to Direction. Lambert BRDF is albedo/pi.
    float Intensity = 3.14159265358979323846f;
};

struct PointLight
{
    DirectX::XMFLOAT3 Position = {0, 3, 0};
    DirectX::XMFLOAT3 Color = {1, 1, 1};
    float Intensity = 12.0f; // Radiant intensity, W/sr.
};

struct RectangleLight
{
    DirectX::XMFLOAT3 Center = {0, 4, 0};
    // Half edges in world units. Cross(HalfU, HalfV) points toward the emitting side.
    DirectX::XMFLOAT3 HalfU = {1, 0, 0};
    DirectX::XMFLOAT3 HalfV = {0, 0, 1};
    DirectX::XMFLOAT3 Color = {1, 1, 1};
    float Radiance = 4.0f; // W/(sr m^2).
};

using SceneLight = std::variant<DirectionalLight, PointLight, RectangleLight>;
