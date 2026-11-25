#pragma once

#include <DirectXMath.h>

struct Material
{
    DirectX::XMFLOAT3 Albedo =
        {
            1.0f,
            1.0f,
            1.0f};
    bool TwoSided = false;
    unsigned int EmitterLight = 0; // One-based index into Scene::GetAdditionalLights(); zero means no emission.
};
