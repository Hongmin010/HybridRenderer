#pragma once

#include <DirectXMath.h>

class Camera
{
public:
    Camera();

    void LookAt(
        const DirectX::XMFLOAT3 &position,
        const DirectX::XMFLOAT3 &target,
        const DirectX::XMFLOAT3 &up);

    void SetLens(
        float fovY,
        float aspect,
        float nearZ,
        float farZ);

    DirectX::XMMATRIX GetView() const;
    DirectX::XMMATRIX GetProjection() const;

private:
    DirectX::XMFLOAT4X4 mView;
    DirectX::XMFLOAT4X4 mProjection;
};