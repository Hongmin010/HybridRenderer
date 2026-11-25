#pragma once

#include <DirectXMath.h>
#include <optional>

class Transform
{
public:
    void SetPosition(float x, float y, float z)
    {
        mWorld.reset();
        mPosition = {x, y, z};
    }

    void SetRotation(float x, float y, float z)
    {
        mWorld.reset();
        mRotation = {x, y, z};
    }

    void SetScale(float x, float y, float z)
    {
        mWorld.reset();
        mScale = {x, y, z};
    }

    void SetWorldMatrix(const DirectX::XMFLOAT4X4 &world) { mWorld = world; }

    DirectX::XMMATRIX GetWorldMatrix() const
    {
        using namespace DirectX;
        if (mWorld) return XMLoadFloat4x4(&*mWorld);

        XMMATRIX scale =
            XMMatrixScaling(
                mScale.x,
                mScale.y,
                mScale.z);

        XMMATRIX rotation =
            XMMatrixRotationRollPitchYaw(
                mRotation.x,
                mRotation.y,
                mRotation.z);

        XMMATRIX translation =
            XMMatrixTranslation(
                mPosition.x,
                mPosition.y,
                mPosition.z);

        return scale * rotation * translation;
    }

private:
    std::optional<DirectX::XMFLOAT4X4> mWorld;
    DirectX::XMFLOAT3 mPosition =
        {
            0.0f,
            0.0f,
            0.0f};

    DirectX::XMFLOAT3 mRotation =
        {
            0.0f,
            0.0f,
            0.0f};

    DirectX::XMFLOAT3 mScale =
        {
            1.0f,
            1.0f,
            1.0f};
};
