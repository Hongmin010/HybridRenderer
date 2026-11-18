#pragma once

#include <DirectXMath.h>

class Transform
{
public:
    void SetPosition(float x, float y, float z)
    {
        mPosition = {x, y, z};
    }

    void SetRotation(float x, float y, float z)
    {
        mRotation = {x, y, z};
    }

    void SetScale(float x, float y, float z)
    {
        mScale = {x, y, z};
    }

    DirectX::XMMATRIX GetWorldMatrix() const
    {
        using namespace DirectX;

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