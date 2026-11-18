#include "Camera.h"

using namespace DirectX;

Camera::Camera()
{
    XMStoreFloat4x4(
        &mView,
        XMMatrixIdentity());

    XMStoreFloat4x4(
        &mProjection,
        XMMatrixIdentity());
}

void Camera::LookAt(
    const XMFLOAT3 &position,
    const XMFLOAT3 &target,
    const XMFLOAT3 &up)
{
    XMVECTOR pos =
        XMLoadFloat3(&position);

    XMVECTOR targetPos =
        XMLoadFloat3(&target);

    XMVECTOR upVector =
        XMLoadFloat3(&up);

    XMMATRIX view =
        XMMatrixLookAtLH(
            pos,
            targetPos,
            upVector);

    XMStoreFloat4x4(
        &mView,
        view);
}

void Camera::SetLens(
    float fovY,
    float aspect,
    float nearZ,
    float farZ)
{
    XMMATRIX projection =
        XMMatrixPerspectiveFovLH(
            fovY,
            aspect,
            nearZ,
            farZ);

    XMStoreFloat4x4(
        &mProjection,
        projection);
}

XMMATRIX Camera::GetView() const
{
    return XMLoadFloat4x4(&mView);
}

XMMATRIX Camera::GetProjection() const
{
    return XMLoadFloat4x4(&mProjection);
}