#pragma once

#include "Scene.h"
#include <filesystem>

class Camera;

struct SceneCamera
{
    DirectX::XMFLOAT3 Position, Forward, Up;
    float Fov, Near, Far;
    bool HorizontalFov;
    unsigned int Width, Height;

    float VerticalFov(float aspect) const;
    void Apply(Camera &camera, unsigned int width, unsigned int height, float xOffset = 0) const;
};

struct SceneDescription
{
    std::vector<SceneObjectDescription> Objects;
    std::vector<SceneLight> Lights;
    std::optional<SceneCamera> Camera;

    void Apply(Scene &scene, ID3D12Device *device) const;
};

SceneDescription ReadSceneFile(const std::filesystem::path &path);
std::filesystem::path DefaultSceneFile();
