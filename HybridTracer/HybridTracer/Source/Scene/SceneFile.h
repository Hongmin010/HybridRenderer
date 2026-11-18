#pragma once

#include "Scene.h"
#include <filesystem>

struct SceneDescription
{
    std::vector<SceneObjectDescription> Objects;
    std::vector<SceneLight> Lights;

    void Apply(Scene &scene, ID3D12Device *device) const;
};

SceneDescription ReadSceneFile(const std::filesystem::path &path);
std::filesystem::path DefaultSceneFile();
