#pragma once

#include "Mesh.h"
#include "Transform.h"
#include "Material.h"
#include "Light.h"

#include <memory>
#include <vector>

struct SceneObject
{
    Mesh *MeshData = nullptr;
    Material *MaterialData = nullptr;

    Transform ObjectTransform;
};

enum class SceneShape { Cube, Plane };

struct SceneObjectDescription
{
    SceneShape Shape;
    DirectX::XMFLOAT3 Position = {0, 0, 0};
    DirectX::XMFLOAT3 Rotation = {0, 0, 0};
    DirectX::XMFLOAT3 Scale = {1, 1, 1};
    DirectX::XMFLOAT3 Albedo = {1, 1, 1};
    std::optional<DirectX::XMFLOAT4X4> World;
    bool TwoSided = false;
    unsigned int EmitterLight = 0;
};

class Scene
{
public:
    void Initialize(
        ID3D12Device *device);
    void Initialize(ID3D12Device *device, const std::vector<SceneObjectDescription> &objects);
    // Transfers resource ownership; add before building the static OptiX scene.
    void AddObject(std::unique_ptr<Mesh> mesh, std::unique_ptr<Material> material, const Transform &transform);
    // Configure lights before initializing the hybrid frame pipeline.
    void SetLights(const std::vector<SceneLight> &lights);
    void SetDirectionalLight(const DirectionalLight &light) { mDirectionalLight = light; }
    void AddLight(const SceneLight &light) { mAdditionalLights.push_back(light); }
    const std::vector<SceneLight> &GetAdditionalLights() const { return mAdditionalLights; }

    const std::vector<SceneObject> &
    GetObjects() const
    {
        return mObjects;
    }

    const DirectionalLight &
    GetDirectionalLight() const
    {
        return mDirectionalLight;
    }

private:
    std::vector<std::unique_ptr<Mesh>>
        mMeshes;

    std::vector<std::unique_ptr<Material>>
        mMaterials;

    std::vector<SceneObject>
        mObjects;

    DirectionalLight mDirectionalLight;
    std::vector<SceneLight> mAdditionalLights;
};
