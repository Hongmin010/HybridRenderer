#include "Scene.h"
#include <stdexcept>

using namespace DirectX;

void Scene::AddObject(std::unique_ptr<Mesh> mesh, std::unique_ptr<Material> material, const Transform &transform)
{
    if (!mesh || !material) throw std::runtime_error("Scene object needs mesh and material ownership.");
    mMeshes.reserve(mMeshes.size()+1);
    mMaterials.reserve(mMaterials.size()+1);
    mObjects.reserve(mObjects.size()+1);
    const SceneObject object = {mesh.get(),material.get(),transform};
    mMeshes.push_back(std::move(mesh));
    mMaterials.push_back(std::move(material));
    mObjects.push_back(object);
}

void Scene::Initialize(ID3D12Device *device)
{
    Initialize(device, {
        {SceneShape::Plane, {0, -1, 0.5f}, {0, 0, 0}, {4, 1, 4}, {0.45f, 0.45f, 0.45f}},
        {SceneShape::Cube, {-1.25f, -0.25f, -0.8f}, {0.1f, -0.3f, 0},
            {0.7f, 0.7f, 0.7f}, {0.85f, 0.10f, 0.08f}},
        {SceneShape::Cube, {1.25f, -0.30f, 1.1f}, {-0.2f, 0.55f, 0.15f},
            {0.65f, 0.65f, 0.65f}, {0.08f, 0.25f, 0.90f}},
        {SceneShape::Plane, {0, 0.3f, 2.2f}, {-0.9f, 0.15f, 0.25f},
            {1, 1, 1.4f}, {0.10f, 0.80f, 0.20f}}
    });
}

void Scene::Initialize(ID3D12Device *device, const std::vector<SceneObjectDescription> &objects)
{
    if (objects.empty() || !mObjects.empty())
        throw std::runtime_error("Scene initialization needs a fresh scene with at least one object.");
    //
    // =========================================================
    // Cube Mesh
    // =========================================================
    //

    std::vector<Vertex> cubeVertices =
        {
            //
            // Front (-Z)
            //

            {
                {-1.0f, -1.0f, -1.0f},
                {0.0f, 0.0f, -1.0f}},
            {{-1.0f, +1.0f, -1.0f},
             {0.0f, 0.0f, -1.0f}},
            {{+1.0f, +1.0f, -1.0f},
             {0.0f, 0.0f, -1.0f}},
            {{+1.0f, -1.0f, -1.0f},
             {0.0f, 0.0f, -1.0f}},

            //
            // Back (+Z)
            //

            {
                {+1.0f, -1.0f, +1.0f},
                {0.0f, 0.0f, +1.0f}},
            {{+1.0f, +1.0f, +1.0f},
             {0.0f, 0.0f, +1.0f}},
            {{-1.0f, +1.0f, +1.0f},
             {0.0f, 0.0f, +1.0f}},
            {{-1.0f, -1.0f, +1.0f},
             {0.0f, 0.0f, +1.0f}},

            //
            // Left (-X)
            //

            {
                {-1.0f, -1.0f, +1.0f},
                {-1.0f, 0.0f, 0.0f}},
            {{-1.0f, +1.0f, +1.0f},
             {-1.0f, 0.0f, 0.0f}},
            {{-1.0f, +1.0f, -1.0f},
             {-1.0f, 0.0f, 0.0f}},
            {{-1.0f, -1.0f, -1.0f},
             {-1.0f, 0.0f, 0.0f}},

            //
            // Right (+X)
            //

            {
                {+1.0f, -1.0f, -1.0f},
                {+1.0f, 0.0f, 0.0f}},
            {{+1.0f, +1.0f, -1.0f},
             {+1.0f, 0.0f, 0.0f}},
            {{+1.0f, +1.0f, +1.0f},
             {+1.0f, 0.0f, 0.0f}},
            {{+1.0f, -1.0f, +1.0f},
             {+1.0f, 0.0f, 0.0f}},

            //
            // Top (+Y)
            //

            {
                {-1.0f, +1.0f, -1.0f},
                {0.0f, +1.0f, 0.0f}},
            {{-1.0f, +1.0f, +1.0f},
             {0.0f, +1.0f, 0.0f}},
            {{+1.0f, +1.0f, +1.0f},
             {0.0f, +1.0f, 0.0f}},
            {{+1.0f, +1.0f, -1.0f},
             {0.0f, +1.0f, 0.0f}},

            //
            // Bottom (-Y)
            //

            {
                {-1.0f, -1.0f, +1.0f},
                {0.0f, -1.0f, 0.0f}},
            {{-1.0f, -1.0f, -1.0f},
             {0.0f, -1.0f, 0.0f}},
            {{+1.0f, -1.0f, -1.0f},
             {0.0f, -1.0f, 0.0f}},
            {{+1.0f, -1.0f, +1.0f},
             {0.0f, -1.0f, 0.0f}}};

    std::vector<std::uint16_t> cubeIndices =
        {
            0, 1, 2,
            0, 2, 3,

            4, 5, 6,
            4, 6, 7,

            8, 9, 10,
            8, 10, 11,

            12, 13, 14,
            12, 14, 15,

            16, 17, 18,
            16, 18, 19,

            20, 21, 22,
            20, 22, 23};

    auto cubeMesh =
        std::make_unique<Mesh>();

    cubeMesh->Initialize(
        device,
        cubeVertices,
        cubeIndices);

    Mesh *cube =
        cubeMesh.get();

    mMeshes.push_back(
        std::move(cubeMesh));

    //
    // =========================================================
    // Plane Mesh
    //
    // Local XZ plane
    // normal = +Y
    // =========================================================
    //

    std::vector<Vertex> planeVertices =
        {
            {{-1.0f, 0.0f, -1.0f},
             {0.0f, 1.0f, 0.0f}},

            {{-1.0f, 0.0f, +1.0f},
             {0.0f, 1.0f, 0.0f}},

            {{+1.0f, 0.0f, +1.0f},
             {0.0f, 1.0f, 0.0f}},

            {{+1.0f, 0.0f, -1.0f},
             {0.0f, 1.0f, 0.0f}}};

    std::vector<std::uint16_t> planeIndices =
        {
            0, 1, 2,
            0, 2, 3};

    auto planeMesh =
        std::make_unique<Mesh>();

    planeMesh->Initialize(
        device,
        planeVertices,
        planeIndices);

    Mesh *plane =
        planeMesh.get();

    mMeshes.push_back(
        std::move(planeMesh));

    mMaterials.reserve(objects.size());
    mObjects.reserve(objects.size());
    for (const auto &description : objects)
    {
        auto material = std::make_unique<Material>();
        material->Albedo = description.Albedo;
        Transform transform;
        const auto &position = description.Position;
        const auto &rotation = description.Rotation;
        const auto &scale = description.Scale;
        transform.SetPosition(position.x,position.y,position.z);
        transform.SetRotation(rotation.x,rotation.y,rotation.z);
        transform.SetScale(scale.x,scale.y,scale.z);
        mObjects.push_back({description.Shape == SceneShape::Cube ? cube : plane,material.get(),transform});
        mMaterials.push_back(std::move(material));
    }
}

void Scene::SetLights(const std::vector<SceneLight> &lights)
{
    DirectionalLight primary;
    primary.Intensity = 0;
    bool hasPrimary = false;
    std::vector<SceneLight> additional;
    additional.reserve(lights.size());
    for (const auto &light : lights)
    {
        if (!hasPrimary)
        {
            if (const auto *directional = std::get_if<DirectionalLight>(&light))
            {
                primary = *directional;
                hasPrimary = true;
                continue;
            }
        }
        additional.push_back(light);
    }
    mDirectionalLight = primary;
    mAdditionalLights = std::move(additional);
}
