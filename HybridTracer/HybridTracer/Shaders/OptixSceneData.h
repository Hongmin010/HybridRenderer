#pragma once

// Host/device geometry ABI. Positions and vertex shading normals are already world-space.
struct GasVertex { float X,Y,Z,Padding; };
struct GasTriangle { unsigned int X,Y,Z; };
struct GasNormal { float X,Y,Z,Padding; };
struct SceneHitData
{
    const GasVertex *Vertices;
    const GasTriangle *Indices;
    const GasNormal *Normals;
    unsigned int ObjectIndex, TriangleCount;
    float AlbedoR,AlbedoG,AlbedoB,Padding;
};
static_assert(sizeof(GasVertex) == 16, "GasVertex ABI");
static_assert(sizeof(GasTriangle) == 12, "GasTriangle ABI");
static_assert(sizeof(GasNormal) == 16, "GasNormal ABI");
static_assert(sizeof(SceneHitData) == 48, "SceneHitData ABI");
