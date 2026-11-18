#pragma once

// Unit world direction, finite origin and 0 <= TMin < TMax. Invalid records are skipped.
struct SurfaceRay
{
    float OriginX,OriginY,OriginZ,TMin;
    float DirectionX,DirectionY,DirectionZ,TMax;
    unsigned int Valid,Padding0,Padding1,Padding2;
};
// Unflipped normals: Ng follows triangle winding; Ns follows original vertex normals.
// ObjectIndex and PrimitiveIndex are zero-based. Check Valid before reading any fields.
struct SurfaceHit
{
    float PositionX,PositionY,PositionZ,Distance;
    float GeometricNormalX,GeometricNormalY,GeometricNormalZ,BarycentricU;
    float ShadingNormalX,ShadingNormalY,ShadingNormalZ,BarycentricV;
    float AlbedoR,AlbedoG,AlbedoB;
    unsigned int ObjectIndex;
    unsigned int PrimitiveIndex,Valid,FrontFace,Padding;
};
static_assert(sizeof(SurfaceRay) == 48, "SurfaceRay ABI");
static_assert(sizeof(SurfaceHit) == 80, "SurfaceHit ABI");
