#include "DepthPositionValidation.h"
#include <DirectXPackedVector.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace DirectX;

namespace
{
    constexpr double PixelTolerance = 0.01;
    constexpr double DepthTolerance = 0.000002;
    constexpr double GeometryTolerance = 0.002;

    struct Point
    {
        double X, Y, Z;
        Point operator+(const Point &p) const { return {X+p.X, Y+p.Y, Z+p.Z}; }
        Point operator-(const Point &p) const { return {X-p.X, Y-p.Y, Z-p.Z}; }
        Point operator*(double s) const { return {X*s, Y*s, Z*s}; }
    };
    double Dot(Point a, Point b) { return a.X*b.X + a.Y*b.Y + a.Z*b.Z; }
    Point Cross(Point a, Point b) { return {a.Y*b.Z-a.Z*b.Y, a.Z*b.X-a.X*b.Z, a.X*b.Y-a.Y*b.X}; }
    double Length(Point p) { return std::sqrt(Dot(p,p)); }

    struct Triangle { Point A, B, C; size_t Object; };
    std::vector<Triangle> GetTriangles(const Scene &scene)
    {
        std::vector<Triangle> triangles;
        const auto &objects = scene.GetObjects();
        for (size_t objectIndex = 0; objectIndex < objects.size(); ++objectIndex)
        {
            const auto &object = objects[objectIndex];
            const auto &vertices = object.MeshData->GetVertices();
            const auto &indices = object.MeshData->GetIndices();
            const auto world = object.ObjectTransform.GetWorldMatrix();
            const auto transform = [&](uint16_t index)
            {
                XMFLOAT3 value;
                XMStoreFloat3(&value, XMVector3TransformCoord(XMLoadFloat3(&vertices[index].Position), world));
                return Point{value.x, value.y, value.z};
            };
            for (size_t i = 0; i < indices.size(); i += 3)
                triangles.push_back({transform(indices[i]), transform(indices[i+1]), transform(indices[i+2]), objectIndex});
        }
        return triangles;
    }

    struct Hit
    {
        double Distance = std::numeric_limits<double>::infinity();
        double Margin = 0;
        size_t Object = 0;
    };
    Hit Trace(Point origin, Point direction, const std::vector<Triangle> &triangles)
    {
        Hit closest;
        for (const auto &triangle : triangles)
        {
            const Point e1 = triangle.B-triangle.A, e2 = triangle.C-triangle.A;
            const Point h = Cross(direction, e2);
            const double determinant = Dot(e1, h);
            if (std::abs(determinant) < 1e-12) continue;
            const Point s = origin-triangle.A;
            const double u = Dot(s, h)/determinant;
            if (u < 0 || u > 1) continue;
            const Point q = Cross(s, e1);
            const double v = Dot(direction, q)/determinant;
            if (v < 0 || u+v > 1) continue;
            const double distance = Dot(e2, q)/determinant;
            if (distance <= 0 || distance >= closest.Distance) continue;
            closest = {distance, std::min({u, v, 1-u-v}), triangle.Object};
        }
        return closest;
    }

    void Fail(UINT frame, UINT x, UINT y, const std::string &reason)
    {
        std::ostringstream message;
        message << "Position frame " << frame << ", pixel(" << x << ',' << y << "): " << reason;
        throw std::runtime_error(message.str());
    }
}

void ValidateDepthPositions(const float *depth, const WorldPosition *positions,
    const uint16_t *normalHalf, const std::vector<float> &dxDepth,
    const Camera &camera, const Scene &scene, UINT width, UINT height, UINT frame,
    std::ostream *samples, const SurfaceInput *surface, std::ostream *surfaceSamples)
{
    XMFLOAT4X4 vp, invView, projection;
    XMStoreFloat4x4(&vp, camera.GetView()*camera.GetProjection());
    XMStoreFloat4x4(&invView, XMMatrixInverse(nullptr, camera.GetView()));
    XMStoreFloat4x4(&projection, camera.GetProjection());
    size_t foreground = 0;
    double maxPixelError = 0, maxDepthError = 0;
    for (UINT y = 0; y < height; ++y)
    {
        for (UINT x = 0; x < width; ++x)
        {
            const size_t i = static_cast<size_t>(y)*width+x;
            const float d = depth[i];
            if (std::memcmp(&d, &dxDepth[i], sizeof(float)) != 0)
                Fail(frame, x, y, "CUDA/DX12 depth bits differ.");
            if (!std::isfinite(d) || d < 0 || d > 1)
                Fail(frame, x, y, "Depth is outside finite [0,1].");
            const auto &p = positions[i];
            if (!std::isfinite(p.X) || !std::isfinite(p.Y) || !std::isfinite(p.Z) || !std::isfinite(p.Valid))
                Fail(frame, x, y, "Non-finite reconstructed position.");
            if (d == 1)
            {
                if (p.X != 0 || p.Y != 0 || p.Z != 0 || p.Valid != 0 || normalHalf[i*4+3] != 0)
                    Fail(frame, x, y, "Background must have zero position/valid flag and zero normal alpha.");
                continue;
            }
            if (p.Valid != 1 || normalHalf[i*4+3] != 0x3c00)
                Fail(frame, x, y, "Depth/normal surface validity disagrees.");
            ++foreground;
            // Forward projection is checked for every GPU position; no inverse formula is reused here.
            const double cx = p.X*double(vp._11)+p.Y*double(vp._21)+p.Z*double(vp._31)+vp._41;
            const double cy = p.X*double(vp._12)+p.Y*double(vp._22)+p.Z*double(vp._32)+vp._42;
            const double cz = p.X*double(vp._13)+p.Y*double(vp._23)+p.Z*double(vp._33)+vp._43;
            const double cw = p.X*double(vp._14)+p.Y*double(vp._24)+p.Z*double(vp._34)+vp._44;
            if (cw <= 0) Fail(frame, x, y, "Reconstructed surface is behind the camera.");
            const double screenX = (cx/cw+1)*0.5*width;
            const double screenY = (1-cy/cw)*0.5*height;
            const double pixelError = std::max(std::abs(screenX-(x+0.5)), std::abs(screenY-(y+0.5)));
            const double depthError = std::abs(cz/cw-d);
            maxPixelError = std::max(maxPixelError, pixelError);
            maxDepthError = std::max(maxDepthError, depthError);
            if (pixelError > PixelTolerance || depthError > DepthTolerance)
                Fail(frame, x, y, "World position does not reproject to its original pixel/depth.");
        }
    }
    if (!foreground) throw std::runtime_error("No visible surfaces for position validation.");

    // Camera ray uses inverse View axes and projection scales, independently of depth and inverse VP.
    const Point origin = {invView._41, invView._42, invView._43};
    const Point right = {invView._11, invView._12, invView._13};
    const Point up = {invView._21, invView._22, invView._23};
    const Point forward = {invView._31, invView._32, invView._33};
    const auto triangles = GetTriangles(scene);
    std::vector<size_t> objectCounts(scene.GetObjects().size());
    std::vector<bool> recorded(objectCounts.size(), false);
    size_t raySamples = 0;
    double maxGeometryError = 0;
    for (UINT y = 3; y < height; y += 11)
    {
        for (UINT x = 4; x < width; x += 13)
        {
            const double vx = (2*(x+0.5)/width-1)/projection._11;
            const double vy = (1-2*(y+0.5)/height)/projection._22;
            const Point rawDirection = right*vx+up*vy+forward;
            const Point direction = rawDirection*(1/Length(rawDirection));
            const Hit hit = Trace(origin, direction, triangles);
            // Raster coverage near edges is affected by fixed-point rasterization/top-left rules.
            // Only interior triangle samples are used for the independent geometric reference.
            if (!std::isfinite(hit.Distance) || hit.Margin < 0.025) continue;
            const size_t i = static_cast<size_t>(y)*width+x;
            if (positions[i].Valid != 1) Fail(frame, x, y, "CPU ray hits a triangle interior, CUDA position is invalid.");
            const Point expected = origin+direction*hit.Distance;
            const auto &p = positions[i];
            const double error = Length(Point{p.X,p.Y,p.Z}-expected);
            maxGeometryError = std::max(maxGeometryError, error);
            if (error > GeometryTolerance) Fail(frame, x, y, "CUDA position differs from CPU nearest ray/triangle hit by " + std::to_string(error));
            if (surface)
            {
                const auto &s = surface[i];
                const auto *material = scene.GetObjects()[hit.Object].MaterialData;
                if (!material) throw std::runtime_error("Surface validation expects a material on every scene object.");
                const float expectedColor[] = {material->Albedo.x, material->Albedo.y, material->Albedo.z};
                const float actualColor[] = {s.AlbedoR, s.AlbedoG, s.AlbedoB};
                for (size_t c = 0; c < 3; ++c)
                    if (std::abs(actualColor[c]-expectedColor[c]) > 0.5f/255.0f+0.000001f)
                        Fail(frame, x, y, "Surface albedo does not match the CPU nearest-hit material.");
            }
            ++raySamples;
            ++objectCounts[hit.Object];
            if (!recorded[hit.Object])
            {
                recorded[hit.Object] = true;
                if (frame == 1)
                    std::cout << "Position object " << hit.Object << " pixel(" << x << ',' << y << "): GPU=("
                              << p.X << ',' << p.Y << ',' << p.Z << "), CPU hit=("
                              << expected.X << ',' << expected.Y << ',' << expected.Z << "), error=" << error << '\n';
                if (samples)
                    *samples << frame << ',' << hit.Object << ',' << x << ',' << y << ',' << depth[i] << ','
                             << p.X << ',' << p.Y << ',' << p.Z << ',' << expected.X << ',' << expected.Y << ','
                             << expected.Z << ',' << error << '\n';
                if (surfaceSamples && surface)
                {
                    const auto &s = surface[i];
                    const auto &a = scene.GetObjects()[hit.Object].MaterialData->Albedo;
                    *surfaceSamples << frame << ',' << hit.Object << ',' << x << ',' << y << ','
                        << s.Position.X << ',' << s.Position.Y << ',' << s.Position.Z << ','
                        << s.NormalX << ',' << s.NormalY << ',' << s.NormalZ << ','
                        << s.AlbedoR << ',' << s.AlbedoG << ',' << s.AlbedoB << ','
                        << a.x << ',' << a.y << ',' << a.z << ',' << error << '\n';
                    if (frame == 1)
                        std::cout << "Surface object " << hit.Object << ": N=(" << s.NormalX << ','
                            << s.NormalY << ',' << s.NormalZ << "), Albedo=(" << s.AlbedoR << ','
                            << s.AlbedoG << ',' << s.AlbedoB << ") matches nearest-hit material.\n";
                }
            }
        }
    }
    for (size_t object = 0; object < objectCounts.size(); ++object)
        if (objectCounts[object] == 0) throw std::runtime_error("CPU ray validation did not cover scene object " + std::to_string(object));
    std::cout << "Position frame " << frame << ": " << foreground << " surfaces; depth bit-exact; max screen error="
              << maxPixelError << " px, max depth error=" << maxDepthError << "; " << raySamples
              << " CPU ray hits, max world error=" << maxGeometryError << '\n';
}

void ValidateSurfaceInputs(const SurfaceInput *surface, const WorldPosition *positions,
    const uint16_t *normalHalf, const uint8_t *albedo, const std::vector<uint8_t> &dxAlbedo,
    UINT width, UINT height, UINT frame)
{
    float maxNormalError = 0, maxUnitError = 0;
    size_t foreground = 0;
    for (UINT y = 0; y < height; ++y)
    {
        for (UINT x = 0; x < width; ++x)
        {
            const size_t i = static_cast<size_t>(y)*width+x;
            if (std::memcmp(albedo+i*4, dxAlbedo.data()+i*4, 4) != 0)
                Fail(frame, x, y, "CUDA/DX12 Albedo bytes differ.");
            const auto &s = surface[i];
            const float values[] = {s.Position.X,s.Position.Y,s.Position.Z,s.Position.Valid,
                s.NormalX,s.NormalY,s.NormalZ,s.NormalPadding,s.AlbedoR,s.AlbedoG,s.AlbedoB,s.AlbedoA};
            for (const auto value : values)
                if (!std::isfinite(value)) Fail(frame, x, y, "Non-finite surface input.");
            if (std::memcmp(&s.Position, positions+i, sizeof(WorldPosition)) != 0)
                Fail(frame, x, y, "Surface position differs from the CUDA position buffer.");
            if (s.Position.Valid == 0)
            {
                for (const auto value : values)
                    if (value != 0) Fail(frame, x, y, "Background surface input must be entirely zero.");
                for (size_t c = 0; c < 4; ++c)
                    if (albedo[i*4+c] != 0) Fail(frame, x, y, "Background Albedo bytes must be zero.");
                continue;
            }
            ++foreground;
            float n[3];
            for (size_t c = 0; c < 3; ++c)
                n[c] = PackedVector::XMConvertHalfToFloat(normalHalf[i*4+c]);
            const float length = std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
            if (length == 0) Fail(frame, x, y, "Surface input has a zero-length source normal.");
            const float actualNormal[] = {s.NormalX,s.NormalY,s.NormalZ};
            for (size_t c = 0; c < 3; ++c)
                maxNormalError = std::max(maxNormalError, std::abs(actualNormal[c]-n[c]/length));
            const float unitLength = std::sqrt(s.NormalX*s.NormalX+s.NormalY*s.NormalY+s.NormalZ*s.NormalZ);
            maxUnitError = std::max(maxUnitError, std::abs(unitLength-1));
            if (maxNormalError > 0.00001f || maxUnitError > 0.00001f || s.NormalPadding != 0)
                Fail(frame, x, y, "Surface normal does not match normalized G-buffer normal.");
            const float a[] = {s.AlbedoR,s.AlbedoG,s.AlbedoB,s.AlbedoA};
            for (size_t c = 0; c < 4; ++c)
                if (std::abs(a[c]-albedo[i*4+c]/255.0f) > 0.0000001f)
                    Fail(frame, x, y, "Surface Albedo differs from RGBA8_UNORM decoding.");
            if (s.AlbedoA != 1) Fail(frame, x, y, "Surface albedo alpha must equal one.");
        }
    }
    std::cout << "Surface frame " << frame << ": " << width*height
        << " RGBA8 pixels byte-exact; " << foreground << " coherent P/N/Albedo inputs; max normal error="
        << maxNormalError << ", max |N length-1|=" << maxUnitError << '\n';
}
