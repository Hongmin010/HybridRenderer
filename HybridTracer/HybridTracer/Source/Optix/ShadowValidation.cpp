#include "ShadowValidation.h"
#include "Scene/Scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace DirectX;

namespace
{
    struct Point
    {
        double X, Y, Z;
        Point operator-(Point b) const { return {X-b.X,Y-b.Y,Z-b.Z}; }
    };
    double Dot(Point a, Point b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
    Point Cross(Point a, Point b) { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
    struct Triangle { Point A,B,C; unsigned int Object; };
    std::vector<Triangle> GetTriangles(const Scene &scene)
    {
        std::vector<Triangle> triangles;
        const auto &objects = scene.GetObjects();
        for (size_t i = 0; i < objects.size(); ++i)
        {
            const auto &object = objects[i];
            XMFLOAT4X4 m;
            XMStoreFloat4x4(&m, object.ObjectTransform.GetWorldMatrix());
            const auto &vertices = object.MeshData->GetVertices();
            const auto &indices = object.MeshData->GetIndices();
            const auto transform = [&](uint16_t index)
            {
                const auto &v = vertices[index].Position;
                // Independent scalar transform of the original mesh; no GAS buffer readback.
                return Point{double(v.x)*m._11+double(v.y)*m._21+double(v.z)*m._31+m._41,
                    double(v.x)*m._12+double(v.y)*m._22+double(v.z)*m._32+m._42,
                    double(v.x)*m._13+double(v.y)*m._23+double(v.z)*m._33+m._43};
            };
            for (size_t j = 0; j < indices.size(); j += 3)
                triangles.push_back({transform(indices[j]),transform(indices[j+1]),transform(indices[j+2]),static_cast<unsigned int>(i)});
        }
        return triangles;
    }
    double Intersect(Point origin, Point direction, const Triangle &triangle, const ShadowSettings &settings)
    {
        const auto e1 = triangle.B-triangle.A, e2 = triangle.C-triangle.A;
        const auto h = Cross(direction, e2);
        const double determinant = Dot(e1,h);
        if (std::abs(determinant) < 1e-12) return std::numeric_limits<double>::infinity();
        const auto s = origin-triangle.A;
        const double u = Dot(s,h)/determinant;
        if (u < 0 || u > 1) return std::numeric_limits<double>::infinity();
        const auto q = Cross(s,e1);
        const double v = Dot(direction,q)/determinant;
        if (v < 0 || u+v > 1) return std::numeric_limits<double>::infinity();
        const double t = Dot(e2,q)/determinant;
        return t >= settings.TMin && t <= settings.TMax ? t : std::numeric_limits<double>::infinity();
    }
    void Fail(unsigned int frame, unsigned int x, unsigned int y, const std::string &reason)
    {
        std::ostringstream message;
        message << "Shadow frame " << frame << " pixel(" << x << ',' << y << "): " << reason;
        throw std::runtime_error(message.str());
    }
}

void ValidateShadowResults(const ShadowResult *results, const SurfaceInput *surfaces,
    const Scene &scene, const ShadowSettings &settings, unsigned int width, unsigned int height,
    unsigned int frame, std::ostream *samples)
{
    const auto triangles = GetTriangles(scene);
    const Point direction = {settings.LightX,settings.LightY,settings.LightZ};
    size_t background = 0, hits = 0, misses = 0, floorShadow = 0;
    std::vector<size_t> blockers(scene.GetObjects().size(),0);
    std::vector<bool> recorded(scene.GetObjects().size()+2,false);
    double maxDistanceError = 0;
    for (unsigned int y = 0; y < height; ++y)
    {
        for (unsigned int x = 0; x < width; ++x)
        {
            const size_t i = static_cast<size_t>(y)*width+x;
            const auto &s = surfaces[i];
            const auto &r = results[i];
            unsigned int category = 0;
            double nearest = std::numeric_limits<double>::infinity();
            if (s.Position.Valid == 0)
            {
                ++background;
                if (r.Valid || r.Visibility != 0 || r.Blocker || r.HitDistance != 0)
                    Fail(frame,x,y,"Background must be zero and untraced.");
            }
            else
            {
                if (r.Valid != 1 || (r.Visibility != 0 && r.Visibility != 1) || !std::isfinite(r.HitDistance))
                    Fail(frame,x,y,"Invalid shadow output contract.");
                // Round only the origin construction to float, as the GPU does (--fmad=false).
                // Triangle intersection itself is double precision and independent of OptiX.
                const Point origin = {float(s.Position.X+s.NormalX*settings.NormalOffset),
                    float(s.Position.Y+s.NormalY*settings.NormalOffset),float(s.Position.Z+s.NormalZ*settings.NormalOffset)};
                double reportedObjectDistance = std::numeric_limits<double>::infinity();
                for (const auto &triangle : triangles)
                {
                    const double t = Intersect(origin,direction,triangle,settings);
                    nearest = std::min(nearest,t);
                    if (triangle.Object+1 == r.Blocker) reportedObjectDistance = std::min(reportedObjectDistance,t);
                }
                const bool blocked = std::isfinite(nearest);
                if (r.Visibility != (blocked ? 0.0f : 1.0f))
                    Fail(frame,x,y,"OptiX hit/miss differs from CPU brute-force triangles.");
                if (!blocked)
                {
                    ++misses;
                    category = 1;
                    if (r.Blocker != 0 || r.HitDistance != 0) Fail(frame,x,y,"Miss contains a blocker/distance.");
                }
                else
                {
                    ++hits;
                    if (r.Blocker < 1 || r.Blocker > blockers.size() || !std::isfinite(reportedObjectDistance))
                        Fail(frame,x,y,"SBT blocker does not intersect the CPU ray.");
                    const double error = std::abs(double(r.HitDistance)-nearest);
                    const double tolerance = 0.00005 * std::max(1.0,nearest);
                    if (error > tolerance || std::abs(reportedObjectDistance-nearest) > tolerance)
                        Fail(frame,x,y,"Closest distance/SBT object differs from CPU geometry.");
                    maxDistanceError = std::max(maxDistanceError,error);
                    ++blockers[r.Blocker-1];
                    category = r.Blocker+1;
                    if (s.NormalY > 0.99f && std::abs(s.Position.Y+1.0f) < 0.002f && r.Blocker > 1) ++floorShadow;
                }
            }
            if (samples && !recorded[category])
            {
                *samples << std::setprecision(10) << frame << ',' << x << ',' << y << ',' << r.Valid << ','
                    << r.Visibility << ',' << r.Blocker << ',' << r.HitDistance << ','
                    << (std::isfinite(nearest) ? nearest : 0) << ',' << settings.LightX << ',' << settings.LightY << ',' << settings.LightZ << '\n';
                recorded[category] = true;
            }
        }
    }
    if (!background || !hits || !misses || !floorShadow)
        throw std::runtime_error("Shadow fixture must exercise background, hit, miss, and an object casting a shadow on the floor.");
    std::cout << "PASS: Shadow frame " << frame << ": CPU checked ALL " << static_cast<size_t>(width)*height
        << " pixels; background=" << background << ", hit=" << hits << ", miss=" << misses
        << ", floor shadow=" << floorShadow << ", max hit-distance error=" << maxDistanceError << "; blockers=";
    for (const auto count : blockers) std::cout << ' ' << count;
    std::cout << ".\n";
}
