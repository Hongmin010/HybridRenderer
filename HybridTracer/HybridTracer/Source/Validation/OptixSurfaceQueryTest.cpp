#include "Interop/InteropCommon.h"
#include "Renderer/DX12Context.h"
#include "Optix/OptixSceneGAS.h"
#include "Optix/OptixTracePass.h"
#include "Scene/Scene.h"
#include <d3d12sdklayers.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
using namespace HybridInterop;
using namespace DirectX;

namespace
{
    constexpr double Pi = 3.14159265358979323846;
    struct Point
    {
        double X,Y,Z;
        Point operator+(Point b) const { return {X+b.X,Y+b.Y,Z+b.Z}; }
        Point operator-(Point b) const { return {X-b.X,Y-b.Y,Z-b.Z}; }
        Point operator*(double s) const { return {X*s,Y*s,Z*s}; }
    };
    double Dot(Point a, Point b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
    Point Cross(Point a, Point b) { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
    double Length(Point a) { return std::sqrt(Dot(a,a)); }
    Point Normalize(Point a) { return a*(1/Length(a)); }
    double Error(Point a, Point b) { return Length(a-b); }
    bool Finite(Point a) { return std::isfinite(a.X) && std::isfinite(a.Y) && std::isfinite(a.Z); }
    void Require(bool valid, const char *message) { if (!valid) throw std::runtime_error(message); }
    Point TransformNormal(const XMFLOAT3 &p, const XMFLOAT4X4 &m)
    {
        // Independent double cofactor inverse-transpose; no DirectX normal transform helper.
        const double a=m._11,b=m._12,c=m._13,d=m._21,e=m._22,f=m._23,g=m._31,h=m._32,i=m._33;
        const double determinant = a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
        return Normalize({(p.x*(e*i-f*h)+p.y*(c*h-b*i)+p.z*(b*f-c*e))/determinant,
            (p.x*(f*g-d*i)+p.y*(a*i-c*g)+p.z*(c*d-a*f))/determinant,
            (p.x*(d*h-e*g)+p.y*(b*g-a*h)+p.z*(a*e-b*d))/determinant});
    }
    struct Triangle { Point P[3],N[3],Albedo; unsigned int Object,Primitive; };
    std::vector<Triangle> GetTriangles(const Scene &scene)
    {
        std::vector<Triangle> result;
        const auto &objects = scene.GetObjects();
        for (size_t objectIndex = 0; objectIndex < objects.size(); ++objectIndex)
        {
            const auto &object = objects[objectIndex];
            XMFLOAT4X4 m;
            XMStoreFloat4x4(&m,object.ObjectTransform.GetWorldMatrix());
            const auto &vertices = object.MeshData->GetVertices();
            const auto &indices = object.MeshData->GetIndices();
            for (size_t j = 0; j < indices.size(); j += 3)
            {
                Triangle triangle = {};
                triangle.Object = static_cast<unsigned int>(objectIndex);
                triangle.Primitive = static_cast<unsigned int>(j/3);
                triangle.Albedo = {object.MaterialData->Albedo.x,object.MaterialData->Albedo.y,object.MaterialData->Albedo.z};
                for (size_t corner = 0; corner < 3; ++corner)
                {
                    const auto &v = vertices[indices[j+corner]];
                    const auto &p = v.Position;
                    triangle.P[corner] = {double(p.x)*m._11+double(p.y)*m._21+double(p.z)*m._31+m._41,
                        double(p.x)*m._12+double(p.y)*m._22+double(p.z)*m._32+m._42,
                        double(p.x)*m._13+double(p.y)*m._23+double(p.z)*m._33+m._43};
                    triangle.N[corner] = TransformNormal(v.Normal,m);
                }
                result.push_back(triangle);
            }
        }
        return result;
    }
    void AddSmoothFixture(Scene &scene, ID3D12Device *device, bool mirrored)
    {
        std::vector<Vertex> vertices = {{{-1,-0.5f,0},{}},{{1,-0.5f,0},{}},{{0,1,0},{}}};
        const Point normals[] = {{0.25,0,1},{-0.4,0.2,1},{0.1,-0.5,1}};
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            const auto n = Normalize(normals[i]);
            vertices[i].Normal = {float(n.X),float(n.Y),float(n.Z)};
        }
        auto mesh = std::make_unique<Mesh>();
        mesh->Initialize(device,vertices,{0,1,2});
        auto material = std::make_unique<Material>();
        material->Albedo = mirrored ? XMFLOAT3{0.7123f,0.2345f,0.5678f} : XMFLOAT3{0.123f,0.456f,0.789f};
        Transform transform;
        if (mirrored)
        {
            transform.SetPosition(-6,1,-1);
            transform.SetScale(-0.8f,1.7f,0.4f);
            transform.SetRotation(-0.2f,0.4f,-0.3f);
        }
        else
        {
            transform.SetPosition(6,1,1);
            transform.SetScale(1.4f,0.6f,1.2f);
            transform.SetRotation(0.4f,-0.3f,0.2f);
        }
        scene.AddObject(std::move(mesh),std::move(material),transform);
    }
    SurfaceRay MakeRay(Point origin, Point direction, float tmin=0.0001f, float tmax=100)
    {
        direction = Normalize(direction);
        return {float(origin.X),float(origin.Y),float(origin.Z),tmin,
            float(direction.X),float(direction.Y),float(direction.Z),tmax,1,0,0,0};
    }
    std::vector<SurfaceRay> MakeQueries(const std::vector<Triangle> &triangles, unsigned int batch)
    {
        std::vector<SurfaceRay> rays;
        std::mt19937 random(137+batch*7919);
        const auto uniform = [&]() { return double(random() >> 8)/16777216.0; };
        // Synthetic first surfaces on the existing floor -> cosine hemisphere secondary rays.
        // CPU-generated diagnostic inputs; this is not yet a production GI ray generator.
        for (unsigned int i = 0; i < 16384; ++i)
        {
            const Point origin = {-3.7+7.4*uniform(),-0.999, -3.2+7.4*uniform()};
            const double radial = uniform(), phi = 2*Pi*uniform();
            rays.push_back(MakeRay(origin,{std::sqrt(radial)*std::cos(phi),std::sqrt(1-radial),std::sqrt(radial)*std::sin(phi)}));
        }
        const double barycentric[][2] = {{0.2,0.2},{0.6,0.2},{0.1,0.7},{0.333,0.333}};
        const double offset = 0.015+0.005*batch;
        for (const auto &triangle : triangles)
        {
            const auto normal = Normalize(Cross(triangle.P[1]-triangle.P[0],triangle.P[2]-triangle.P[0]));
            for (const auto &uv : barycentric)
            {
                const auto point = triangle.P[0]*(1-uv[0]-uv[1])+triangle.P[1]*uv[0]+triangle.P[2]*uv[1];
                rays.push_back(MakeRay(point+normal*offset,normal*(-1),0.0001f,float(offset*2)));
                rays.push_back(MakeRay(point-normal*offset,normal,0.0001f,float(offset*2)));
            }
            if (triangle.Object >= 4)
            {
                const auto point = (triangle.P[0]+triangle.P[1]+triangle.P[2])*(1.0/3);
                rays.push_back(MakeRay(point+normal*0.03,normal*(-1),0.0001f,0.015f));
                rays.push_back(MakeRay(point+normal*0.03,normal*(-1),0.045f,0.06f));
            }
        }
        rays.push_back(MakeRay({0,10,0},{0,1,0}));
        rays.push_back(MakeRay({100,100,100},{1,1,1}));
        const auto baseline = MakeRay({0,10,0},{0,-1,0});
        auto ray = baseline; ray.Valid=0; rays.push_back(ray);
        ray = baseline; ray.Valid=2; rays.push_back(ray);
        ray = baseline; ray.DirectionX=ray.DirectionY=ray.DirectionZ=0; rays.push_back(ray);
        ray = baseline; ray.DirectionY=-2; rays.push_back(ray);
        ray = baseline; ray.OriginX=std::numeric_limits<float>::quiet_NaN(); rays.push_back(ray);
        ray = baseline; ray.DirectionY=std::numeric_limits<float>::infinity(); rays.push_back(ray);
        ray = baseline; ray.TMax=std::numeric_limits<float>::infinity(); rays.push_back(ray);
        ray = baseline; ray.TMin=-1; rays.push_back(ray);
        ray = baseline; ray.TMax=ray.TMin; rays.push_back(ray);
        return rays;
    }
    struct Intersection { double T=std::numeric_limits<double>::infinity(),U=0,V=0; };
    Intersection Intersect(const SurfaceRay &ray, const Triangle &triangle, double margin=0)
    {
        const Point origin={ray.OriginX,ray.OriginY,ray.OriginZ}, direction={ray.DirectionX,ray.DirectionY,ray.DirectionZ};
        const auto e1=triangle.P[1]-triangle.P[0], e2=triangle.P[2]-triangle.P[0];
        const auto h=Cross(direction,e2);
        const double determinant=Dot(e1,h);
        if (std::abs(determinant)<1e-12) return {};
        const auto s=origin-triangle.P[0], q=Cross(s,e1);
        const double u=Dot(s,h)/determinant, v=Dot(direction,q)/determinant, t=Dot(e2,q)/determinant;
        if (u < -margin || v < -margin || u+v > 1+margin || t < ray.TMin || t > ray.TMax) return {};
        return {t,u,v};
    }
    bool Usable(const SurfaceRay &ray)
    {
        const Point o={ray.OriginX,ray.OriginY,ray.OriginZ},d={ray.DirectionX,ray.DirectionY,ray.DirectionZ};
        return ray.Valid==1 && Finite(o) && Finite(d) && std::abs(Dot(d,d)-1)<=0.0001 &&
            std::isfinite(ray.TMin) && std::isfinite(ray.TMax) && ray.TMin>=0 && ray.TMax>ray.TMin;
    }
    void Fail(unsigned int batch, size_t ray, const char *reason)
    {
        throw std::runtime_error("Surface batch "+std::to_string(batch)+", ray "+std::to_string(ray)+": "+reason);
    }
    void Validate(const std::vector<SurfaceRay> &rays, const std::vector<SurfaceHit> &hits,
        const std::vector<Triangle> &triangles, unsigned int batch, std::ostream *csv)
    {
        size_t hitCount=0,missCount=0,invalidCount=0,frontCount=0,backCount=0,smoothCount=0,mirroredCount=0;
        double maxT=0,maxPosition=0,maxNg=0,maxNs=0,maxBary=0,maxAlbedo=0;
        std::vector<unsigned int> covered(triangles.size(),0),recorded(6,0);
        bool recordedMiss=false;
        for (size_t index=0;index<rays.size();++index)
        {
            const auto &ray=rays[index]; const auto &hit=hits[index];
            const bool usable=Usable(ray);
            Intersection nearest;
            size_t reported=triangles.size();
            if (usable) for (size_t j=0;j<triangles.size();++j)
            {
                const auto candidate=Intersect(ray,triangles[j]);
                if (candidate.T<nearest.T) nearest=candidate;
                if (triangles[j].Object==hit.ObjectIndex && triangles[j].Primitive==hit.PrimitiveIndex) reported=j;
            }
            if (!std::isfinite(nearest.T))
            {
                const SurfaceHit zero={};
                if (std::memcmp(&hit,&zero,sizeof(hit))!=0) Fail(batch,index,"Miss/invalid ray must overwrite the full record with zeros.");
                if (usable) ++missCount; else ++invalidCount;
                if (csv && usable && !recordedMiss)
                {
                    *csv << batch << ',' << index;
                    for (unsigned int field=0;field<19;++field) *csv << ",0";
                    *csv << '\n'; recordedMiss=true;
                }
                continue;
            }
            if (hit.Valid!=1 || reported==triangles.size()) Fail(batch,index,"GPU hit/miss or SBT object/primitive ID differs from Scene.");
            const auto &triangle=triangles[reported];
            const auto expected=Intersect(ray,triangle,0.000001);
            if (!std::isfinite(expected.T) || std::abs(expected.T-nearest.T)>0.00002*std::max(1.0,nearest.T))
                Fail(batch,index,"Reported primitive is not a CPU nearest intersection.");
            const Point ng=Normalize(Cross(triangle.P[1]-triangle.P[0],triangle.P[2]-triangle.P[0]));
            const Point ns=Normalize(triangle.N[0]*(1-expected.U-expected.V)+triangle.N[1]*expected.U+triangle.N[2]*expected.V);
            const Point p=triangle.P[0]*(1-expected.U-expected.V)+triangle.P[1]*expected.U+triangle.P[2]*expected.V;
            const Point gpuP={hit.PositionX,hit.PositionY,hit.PositionZ},gpuNg={hit.GeometricNormalX,hit.GeometricNormalY,hit.GeometricNormalZ};
            const Point gpuNs={hit.ShadingNormalX,hit.ShadingNormalY,hit.ShadingNormalZ},gpuAlbedo={hit.AlbedoR,hit.AlbedoG,hit.AlbedoB};
            const Point direction={ray.DirectionX,ray.DirectionY,ray.DirectionZ},origin={ray.OriginX,ray.OriginY,ray.OriginZ};
            const double tError=std::abs(double(hit.Distance)-expected.T),pError=Error(gpuP,p),ngError=Error(gpuNg,ng),nsError=Error(gpuNs,ns);
            const double baryError=std::max(std::abs(hit.BarycentricU-expected.U),std::abs(hit.BarycentricV-expected.V));
            const double albedoError=Error(gpuAlbedo,triangle.Albedo);
            if (!Finite(gpuP) || !Finite(gpuNg) || !Finite(gpuNs) || !Finite(gpuAlbedo) || !std::isfinite(hit.Distance) ||
                !std::isfinite(hit.BarycentricU) || !std::isfinite(hit.BarycentricV)) Fail(batch,index,"Non-finite surface result.");
            if (tError>0.00002*std::max(1.0,expected.T) || pError>0.00003 || Error(gpuP,origin+direction*hit.Distance)>0.00003)
                Fail(batch,index,"Position/distance differs from CPU ray and original triangle.");
            if (ngError>0.00002 || nsError>0.00002 || std::abs(Length(gpuNg)-1)>0.000003 || std::abs(Length(gpuNs)-1)>0.000003)
                Fail(batch,index,"Geometric/interpolated inverse-transpose normal mismatch.");
            if (baryError>0.00002 || hit.BarycentricU < -0.000001f || hit.BarycentricV < -0.000001f ||
                hit.BarycentricU+hit.BarycentricV>1.000001f) Fail(batch,index,"Barycentric attribute mismatch.");
            if (albedoError>0.0000001) Fail(batch,index,"SBT full-precision material color mismatch.");
            const unsigned int expectedFront=Dot(ng,direction)<0 ? 1u:0u;
            if (hit.FrontFace!=expectedFront || hit.TwoSided!=0) Fail(batch,index,"Native triangle front/back flag mismatch.");
            ++hitCount; ++covered[reported]; if (hit.FrontFace) ++frontCount; else ++backCount;
            if (triangle.Object==4 && Dot(ng,ns)<0.999) ++smoothCount;
            if (triangle.Object==5 && Dot(ng,ns)<0) ++mirroredCount;
            maxT=std::max(maxT,tError); maxPosition=std::max(maxPosition,pError); maxNg=std::max(maxNg,ngError);
            maxNs=std::max(maxNs,nsError); maxBary=std::max(maxBary,baryError); maxAlbedo=std::max(maxAlbedo,albedoError);
            if (csv && recorded[triangle.Object]++<2)
                *csv << std::setprecision(10) << batch << ',' << index << ',' << hit.Valid << ',' << hit.ObjectIndex << ',' << hit.PrimitiveIndex << ','
                    << hit.FrontFace << ',' << hit.Distance << ',' << hit.PositionX << ',' << hit.PositionY << ',' << hit.PositionZ << ','
                    << hit.GeometricNormalX << ',' << hit.GeometricNormalY << ',' << hit.GeometricNormalZ << ','
                    << hit.ShadingNormalX << ',' << hit.ShadingNormalY << ',' << hit.ShadingNormalZ << ','
                    << hit.AlbedoR << ',' << hit.AlbedoG << ',' << hit.AlbedoB << ',' << hit.BarycentricU << ',' << hit.BarycentricV << '\n';
        }
        const size_t coverage=static_cast<size_t>(std::count_if(covered.begin(),covered.end(),[](unsigned int n){return n!=0;}));
        Require(coverage==triangles.size(),"Directed probes did not hit every Scene primitive.");
        Require(frontCount && backCount && missCount && invalidCount && smoothCount && mirroredCount,
            "Fixture must exercise front/back, miss, invalid, smooth shading and mirrored normals.");
        std::cout << "PASS: Surface batch " << batch << ": rays=" << rays.size() << ", hits=" << hitCount << ", miss=" << missCount
            << ", invalid=" << invalidCount << ", primitives=" << coverage << '/' << triangles.size() << ", front/back=" << frontCount << '/' << backCount
            << "; max t/position/Ng/Ns/bary/albedo error=" << maxT << '/' << maxPosition << '/' << maxNg << '/' << maxNs << '/' << maxBary << '/' << maxAlbedo << ".\n";
    }
    struct CudaResources
    {
        CUcontext Context=nullptr;
        CUstream Stream=nullptr;
        CUdeviceptr Rays=0,Hits=0,Surfaces=0;
        void *Host=nullptr;
        ~CudaResources()
        {
            if (Context) cuCtxSetCurrent(Context);
            if (Stream) cuStreamSynchronize(Stream);
            if (Rays) cuMemFree(Rays); if (Hits) cuMemFree(Hits); if (Surfaces) cuMemFree(Surfaces);
            if (Host) cuMemFreeHost(Host);
            if (Stream) cuStreamDestroy(Stream);
            if (Context) cuCtxDestroy(Context);
        }
    };
    void CheckMessages(ID3D12Device *device)
    {
        ComPtr<ID3D12InfoQueue> messages;
        CheckHR(device->QueryInterface(IID_PPV_ARGS(&messages)),"QueryInterface(info queue)");
        for (UINT64 i=0;i<messages->GetNumStoredMessagesAllowedByRetrievalFilter();++i)
        {
            SIZE_T bytes=0; CheckHR(messages->GetMessage(i,nullptr,&bytes),"GetMessage(size)");
            std::vector<unsigned char> data(bytes);
            auto *message=reinterpret_cast<D3D12_MESSAGE *>(data.data());
            CheckHR(messages->GetMessage(i,message,&bytes),"GetMessage");
            if (message->Severity<=D3D12_MESSAGE_SEVERITY_WARNING) throw std::runtime_error(message->pDescription);
        }
    }
    void Run(bool samples)
    {
        DX12Context dx; dx.InitializeHeadless();
        Scene scene; scene.Initialize(dx.GetDevice());
        AddSmoothFixture(scene,dx.GetDevice(),false); AddSmoothFixture(scene,dx.GetDevice(),true);
        Require(scene.GetObjects().size()==6,"Surface-query fixture expects four default objects and two diagnostic triangles.");
        const auto triangles=GetTriangles(scene);
        const size_t count=MakeQueries(triangles,0).size();
        CudaResources cuda;
        CheckCuda(cuCtxCreate(&cuda.Context,CU_CTX_SCHED_AUTO,FindCudaDevice(dx.GetDevice())),"cuCtxCreate(surface query)");
        CheckCuda(cuStreamCreate(&cuda.Stream,CU_STREAM_NON_BLOCKING),"cuStreamCreate(surface query)");
        CheckCuda(cuMemAlloc(&cuda.Rays,count*sizeof(SurfaceRay)),"cuMemAlloc(query rays)");
        CheckCuda(cuMemAlloc(&cuda.Hits,count*sizeof(SurfaceHit)),"cuMemAlloc(query hits)");
        CheckCuda(cuMemAlloc(&cuda.Surfaces,count*sizeof(SurfaceInput)),"cuMemAlloc(shadow surfaces)");
        CheckCuda(cuMemAllocHost(&cuda.Host,count*std::max(sizeof(SurfaceRay),sizeof(SurfaceInput))),"cuMemAllocHost(query input)");
        OptixSceneGAS gas; gas.Initialize(cuda.Context,cuda.Stream); gas.Build(scene); gas.ValidateAgainstScene(scene);
        OptixTracePass trace; trace.Initialize(cuda.Context,cuda.Stream,gas,static_cast<unsigned int>(count),1);
        std::ofstream csv;
        if (samples)
        {
            csv.open(std::filesystem::current_path()/L"OptixSurfaceHitSamples.csv");
            Require(bool(csv),"Cannot create surface sample report.");
            csv << "batch,ray,valid,object,primitive,front,t,position_x,position_y,position_z,ng_x,ng_y,ng_z,ns_x,ns_y,ns_z,albedo_r,albedo_g,albedo_b,bary_u,bary_v\n";
        }
        std::vector<SurfaceHit> hits(count);
        for (unsigned int batch=0;batch<4;++batch)
        {
            const auto rays=MakeQueries(triangles,batch);
            Require(rays.size()==count,"Query batch dimensions changed.");
            std::memcpy(cuda.Host,rays.data(),count*sizeof(SurfaceRay));
            CheckCuda(cuMemcpyHtoDAsync(cuda.Rays,cuda.Host,count*sizeof(SurfaceRay),cuda.Stream),"cuMemcpyHtoDAsync(query rays)");
            // A poison fill makes zeroed misses/invalid records observable across every batch.
            CheckCuda(cuMemsetD8Async(cuda.Hits,0xcd,count*sizeof(SurfaceHit),cuda.Stream),"cuMemsetD8Async(query output poison)");
            trace.LaunchSurfaceQueries(cuda.Rays,cuda.Hits);
            CheckCuda(cuStreamSynchronize(cuda.Stream),"cuStreamSynchronize(surface query)");
            CheckCuda(cuMemcpyDtoH(hits.data(),cuda.Hits,count*sizeof(SurfaceHit)),"cuMemcpyDtoH(query hits)");
            Validate(rays,hits,triangles,batch,samples ? &csv : nullptr);
            // Exercise query -> shadow -> query parameter reuse on the SAME production pass.
            auto *surfaces=static_cast<SurfaceInput *>(cuda.Host);
            for (size_t i=0;i<count;++i) surfaces[i]={{0,10,0,1},0,1,0,0,0.5f,0.5f,0.5f,1};
            CheckCuda(cuMemcpyHtoDAsync(cuda.Surfaces,cuda.Host,count*sizeof(SurfaceInput),cuda.Stream),"cuMemcpyHtoDAsync(shadow surfaces)");
            trace.Launch(cuda.Surfaces,{0,1,0,0.001f,0.0001f,100});
            CheckCuda(cuStreamSynchronize(cuda.Stream),"cuStreamSynchronize(shadow mode)");
            std::vector<ShadowResult> shadows(count);
            CheckCuda(cuMemcpyDtoH(shadows.data(),trace.GetDeviceOutput(),shadows.size()*sizeof(ShadowResult)),"cuMemcpyDtoH(shadow mode)");
            for (const auto &shadow:shadows) Require(shadow.Valid==1 && shadow.Visibility==1 && shadow.Blocker==0 && shadow.HitDistance==0,
                "Switching query/shadow mode retained stale launch parameters.");
        }
        if (samples) { csv.flush(); Require(bool(csv),"Cannot flush surface samples."); }
        gas.ValidateAgainstScene(scene); CheckMessages(dx.GetDevice());
        std::cout << "PASS: GPU closest-hit position, barycentrics, geometric/shading normals and full-precision material; every primitive and mode reuse verified.\n";
    }
}

int main(int argc,char **argv)
{
    try
    {
        const bool samples=argc==2 && std::string(argv[1])=="--samples";
        Require(argc==1 || samples,"Usage: HybridTracerSurfaceQueryTest [--samples]");
        ComPtr<ID3D12Debug> debug;
        CheckHR(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),"D3D12GetDebugInterface"); debug->EnableDebugLayer();
        Run(samples); return 0;
    }
    catch (const std::exception &error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}
