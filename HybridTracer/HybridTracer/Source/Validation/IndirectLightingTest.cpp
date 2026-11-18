#include "Interop/InteropCommon.h"
#include "Renderer/DX12Context.h"
#include "Optix/OptixSceneGAS.h"
#include "Optix/OptixTracePass.h"
#include "Scene/Scene.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <vector>
using namespace HybridInterop;
using namespace DirectX;

namespace
{
    constexpr size_t Count=16384;
    constexpr double Pi=3.14159265358979323846;
    void Require(bool condition,const char *message) { if (!condition) throw std::runtime_error(message); }
    struct V
    {
        double X,Y,Z;
        V operator+(V b) const { return {X+b.X,Y+b.Y,Z+b.Z}; }
        V operator-(V b) const { return {X-b.X,Y-b.Y,Z-b.Z}; }
        V operator*(double s) const { return {X*s,Y*s,Z*s}; }
    };
    double Dot(V a,V b) { return a.X*b.X+a.Y*b.Y+a.Z*b.Z; }
    V Cross(V a,V b) { return {a.Y*b.Z-a.Z*b.Y,a.Z*b.X-a.X*b.Z,a.X*b.Y-a.Y*b.X}; }
    V Unit(V a) { return a*(1/std::sqrt(Dot(a,a))); }
    double Error(V a,V b) { const auto d=a-b; return std::sqrt(Dot(d,d)); }
    struct Triangle { V A,B,C,N,Rho; unsigned int Object; };
    std::vector<Triangle> Triangles(const Scene &scene)
    {
        std::vector<Triangle> result;
        for (size_t object=0;object<scene.GetObjects().size();++object)
        {
            const auto &o=scene.GetObjects()[object];
            XMFLOAT4X4 m; XMStoreFloat4x4(&m,o.ObjectTransform.GetWorldMatrix());
            const auto &vertices=o.MeshData->GetVertices(); const auto &indices=o.MeshData->GetIndices();
            const auto point=[&](uint16_t index)
            {
                const auto &v=vertices[index].Position;
                return V{v.x*double(m._11)+v.y*double(m._21)+v.z*double(m._31)+m._41,
                    v.x*double(m._12)+v.y*double(m._22)+v.z*double(m._32)+m._42,
                    v.x*double(m._13)+v.y*double(m._23)+v.z*double(m._33)+m._43};
            };
            // This fixture uses the default scene's flat, positively scaled faces.
            for (size_t j=0;j<indices.size();j+=3)
            {
                const V a=point(indices[j]),b=point(indices[j+1]),c=point(indices[j+2]);
                const auto &rho=o.MaterialData->Albedo;
                result.push_back({a,b,c,Unit(Cross(b-a,c-a)),{rho.x,rho.y,rho.z},static_cast<unsigned int>(object)});
            }
        }
        return result;
    }
    double Intersect(V origin,V direction,const Triangle &triangle,float tMin,float tMax)
    {
        const V e1=triangle.B-triangle.A,e2=triangle.C-triangle.A,h=Cross(direction,e2);
        const double determinant=Dot(e1,h);
        if (std::abs(determinant)<1e-12) return INFINITY;
        const V s=origin-triangle.A,q=Cross(s,e1);
        const double u=Dot(s,h)/determinant,v=Dot(direction,q)/determinant,t=Dot(e2,q)/determinant;
        return u>=0 && v>=0 && u+v<=1 && t>=tMin && t<=tMax ? t : INFINITY;
    }
    const Triangle *Trace(const std::vector<Triangle> &triangles,V origin,V direction,float tMin,float tMax,double &nearest)
    {
        nearest=INFINITY; const Triangle *hit=nullptr;
        for (const auto &triangle:triangles)
        {
            const double t=Intersect(origin,direction,triangle,tMin,tMax);
            if (t<nearest) { nearest=t; hit=&triangle; }
        }
        return hit;
    }
    struct Cuda
    {
        CUcontext Context=nullptr; CUstream Stream=nullptr; CUdeviceptr Input=0; SurfaceInput *Host=nullptr;
        ~Cuda()
        {
            if (Context) cuCtxSetCurrent(Context);
            if (Stream) cuStreamSynchronize(Stream);
            if (Input) cuMemFree(Input); if (Host) cuMemFreeHost(Host);
            if (Stream) cuStreamDestroy(Stream); if (Context) cuCtxDestroy(Context);
        }
        void Initialize(ID3D12Device *device)
        {
            CheckCuda(cuCtxCreate(&Context,CU_CTX_SCHED_AUTO,FindCudaDevice(device)),"cuCtxCreate(indirect test)");
            CheckCuda(cuStreamCreate(&Stream,CU_STREAM_NON_BLOCKING),"cuStreamCreate(indirect test)");
            CheckCuda(cuMemAlloc(&Input,Count*sizeof(SurfaceInput)),"cuMemAlloc(indirect input)");
            CheckCuda(cuMemAllocHost(reinterpret_cast<void **>(&Host),Count*sizeof(SurfaceInput)),"cuMemAllocHost(indirect input)");
        }
    };
    void Upload(Cuda &cuda)
    {
        CheckCuda(cuMemcpyHtoDAsync(cuda.Input,cuda.Host,Count*sizeof(SurfaceInput),cuda.Stream),"cuMemcpyHtoDAsync(indirect test)");
    }
    std::vector<IndirectLighting> Read(Cuda &cuda,const OptixTracePass &pass)
    {
        CheckCuda(cuStreamSynchronize(cuda.Stream),"cuStreamSynchronize(indirect test)");
        std::vector<IndirectLighting> values(Count*pass.GetLightingPlaneCount());
        CheckCuda(cuMemcpyDtoH(values.data(),pass.GetDeviceIndirect(),pass.GetIndirectBytes()),"cuMemcpyDtoH(indirect test)");
        return values;
    }
    void Validate(const std::vector<Triangle> &triangles,Cuda &cuda,const OptixTracePass &pass,
        const std::vector<IndirectLighting> &values,const ShadowSettings &shadow,const IndirectSettings &settings,
        bool zero,std::ostream *csv)
    {
        std::vector<IndirectSampleDebug> debug(Count);
        CheckCuda(cuMemcpyDtoH(debug.data(),pass.GetDeviceIndirectDebug(),Count*sizeof(IndirectSampleDebug)),"cuMemcpyDtoH(indirect diagnostics)");
        size_t misses=0,hits=0,lit=0,blocked=0,backfaces=0;
        double maxError=0; std::array<bool,8> recorded={};
        const V light={shadow.LightX,shadow.LightY,shadow.LightZ};
        for (size_t i=0;i<Count;++i)
        {
            const auto &a=cuda.Host[i]; const auto &d=debug[i]; const auto &r=values[i];
            Require(r.Valid==(a.Position.Valid==1 ? 1u:0u),"Indirect validity differs from primary surface.");
            V expected={}; unsigned int category=0;
            if (d.SampleValid)
            {
                const V direction={d.Ray.DirectionX,d.Ray.DirectionY,d.Ray.DirectionZ};
                const V origin={d.Ray.OriginX,d.Ray.OriginY,d.Ray.OriginZ};
                const V normal={a.NormalX,a.NormalY,a.NormalZ};
                Require(std::abs(Dot(direction,direction)-1)<1e-5 && Dot(normal,direction)>0,"Secondary ray not in unit cosine hemisphere.");
                Require(std::abs(d.Pdf-Dot(normal,direction)/Pi)<1e-6,"Secondary PDF differs from cosine/pi.");
                Require(d.WeightR==a.AlbedoR && d.WeightG==a.AlbedoG && d.WeightB==a.AlbedoB,"Primary BSDF weight must be reflectance.");
                Require(Error(origin,{float(a.Position.X+a.NormalX*shadow.NormalOffset),
                    float(a.Position.Y+a.NormalY*shadow.NormalOffset),float(a.Position.Z+a.NormalZ*shadow.NormalOffset)})<1e-7,
                    "Secondary origin differs from normal offset.");
                double t; const auto *b=Trace(triangles,origin,direction,shadow.TMin,shadow.TMax,t);
                Require(d.Hit.Valid==(b ? 1u:0u),"GPU secondary hit/miss differs from CPU triangles.");
                if (!b) { ++misses; category=1; Require(d.Visibility==0,"Miss must have zero indirect visibility."); }
                else
                {
                    ++hits; category=2;
                    const V p=origin+direction*t;
                    Require(Error(p,{d.Hit.PositionX,d.Hit.PositionY,d.Hit.PositionZ})<0.00005,"Secondary position differs from CPU trace.");
                    Require(d.Hit.ObjectIndex==b->Object,"Secondary material object differs from CPU trace.");
                    const bool front=Dot(b->N,direction)<0;
                    Require(d.Hit.FrontFace==(front ? 1u:0u),"Secondary front face differs from triangle winding.");
                    const double cosine=std::max(0.0,Dot(b->N,light));
                    if (!front) { ++backfaces; category=3; }
                    else if (cosine>0)
                    {
                        // Independent secondary shadow from original CPU scene and CPU hit position.
                        const V o={float(float(p.X)+float(b->N.X)*shadow.NormalOffset),
                            float(float(p.Y)+float(b->N.Y)*shadow.NormalOffset),float(float(p.Z)+float(b->N.Z)*shadow.NormalOffset)};
                        double shadowT; const bool visible=Trace(triangles,o,light,shadow.TMin,shadow.TMax,shadowT)==nullptr;
                        Require(d.Visibility==(visible ? 1.0f:0.0f),"Secondary light visibility differs from CPU triangles.");
                        if (visible)
                        {
                            ++lit; category=4+b->Object;
                            expected={a.AlbedoR*b->Rho.X*settings.IrradianceR*cosine/Pi,
                                a.AlbedoG*b->Rho.Y*settings.IrradianceG*cosine/Pi,a.AlbedoB*b->Rho.Z*settings.IrradianceB*cosine/Pi};
                        }
                        else { ++blocked; category=2; }
                    }
                }
            }
            if (zero) expected={};
            const double error=Error(expected,{r.R,r.G,r.B}); maxError=std::max(maxError,error);
            if (!std::isfinite(error) || error>0.00001)
                throw std::runtime_error("Indirect radiance differs from rho_A*rho_B/pi*E*cos_B*V_B at pixel "+std::to_string(i));
            if (csv && !recorded[category])
            {
                *csv<<settings.Seed<<','<<i<<','<<d.Hit.Valid<<','<<d.Hit.ObjectIndex<<','<<d.Visibility<<','
                    <<r.R<<','<<r.G<<','<<r.B<<','<<expected.X<<','<<expected.Y<<','<<expected.Z<<'\n';
                recorded[category]=true;
            }
        }

        std::cout<<"PASS: CPU one-bounce reference: "<<Count<<" pixels, seed="<<settings.Seed
            <<", misses="<<misses<<", hits="<<hits<<", lit="<<lit<<", shadowed B="<<blocked
            <<", backfaces="<<backfaces<<", max linear RGB error="<<maxError<<".\n";
        if (!zero) Require(misses && hits && lit && blocked,"Indirect fixture must cover miss, hit, lit B and shadowed B.");
    }
    void DefaultScene(ID3D12Device *device,Cuda &cuda,std::ostream *csv)
    {
        Scene scene; scene.Initialize(device); const auto triangles=Triangles(scene);
        OptixSceneGAS gas; gas.Initialize(cuda.Context,cuda.Stream); gas.Build(scene);
        OptixTracePass pass; pass.Initialize(cuda.Context,cuda.Stream,gas,static_cast<unsigned int>(Count),1,true,true);
        const V l=Unit({-0.5,1,-0.5});
        const ShadowSettings shadow={float(l.X),float(l.Y),float(l.Z),0.001f,0.0001f,100};
        std::vector<IndirectLighting> first;
        for (unsigned int batch=0;batch<7;++batch)
        {
            for (size_t i=0;i<Count;++i)
                cuda.Host[i]={{float((i%128)/127.0*7.6-3.8),i%2 ? 4.0f:-1.0f,float((i/128)/127.0*7.6-3.2),1},
                    0,i%2 ? -1.0f:1.0f,0,0,0.6f,0.4f,0.2f,1};
            cuda.Host[0]={}; // Background.
            IndirectSettings settings={0,3,-7,batch==2 ? 29u:17u,float(Pi),float(1.2*Pi),float(0.7*Pi),1,1};
            if (batch==3) settings.IrradianceR=settings.IrradianceG=settings.IrradianceB=0;
            if (batch==4) for (size_t i=0;i<Count;++i) cuda.Host[i].AlbedoR=cuda.Host[i].AlbedoG=cuda.Host[i].AlbedoB=0;
            if (batch==5) for (size_t i=0;i<Count;++i) cuda.Host[i].NormalY *= -1; // Faces away from camera.
            if (batch==6) for (size_t i=0;i<Count;++i) cuda.Host[i].Position.Valid=0;
            Upload(cuda); pass.LaunchIndirect(cuda.Input,shadow,settings); const auto result=Read(cuda,pass);
            Validate(triangles,cuda,pass,result,shadow,settings,batch>=3,csv);
            if (batch==0) first=result;
            if (batch==1) Require(std::memcmp(first.data(),result.data(),Count*sizeof(IndirectLighting))==0,"Same seed did not reproduce exact radiance.");
            if (batch==2)
            {
                size_t changed=0;
                for (size_t i=0;i<Count;++i) if (std::memcmp(&first[i],&result[i],sizeof(IndirectLighting))) ++changed;
                Require(changed>1000,"Changing seed did not change stochastic indirect samples.");
                std::cout<<"PASS: changed seed modifies "<<changed<<" radiance records; same seed is bit exact.\n";
            }
        }
        gas.ValidateAgainstScene(scene);
    }
    void AnalyticPlane(ID3D12Device *device,Cuda &cuda)
    {
        Scene scene;
        // Gap after offset is 0.001; 24-bit sampling bounds all ray footprints inside this plane.
        auto mesh=std::make_unique<Mesh>();
        mesh->Initialize(device,{{{-20,0.002f,-20},{0,-1,0}},{{20,0.002f,-20},{0,-1,0}},
            {{20,0.002f,20},{0,-1,0}},{{-20,0.002f,20},{0,-1,0}}},{0,1,2,0,2,3});
        auto material=std::make_unique<Material>(); material->Albedo={0.25f,0.5f,0.75f};
        scene.AddObject(std::move(mesh),std::move(material),Transform{});
        OptixSceneGAS gas; gas.Initialize(cuda.Context,cuda.Stream); gas.Build(scene);
        OptixTracePass pass; pass.Initialize(cuda.Context,cuda.Stream,gas,static_cast<unsigned int>(Count),1,true);
        const ShadowSettings shadow={0,-1,0,0.001f,0.0001f,100000};
        for (size_t i=0;i<Count;++i) cuda.Host[i]={{0,0,0,1},0,1,0,0,0.8f,0.6f,0.4f,1};
        Upload(cuda);
        for (unsigned int spp:{1u,4u,64u,128u})
        {
            const IndirectSettings settings={0,0.0015f,0,71,float(Pi),float(2*Pi),float(0.5*Pi),spp,1};
            pass.LaunchIndirect(cuda.Input,shadow,settings); const auto result=Read(cuda,pass);
            double error=0;
            for (const auto &r:result)
            {
                Require(r.Valid==1,"Analytic plane lost valid surface.");
                error=std::max(error,Error({r.R,r.G,r.B},{0.2,0.6,0.15}));
            }
            Require(error<0.000003,"Analytic plane energy/SPP average differs from rho_A*rho_B*E/pi.");
            std::cout<<"PASS: constant-radiance plane, "<<spp<<" spp, "<<Count<<" pixels; expected RGB=(0.2,0.6,0.15), max error="<<error<<".\n";
        }
        // Host-side validation must reject unsupported lighting, before optixLaunch.
        auto invalid=IndirectSettings{0,1,0,1,1,1,1,0,1};
        bool rejected=false; try { pass.LaunchIndirect(cuda.Input,shadow,invalid); } catch (const std::exception &) { rejected=true; }
        Require(rejected,"Zero samples must be rejected.");
        invalid.SamplesPerPixel=1; invalid.IrradianceG=-1; rejected=false;
        try { pass.LaunchIndirect(cuda.Input,shadow,invalid); } catch (const std::exception &) { rejected=true; }
        Require(rejected,"Negative irradiance must be rejected.");
        invalid.IrradianceG=1; invalid.MaxBounces=MaxIndirectBounces+1; rejected=false;
        try { pass.LaunchIndirect(cuda.Input,shadow,invalid); } catch (const std::exception &) { rejected=true; }
        Require(rejected,"Unsupported bounce limit must be rejected.");
        gas.ValidateAgainstScene(scene);
    }
    void AnalyticTwoBounces(ID3D12Device *device,Cuda &cuda)
    {
        Scene scene;
        // Light points upward at a grazing angle. The first hit faces down.
        // From the floor, the shadow ray clears the finite ceiling on its +X side.
        constexpr float yB=0.002f,yC=-0.002f,lightY=0.00004f;
        auto ceiling=std::make_unique<Mesh>();
        ceiling->Initialize(device,{{{-20,yB,-20},{0,-1,0}},{{20,yB,-20},{0,-1,0}},
            {{20,yB,20},{0,-1,0}},{{-20,yB,20},{0,-1,0}}},{0,1,2,0,2,3});
        auto ceilingMaterial=std::make_unique<Material>();
        ceilingMaterial->Albedo={0.25f,0.5f,0.75f};
        scene.AddObject(std::move(ceiling),std::move(ceilingMaterial),Transform{});
        auto floor=std::make_unique<Mesh>();
        floor->Initialize(device,{{{-40,yC,-40},{0,1,0}},{{40,yC,-40},{0,1,0}},
            {{40,yC,40},{0,1,0}},{{-40,yC,40},{0,1,0}}},{0,2,1,0,3,2});
        auto floorMaterial=std::make_unique<Material>();
        floorMaterial->Albedo={0.5f,0.25f,0.5f};
        scene.AddObject(std::move(floor),std::move(floorMaterial),Transform{});
        OptixSceneGAS gas; gas.Initialize(cuda.Context,cuda.Stream); gas.Build(scene);
        OptixTracePass pass; pass.Initialize(cuda.Context,cuda.Stream,gas,static_cast<unsigned int>(Count),1,true);
        const ShadowSettings shadow={1,lightY,0,0.001f,0.0001f,100000};
        for (size_t i=0;i<Count;++i) cuda.Host[i]={{0,0,0,1},0,1,0,0,0.8f,0.6f,0.4f,1};
        cuda.Host[0]={}; // Background must remain invalid even when zero bounces are requested.
        Upload(cuda);
        const V expected={0.8*0.25*0.5*lightY,0.6*0.5*0.25*2*lightY,0.4*0.75*0.5*0.5*lightY};
        for (unsigned int bounces:{0u,1u,2u})
        {
            const IndirectSettings settings={0,0.0015f,0,71,float(Pi),float(2*Pi),float(0.5*Pi),4,bounces};
            pass.LaunchIndirect(cuda.Input,shadow,settings);
            const auto result=Read(cuda,pass);
            double maxError=0;
            for (size_t i=0;i<result.size();++i)
            {
                const auto &pixel=result[i];
                Require(pixel.Valid==(i==0 ? 0u:1u),"Changing bounce limit changed surface validity.");
                const V reference=bounces==2 && i!=0 ? expected:V{};
                const double error=Error({pixel.R,pixel.G,pixel.B},reference);
                Require(std::isfinite(error) && error<1e-10,"Two-bounce Lambert throughput differs from the analytic result.");
                maxError=std::max(maxError,error);
            }
            std::cout<<"PASS: analytic ceiling/floor, "<<bounces<<" indirect bounces; max linear RGB error="<<maxError<<".\n";
        }
    }

    double PanelIntegral(V position, unsigned int steps)
    {
        // Unit Lambert albedo looking perpendicularly at a 2 x 2 panel of radiance 3.
        const double heightSquared = (position.Y-2)*(position.Y-2);
        const double step = 2.0/steps;
        double sum = 0;
        for (unsigned int u=0; u<steps; ++u)
        {
            for (unsigned int v=0; v<steps; ++v)
            {
                const double dx=(u+0.5)*step-1-position.X, dz=(v+0.5)*step-1-position.Z;
                const double r2=dx*dx+dz*dz+heightSquared;
                sum += heightSquared/(r2*r2);
            }
        }
        return 3*sum*step*step/Pi;
    }

    void AnalyticAdditionalLights(ID3D12Device *device,Cuda &cuda)
    {
        Scene scene;
        const auto addCeiling = [&](float xmin,float xmax,float y,float halfZ)
        {
            auto mesh=std::make_unique<Mesh>();
            mesh->Initialize(device,{{{xmin,y,-halfZ},{0,-1,0}},{{xmax,y,-halfZ},{0,-1,0}},
                {{xmax,y,halfZ},{0,-1,0}},{{xmin,y,halfZ},{0,-1,0}}},{0,1,2,0,2,3});
            scene.AddObject(std::move(mesh),std::make_unique<Material>(),Transform{});
        };
        addCeiling(-20,20,4,20);       // Beyond the finite emitters.
        addCeiling(1.25f,2.75f,1,1);  // Occludes the right probe and part of the penumbra probe.
        OptixSceneGAS gas; gas.Initialize(cuda.Context,cuda.Stream); gas.Build(scene);
        OptixTracePass pass;
        pass.Initialize(cuda.Context,cuda.Stream,gas,static_cast<unsigned int>(Count),1,true,true,{
            PointLight{{0,2,0},{1,0,0},16},
            RectangleLight{{0,2,0},{1,0,0},{0,0,1},{0,1,0},3},
            RectangleLight{{0,2,0},{1,0,0},{0,0,-1},{0,0,1},3},
            DirectionalLight{{0,1,0},{0,0,1},float(Pi)}});
        const ShadowSettings shadow={0,1,0,0.001f,0.0001f,100};
        const SurfaceInput probes[] = {
            {{0,0,0,1},0,1,0,0,1,1,1,1},
            {{0,-2,0,1},0,1,0,0,1,1,1,1},
            {{4,0,0,1},0,1,0,0,1,1,1,1},
            {{0,3,0,1},0,-1,0,0,1,1,1,1},
            {{2.4f,0,0,1},0,1,0,0,1,1,1,1},
            {}};
        for (size_t i=0; i<Count; ++i) cuda.Host[i]=probes[i%std::size(probes)];
        Upload(cuda);
        IndirectSettings settings={0,0.5f,0,91,0,0,0,32,0};
        pass.LaunchIndirect(cuda.Input,shadow,settings);
        const auto direct=Read(cuda,pass);
        const double panelReference=PanelIntegral({0,0,0},192);
        double panelSum=0; size_t panelCount=0,penumbraCount=0;
        for (size_t i=0; i<Count; ++i)
        {
            const auto probe=i%std::size(probes);
            const auto &d=direct[AdditionalDirectPlane*Count+i], &u=direct[AdditionalUnshadowedPlane*Count+i];
            Require(d.Valid==(probe!=5 ? 1u : 0u) && u.Valid==d.Valid && direct[i].R==0 && direct[i].G==0 && direct[i].B==0,
                "Direct light changed validity or produced zero-bounce indirect light.");
            if (probe==0)
            {
                Require(std::abs(d.R-4/Pi)<1e-6 && d.R==u.R && d.G==u.G && d.B==0 && std::abs(u.B-1)<1e-6,
                    "Point falloff, finite light distance or combined directional occlusion failed.");
                panelSum+=d.G; ++panelCount;
            }
            if (probe==1) Require(std::abs(d.R-1/Pi)<1e-6,"Doubling point distance must quarter the direct radiance.");
            if (probe==2) Require(d.R==0 && d.G==0 && u.R>0 && u.G>0,"Occluder should block both finite emitters.");
            if (probe==3) Require(d.G==0 && u.G==0 && d.B>0 && std::abs(d.R-16/Pi)<1e-6,
                "Rectangle must emit only from its front.");
            if (probe==4 && d.G>0 && d.G<u.G) ++penumbraCount;
        }
        Require(std::abs(panelSum/panelCount-panelReference)<0.003 && penumbraCount>1000,
            "Uniform-area integral or partial rectangle visibility disagrees with the independent check.");
        std::cout<<"PASS: point inverse-square falloff, finite shadow bounds, mixed lights, one-sided panel and partial shadows. "
            <<"Panel mean "<<panelSum/panelCount<<" vs quadrature "<<panelReference<<".\n";

        for (size_t i=0; i<Count; ++i) cuda.Host[i]=probes[0];
        Upload(cuda);
        settings.SamplesPerPixel=1; settings.MaxBounces=1;
        pass.LaunchIndirect(cuda.Input,shadow,settings);
        const auto indirect=Read(cuda,pass);
        std::vector<IndirectSampleDebug> debug(Count);
        CheckCuda(cuMemcpyDtoH(debug.data(),pass.GetDeviceIndirectDebug(),Count*sizeof(IndirectSampleDebug)),"cuMemcpyDtoH(light test hits)");
        double indirectPanelSum=0,indirectPanelReference=0;
        for (size_t i=0; i<Count; ++i)
        {
            const auto &hit=debug[i].Hit;
            double expectedPoint=0;
            if (hit.Valid && hit.FrontFace && hit.PositionY>3)
            {
                const double x=hit.PositionX,z=hit.PositionZ,r2=x*x+z*z+4;
                expectedPoint=32/(Pi*r2*std::sqrt(r2));
                indirectPanelReference+=PanelIntegral({x,hit.PositionY,z},24);
            }
            Require(std::abs(indirect[i].R-expectedPoint)<2e-6 && indirect[i].G==0,
                "Analytic point lighting at the secondary hit differs from OptiX.");
            indirectPanelSum+=indirect[i].B;
        }
        Require(indirectPanelReference>0 && std::abs(indirectPanelSum/indirectPanelReference-1)<0.025,
            "Indirect panel irradiance disagrees with the independent quadrature.");
        std::cout<<"PASS: point and panel after one bounce. Indirect panel mean "<<indirectPanelSum/Count
            <<" vs quadrature "<<indirectPanelReference/Count<<".\n";
    }
}
int main(int argc,char **argv)
{
    try
    {
        Require(argc==1 || (argc==2 && std::string(argv[1])=="--samples"),"Usage: HybridTracerIndirectTest [--samples]");
        DX12Context dx; dx.InitializeHeadless(); Cuda cuda; cuda.Initialize(dx.GetDevice());
        std::ofstream csv;
        if (argc==2) { csv.open("IndirectLightingSamples.csv"); Require(bool(csv),"Cannot open indirect samples.");
            csv<<"seed,pixel,hit,object,visibility,r,g,b,cpu_r,cpu_g,cpu_b\n"; }
        DefaultScene(dx.GetDevice(),cuda,csv ? &csv:nullptr); AnalyticPlane(dx.GetDevice(),cuda);
        AnalyticTwoBounces(dx.GetDevice(),cuda);
        AnalyticAdditionalLights(dx.GetDevice(),cuda);
        if (csv) { csv.flush(); Require(bool(csv),"Cannot flush indirect samples."); }
        std::cout<<"PASS: diffuse transport through two bounces, material weights, secondary shadows, miss/black/zero-light, seeds and SPP normalization.\n";
        return 0;
    }
    catch (const std::exception &error) { std::cerr<<"FAIL: "<<error.what()<<'\n'; return 1; }
}


