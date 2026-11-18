# OptixTracePass: 첫 directional shadow ray

후속 구현: [메인 화면 전체 패스](Hybrid_Frame_Pipeline.md). 아래 기록 이후 공유 shadow
출력·Composite·ToneMap·메인 Application을 연결했으며 현재 전체 CTest는 12/12다.
아래의 headless 전용 설명은 첫 shadow 단계의 범위다.

2026-11-03. GPU SurfaceInput과 정적 GAS 다음 단계로 실제 OptiX module / program
group / pipeline / SBT / optixLaunch를 구현했다. 카메라의 첫 표면은 기존 DX12
rasterization이 결정하고, OptiX는 그 표면에서 광원으로 향하는 secondary ray만 추적한다.

현재 실행 위치는 headless 진단 프로그램이다. 일반 `HybridTracer.exe`의 화면에
그림자를 합성하는 CompositePass 및 ToneMapPass 연결은 다음 단계다.

## 패스의 입출력과 역할

`Source/Optix/OptixTracePass.h/.cpp`는 테스트 코드와 분리된 실행 패스다.

- 초기화 입력: SurfaceInput의 CUDA context/stream, 살아 있는 OptixSceneGAS, 화면 크기.
- 프레임 입력: GPU SurfaceInput 배열과 ShadowSettings.
- 출력: GPU ShadowResult 배열, 픽셀당 16바이트. CPU로 내려보내는 기능은 패스에 없다.
- Launch는 기존 SurfaceInput 생성 stream에 작업을 추가한다. CPU 대기는 하지 않는다.
- 패스는 pipeline/program/module/SBT/launch parameter/output을 소유한다.
- CUDA/OptiX context와 GAS는 빌려 쓴다. 파괴 순서는 TracePass → GAS → CUDA 소유자다.

SurfaceInput 정의를 `Shaders/SurfaceInput.h`로 옮겨 host와 OptiX가 같은 구조를 사용한다.
기존 CUDA packing 커널이 쓰는 48바이트 P/N/Albedo 계약과 실제 값 검증도 유지한다.
`Shaders/OptixTraceShared.h`에는 다음 결과와 설정, launch parameter ABI를 정의했다.

```cpp
struct ShadowResult {
    float Visibility;       // 유효 표면: miss=1, hit=0
    unsigned int Blocker;   // 맞은 SceneObject 번호 + 1, miss/배경=0
    float HitDistance;      // offset 적용된 광선 원점에서의 t, miss/배경=0
    unsigned int Valid;     // 표면=1, 배경=0
};
```

배경은 모든 성분을 0으로 쓰고 optixTrace를 호출하지 않는다. Visibility만으로는
배경과 차폐 표면을 구분할 수 없으므로 Valid도 사용한다. 모든 유효 표면을 추적하며,
최종 직접광의 `max(dot(N,L),0)` 적용은 앞으로 CompositePass가 담당한다.

## 광선과 SBT

현재 Scene의 DirectionalLight.Direction은 **표면 → 광원** 방향이다. host가 정규화해
설정으로 넘기며, shader에서 다시 부호를 뒤집지 않는다. 테스트는 홀수 프레임에
Scene 방향 (-0.5,1,-0.5), 짝수에는 (0.7,1,0.2)를 사용해 광원과 카메라를 바꾼다.

```text
origin = WorldPosition + normalizedNormal * 0.001
direction = normalized surface-to-light direction
tmin = 0.0001, tmax = 1e16
```

offset은 현재 수 단위 장면의 위치 복원 오차(약 0.00022)보다 크게 잡았다. 임의의
장면 크기와 grazing ray를 모두 해결하는 일반적인 오차 경계 방식으로 검증한 것은
아니다. 크기·재질·geometry를 확장할 때 다시 정해야 한다.

단일 GAS, triangle primitive, trace depth=1, payload 2개, attribute 2개를 사용한다.
raygen/miss/closest-hit 각 1개와 SceneObject별 hit record 4개를 만든다. GAS의
build input별 numSbtRecords=1과 같은 순서로 ObjectIndex를 넣는다. SBT record는
SDK가 요구하는 정렬/헤더 크기를 사용하며, stack 크기는 SDK helper로 계산한다.

처음에는 **closest-hit**으로 물체 번호와 거리까지 검증한다. any-hit은 비활성화하고
삼각형 양면을 불투명하게 취급한다. 반투명 재질, alpha test, 재귀 반사, soft shadow,
path throughput/BSDF 계산은 포함하지 않는다. 검증 이후 visibility 전용 경로는
terminate-on-first-hit 방식으로 최적화할 수 있다.

## 컴파일과 동기화

CUDA 12.0 NVRTC가 MX450의 compute_75 PTX를 만들고 OptiX 8.0.0이 module을 생성한다.
NVCC나 MSVC 14.38을 이 실행 경로에서 호출하지 않는다. 기존 host 빌드 설정도 유지한다.

이 환경에서는 NVRTC의 include-path로 한글 SDK 경로를 넘기면 `optix.h: Invalid
argument`가 발생했다. SDK header는 C++ filesystem의 Unicode 경로로 읽어 NVRTC의
이름 있는 메모리 header로 전달한다. SDK는 외부 설치 경로에 유지하며 저장소나 실행
폴더에 복사하지 않는다. 실행 시에도 configure한 SDK include 경로가 존재해야 한다.

```text
DX12 raster + 공유 fence
    → CUDA wait → G-buffer copy / SurfaceInput 생성
    → 같은 stream에서 parameter upload → OptixTracePass::Launch
    → 진단용 stream 완료 대기 → shadow readback → CPU 비교
```

기존 SurfaceInput 모듈의 CUDA 완료 fence는 SurfaceInput 작업 뒤에 signal된다.
현재 DX12 재개 작업은 G-buffer만 읽고, OptiX는 이미 분리된 SurfaceInput 배열을 읽으므로
겹쳐도 된다. **이 fence가 shadow 결과 완료를 보장하는 것은 아니다.** 다음 DX12 합성
단계에서는 공유 shadow 출력과 그 뒤의 CUDA signal / DX12 wait를 추가해야 한다.

launch parameter는 pinned host memory에서 비동기로 업로드한다. 완료 event를 기록해
이전 launch가 끝나기 전에 같은 parameter/output을 재사용하면 실패하도록 검사한다.
현재 진단 루프는 프레임마다 완료를 기다린다. 실시간 다중 프레임 자원화는 다음 통합 범위다.

## 독립적인 검증

`ShadowValidation.cpp`는 원본 Scene/Mesh에 scalar/double world transform을 적용하고,
모든 유효 표면의 광선을 모든 삼각형과 brute-force 교차 검사한다. OptiX/GAS의 출력
정점이나 교차 결과를 CPU 기준값으로 재사용하지 않는다. 원점의 float 반올림과 입력
방향은 GPU와 같은 계약을 사용하고, 삼각형 검사는 독립적인 double 계산이다.

- 배경 전체가 zero이며 유효 표면의 visibility가 정확히 0/1인지 검사한다.
- 모든 픽셀의 hit/miss를 비교한다. 경계 픽셀을 제외하지 않는다.
- hit마다 SBT 물체가 실제로 CPU 광선에 맞는지와 closest distance를 검사한다.
- distance 허용 오차는 `0.00005 * max(1, CPU distance)` 월드 단위다.
- background, hit, miss, 물체가 바닥에 드리운 그림자가 모두 있어야 통과한다.
- 기존 모든 G-buffer, Position, SurfaceInput, GAS 검사와 D3D12 validation도 유지한다.

실제 MX450 / 드라이버 560.94 / OptiX 8.0.0에서 Debug 빌드로 확인했다.

| 해상도 | 프레임 | 전체 픽셀 검사 | 홀수/짝수 hit 수 | 최대 거리 오차 |
|---|---:|---:|---:|---:|
| 321×181 | 4 | 58,101 / frame | 639 / 2,489 | 6.1727704e-7 |
| 1280×720 | 4 | 921,600 / frame | 10,114 / 39,282 | 6.86021378e-7 |

두 해상도 모두 CPU와 hit/miss 불일치 0개였다. 1280×720에서는 바닥을 포함한
네 종류의 blocker record가 실제 hit로 검사됐다. 기존 8개에 shadow 검사 2개를 더한
CTest **10/10 통과**. 이 시간에는 컴파일/시작/CPU readback/기준 계산이 포함되므로
GPU 렌더링 프레임 시간이나 성능 수치로 사용하지 않는다.

OptiX 최초 launch에서 아래 메시지가 2회 기록됐다.

```text
OptiX [3][MEMORY] cblCmdListBufferSize insufficient, attempting allocation
```

NVIDIA가 무해한 내부 allocation notice라고 설명한 **정확한 level/tag/message만**
별도 집계한다. 로그는 숨기지 않으며 횟수를 출력한다. 나머지 warning/error는 모두
실패 처리한다. D3D12 warning/error 및 예상 밖 OptiX warning/error는 0개였다.
[NVIDIA 설명](https://forums.developer.nvidia.com/t/freeze-in-optix-scene-update-on-windows-with-drivers-555-99-and-560-70/301834/2).
드라이버 변경이나 deprecated launcher 환경변수 설정은 하지 않았다.

## 실행

GAS와 같은 CMake 옵션 `HYBRIDTRACER_BUILD_OPTIX_GAS_TEST=ON`을 사용한다.
SDK 설정과 전체 build 명령은 [GAS 문서](Optix_Scene_GAS.md)를 따른다.

```powershell
Set-Location -LiteralPath 'C:\GitHub\HybridRenderer\HybridTracer\build'
.\Debug\HybridTracerNormalInteropTest.exe --optix-shadow --frames 4 --samples
$LASTEXITCODE
# 전체 해상도 검사
.\Debug\HybridTracerNormalInteropTest.exe --optix-shadow --frames 4 --full-resolution
```

마지막 PASS와 exit code 0을 확인한다. `--samples`는 현재 실행 폴더에
`OptixShadowSamples.csv`를 쓰며 background/miss 및 각 blocker의 대표 픽셀을 기록한다.
기존 OptixNormal/Position/SurfaceInput/GAS CSV도 함께 생성한다.

## 제공된 코드베이스를 적용한 부분

- DXRPathTracer: 패스 입출력/실행 책임을 분리하는 원칙. 테스트 안에 trace 구현을 넣지
  않고 명시적인 GPU 입력·출력이 있는 OptixTracePass로 작성했다.
- [RoyalTracer KernelLights](https://github.com/Royal-Project-Group/royaltracer/blob/fae76756d42d1c36af0c558f24f24017d8dded35/backend/src/raytracer/kernelFiles/KernelLights.h#L128):
  표면에서 광원 방향으로 광선을 보내 visibility를 구하는 흐름. 방향 부호, cone jitter,
  자체 BVH 호출을 그대로 옮기지 않고 현재 방향 계약과 OptiX로 구현했다.
- NVIDIA 설치 SDK 8.0.0 `optixTriangle`: host/device API, program group, SBT 정렬,
  stack 계산 호출을 대조했다. 구현은 프로젝트의 SurfaceInput/Driver API에 맞춰 작성했다.
- TracerBoy: 다음 Composite/ToneMap 단계의 참고 대상으로 유지한다.

다음 작업은 shadow GPU 결과를 DX12에서 읽을 공유 출력으로 연결하고, CompositePass가
직접광에 visibility를 적용한 뒤 ToneMapPass를 거쳐 메인 화면에 표시하도록 통합하는 것이다.
