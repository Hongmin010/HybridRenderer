# OptiX Scene geometry 업로드와 GAS

후속 구현: [OptixTracePass와 Shadow Hit/Miss](Optix_Shadow_Trace.md). 아래 내용은
GAS 단독 단계의 기록이며, 현재는 별도 `--optix-shadow` 모드에서 pipeline/SBT/launch와
CPU 그림자 비교까지 구현했다. 전체 검사는 10/10이며 기존 GAS 모드도 유지한다.

2026-11-03, GPU SurfaceInput(Position/Normal/Albedo) 준비 다음으로 구현한 단계다.
현재 Scene의 원본 삼각형을 월드 좌표로 변환해 CUDA에 올리고, **실제 OptiX 8.0.0
API로 GAS(Geometry Acceleration Structure)**를 빌드했다.

이번 범위는 장면 교차 검색 구조 생성이다. OptiX module/program group/pipeline/SBT와
`optixLaunch`는 아직 없으며, Hit/Miss·그림자·반사 결과를 검증한 단계는 아니다.

## 두 종류의 GPU 데이터가 필요한 이유

```text
Scene / Camera
  ├─ DX12 Rasterization → 화면에서 보이는 첫 표면
  │    └─ CUDA SurfaceInput[] : World Position / Normal / Albedo
  │
  └─ Mesh 전체 삼각형 + Object Transform
       └─ CUDA world-space vertex/index buffers → OptiX GAS

다음 단계: SurfaceInput의 표면에서 광선을 만들어 GAS에 대해 교차 검색
```

G-buffer에는 현재 카메라에서 보이는 표면만 들어 있다. Secondary Ray가 다른 물체나
가려진 면을 만나는지 알아내려면 **장면의 전체 geometry**가 필요하다. 그래서 GAS는
G-buffer에서 보이는 위치들을 모아 만드는 것이 아니라 원본 `Scene/Mesh`에서 만든다.

## 컨텍스트와 수명

`OptixSceneGAS::Initialize()`는 `CudaDepthPositionReader`가 SurfaceInput을 만든
**동일 CUDA 컨텍스트·동일 stream**을 받는다. `cuStreamGetCtx()`로 둘의 소유 관계를
확인하고, 그 컨텍스트를 인자로 `optixDeviceContextCreate()`를 호출한다.

OptixDeviceContext는 CUDA 컨텍스트에 연결된 OptiX API 객체다. 별도의 CUDA 장치를
선택하거나 다른 GPU로 데이터를 옮기지 않는다. DX12/CUDA의 MX450 일치 검사는
기존 LUID 검사로 유지한다. 이번 실행의 LUID는 `00000000:00014aaa`였으며 상수로
저장해 다음 실행에 재사용하지 않는다.

GAS 객체는 CUDA 소유자보다 먼저 파괴한다. stream 완료 후 GAS 출력과 geometry
GPU 메모리를 해제하고 OptiX 컨텍스트를 파괴한다. CUDA 컨텍스트·stream 자체는
기존 SurfaceInput 소유자가 관리한다.

## geometry 입력 계약

SceneObject마다 기존 Mesh와 World Transform을 읽어 정점을 월드 좌표로 변환한다.
동일 Mesh를 공유하는 물체도 서로 다른 World Transform을 적용하므로, 이번 정적
구조에서는 GPU geometry를 오브젝트별로 만든다.

| SceneObject 순서 | 대상 | 정점 수 | 삼각형 수 |
|---:|---|---:|---:|
| 0 | 바닥 | 4 | 2 |
| 1 | 빨간 큐브 | 24 | 12 |
| 2 | 파란 큐브 | 24 | 12 |
| 3 | 기울어진 초록 평면 | 4 | 2 |
| 합계 | 하나의 GAS, build input 4개 | 56 | 28 |

정점 GPU 형식은 `GasVertex {float X,Y,Z,Padding}`(16바이트)이며 OptiX에는
`OPTIX_VERTEX_FORMAT_FLOAT3`, stride=16을 지정한다. 인덱스는 원본 uint16 배열의
삼각형 순서를 유지하면서 `GasTriangle {uint32 X,Y,Z}`(12바이트)로 업로드하고,
`OPTIX_INDICES_FORMAT_UNSIGNED_INT3`, stride=12를 사용한다.

각 build input은 `numSbtRecords=1`이다. 앞으로 hitgroup 레코드를 만들 때
SceneObject 순서와 대응시키기 위한 계약이며, **현재 SBT를 생성한 것은 아니다.**
GAS에는 교차 계산에 필요한 geometry만 넣는다. SurfaceInput의 Normal/Albedo는
별도의 표면 입력으로 유지한다.

이번에는 하나의 world-space GAS를 만들며 IAS·인스턴싱·동적 update·compaction은
도입하지 않았다. World Transform을 변경하면 새 GAS 객체로 다시 빌드해야 한다.
카메라 이동만으로 GAS geometry가 달라지지는 않는다.

## 실제 빌드 흐름

1. 기존 CUDA 컨텍스트를 선택하고 `optixInit()`로 드라이버 API를 로드한다.
2. OptiX 컨텍스트를 생성하고 validation ALL 및 warning/error 로그를 활성화한다.
3. SceneObject의 월드 정점과 원본 인덱스를 CUDA 메모리에 업로드한다.
4. `OptixBuildInput` 4개를 구성한다.
5. `optixAccelComputeMemoryUsage()`로 scratch/output 크기를 구한다.
6. CUDA 메모리를 할당하고 `optixAccelBuild()`로 GAS를 만든다.
7. 같은 stream의 완료를 기다린 뒤 scratch를 해제한다.

정적 장면용 `OPTIX_BUILD_FLAG_PREFER_FAST_TRACE`와 BUILD operation을 사용한다.
motion key는 1개다. geometry 버퍼와 GAS 출력은 객체 수명 동안 유지한다.
GAS 출력 핸들은 불투명한 `OptixTraversableHandle`이며, CPU 포인터로 역참조하지 않는다.

## 검증 범위와 결과

빌드 직후와 4프레임의 기존 CUDA/DebugView 작업 뒤에 같은 검사를 수행한다.

- CUDA에서 정점·인덱스를 다시 읽어 실제 SceneObject 수와 정점/삼각형 수를 검사한다.
- GPU의 모든 인덱스 triplet을 원본 Mesh 인덱스와 정확히 비교한다.
- GPU의 모든 월드 정점을 원본 로컬 정점과 World 행렬의 독립적인 scalar/double
  계산값과 비교한다. 업로드에 사용한 staging 배열을 기준으로 재사용하지 않는다.
- OptiX가 `OPTIX_PROPERTY_TYPE_AABBS`로 출력한 **GAS root 경계 상자**를 CPU의
  원본 삼각형 전체 경계와 비교한다. 허용 성분 오차는 0.00001 월드 단위다.
- `optixAccelGetRelocationInfo()`가 생성된 핸들을 인식하는지도 확인한다.
  실제 relocation은 수행하지 않는다.
- OptiX validation warning/error가 없는지 확인한다. 기존 D3D12와 P/N/Albedo 검사도
  같은 테스트 프로세스에서 계속 실행한다.

실제 MX450 / 드라이버 560.94 / SDK 8.0.0에서 확인한 결과:

| 항목 | 결과 |
|---|---|
| geometry | 오브젝트 4개, 정점 56개, 삼각형 28개 |
| GAS output 크기 | 7,040바이트 |
| build scratch 크기 | 2,944바이트, 완료 후 해제 |
| 정점 최대 성분 오차 | 1.415610313e-7 |
| root AABB 최대 성분 오차 | 1.758337021e-6 |
| 초기 / 프레임 작업 후 검사 | 모두 통과 |
| OptiX / D3D12 validation | warning/error 없음 |
| 기존 검사 포함 CTest | 8/8 통과 |

```text
CPU world bounds
  min = (-4, -1.154223591, -3.5)
  max = ( 4,  1.550446421,  4.5)

OptiX emitted root AABB
  min = (-4.000001431, -1.154225111, -3.500001669)
  max = ( 4.000001431,  1.550448179,  4.500001431)
```

OptiX의 경계 상자는 부동소수점 정밀도와 보수적인 확장 때문에 CPU 경계와 완전한
비트 일치를 요구하지 않는다. 메모리 크기 역시 이 실행의 결과이며 장면·SDK·드라이버에
따라 달라질 수 있다. 이 검증은 geometry 전달과 GAS 빌드의 근거이며, 아직 실제
광선이 올바른 삼각형을 맞히는지 증명한 것은 아니다.

## 빌드와 실행

설치된 SDK는 다음 위치에서 찾았다.

```text
C:\Users\hongm\Documents\ChatGPT\졸업프로젝트\references\optix-sdk-8.0
```

SDK 파일을 프로젝트에 복사하지 않고 해당 include 경로를 사용한다. Windows에서는
SDK stubs가 설치된 NVIDIA 드라이버의 OptiX 라이브러리를 로드한다. 이번 GAS 단계는
일반 C++에서 OptiX host API를 호출하며, 새 OptiX 디바이스 프로그램은 아직 컴파일하지
않는다. 앞서 확인한 MSVC 14.38/NVCC 조합은 다음 디바이스 코드 단계에 사용할 수 있다.

프로젝트 루트에서 PowerShell로 실행한다.

```powershell
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$optixSdk = 'C:\Users\hongm\Documents\ChatGPT\졸업프로젝트\references\optix-sdk-8.0'
& $cmakeExe -S . -B build -DHYBRIDTRACER_BUILD_INTEROP_TEST=ON -DHYBRIDTRACER_BUILD_OPTIX_GAS_TEST=ON "-DHYBRIDTRACER_OPTIX_ROOT=$optixSdk"
& $cmakeExe --build build --config Debug
.\build\Debug\HybridTracerNormalInteropTest.exe --optix-gas --frames 4 --samples
& (Join-Path (Split-Path $cmakeExe) 'ctest.exe') --test-dir build -C Debug --output-on-failure
```

GAS 빌드 옵션의 기본값은 OFF이며 SDK 탐색은 옵션을 ON으로 켰을 때만 수행한다.
`--optix-gas`는 SurfaceInput 및 Depth/Position 모드도 함께 활성화한다.
일반 `HybridTracer.exe`는 계속 DX12 렌더러이며 새로운 OptiX 호출은 진단 모드에서 실행한다.

## CSV와 파일별 역할

`--samples`는 실행한 현재 폴더에 파일을 쓴다. CTest에서는 `build` 폴더다.

| 파일 | 내용 |
|---|---|
| OptixGASReport.csv | 오브젝트 4행과 GAS root 1행, 정점·삼각형 수 및 CPU/GPU 경계 |
| OptixSurfaceInputSamples.csv | 같은 프로세스에서 확인한 Position/Normal/Albedo 표본 |
| OptixPositionSamples.csv | CUDA 위치와 CPU 교차 위치 |
| OptixNormalSamples.csv | 원본 Normal half 값과 비트 |

`Source/Optix/OptixSceneGAS.h/.cpp`가 geometry GPU 메모리와 OptiX GAS를 소유한다.
`CudaNormalInteropTest.cpp`에는 선택적 테스트 모드와 수명 순서, 프레임 작업 전후
검사를 추가했다. CMake에는 SDK 경로와 빌드 옵션, 추가 CTest 하나를 등록했다.
`optix_function_table_definition.h`는 구현 .cpp 한 곳에서만 include한다.

다음 단계는 OptiX module/program group/pipeline/SBT/launch를 구성하고, SurfaceInput의
위치·법선에서 시작하는 첫 shadow ray의 Hit/Miss를 검사하는 것이다. 최종 DX12 합성과
실시간 다중 프레임 통합은 그 이후에 연결한다.

공식 참고:
[OptiX 8 host API](https://raytracing-docs.nvidia.com/optix8/api/optix__host_8h.html),
[triangle build input](https://raytracing-docs.nvidia.com/optix8/api/struct_optix_build_input_triangle_array.html),
[emitted AABB와 build flags](https://raytracing-docs.nvidia.com/optix8/api/group__optix__types.html).
실제 API 사용은 설치된 SDK 8.0.0 헤더와 optixTriangle 샘플에도 대조했다.
