# Position · Normal · Albedo를 GPU 표면 입력으로 묶기

2026-10-30, Depth/World Position 복원 다음으로 구현한 단계다.
`HybridTracerNormalInteropTest.exe --surface-inputs`에서 **실제 Normal/Depth/Albedo
G-buffer 세 개를 CUDA에서 읽고, 픽셀별 표면 정보를 GPU 버퍼 하나에 묶는다.**
아직 OptiX launch와 메인 Application 통합은 포함하지 않는다.

2026-11-03에 후속 **OptiX 장면 geometry 업로드와 GAS 빌드**도 완료했다.
[OptiX GAS 문서](Optix_Scene_GAS.md)에 데이터 배치와 실제 검증 수치를 정리했다.

## 먼저 확인한 개발 환경 변경

사용자가 설치했다고 알려 준 툴셋은 **MSVC 14.38.33130**이었다. 현재 PC에는
14.38과 14.44가 함께 설치돼 있으며 CUDA Toolkit은 12.0이다.

실제로 `vcvarsall.bat x64 -vcvars_ver=14.38`로 환경을 선택하고, NVCC에
`--compiler-bindir`로 14.38의 `Hostx64/x64` 경로를 지정해 다음 두 커널을 PTX로
컴파일하는 데 성공했다.

- ReconstructWorldPosition
- BuildSurfaceInputs

검증 옵션은 `--ptx --gpu-architecture=compute_75 --fmad=false -std=c++14`와
MSVC UTF-8 옵션 `-Xcompiler /utf-8`이다. `-allow-unsupported-compiler`는 사용하지
않았다. 생성된 PTX에 두 entry가 들어 있는 것도 확인했다. 이 검증은 **NVCC의 실제
오프라인 컴파일 성공**이며, 해당 PTX를 OptiX 모듈로 실행한 검증은 아니다.

현재 프로그램은 기존처럼 NVRTC로 커널을 컴파일해 실행한다. 일반 C++ Debug 빌드는
기존 14.44 설정을 사용한다. 앞으로 NVCC를 사용할 때는 설치된 **14.38을 명시적으로
선택**하면 된다. 설치만으로 기존 CMake 빌드 폴더의 툴셋이 자동 변경되지는 않는다.

## 입력과 GPU 출력

| 입력 | DX12 저장 형식 | CUDA 해석 |
|---|---|---|
| Depth RT3 | R32_FLOAT | 비선형 D3D Depth, float 1개 |
| Normal RT2 | R16G16B16A16_FLOAT | signed half 4개, World Space |
| Albedo RT1 | R8G8B8A8_UNORM | unsigned byte 4개, 성분별 `/255.0f` |

Albedo는 Material의 기본 재질색이다. Lambert·Ambient를 적용한 RT0 Shaded Color를
입력으로 사용하지 않는다. 현재 UNORM 값을 수치로 변환하며 sRGB 변환은 적용하지 않는다.
Albedo alpha는 현재 표면 1/배경 0이며, 투명도 렌더링을 구현한 의미는 아니다.

GPU 출력 `SurfaceInput`은 픽셀당 48바이트, float4 그룹 세 개로 구성한다.

```cpp
struct SurfaceInput
{
    WorldPosition Position; // float X/Y/Z/Valid, offset 0
    float NormalX, NormalY, NormalZ, NormalPadding; // offset 16
    float AlbedoR, AlbedoG, AlbedoB, AlbedoA;      // offset 32
}; // sizeof == 48
```

`WorldPosition.Valid`가 유효 표면의 공통 판정이다. 배경은 12개 float 모두 0으로
기록한다. NormalPadding은 항상 0이다. Normal은 half 저장값을 float로 변환한 뒤
GPU에서 다시 정규화한다. 따라서 원본 Normal RT의 길이 오차를 입력 단계에서 보정한다.
정규화는 이미 존재하는 방향 벡터의 길이를 맞추는 것이며 기하 법선을 새로 계산하지 않는다.

## 한 프레임의 데이터 흐름

```text
DX12 RasterPass
  → 실제 Albedo / World Normal / Depth RT 생성
  → 세 리소스를 COMMON으로 전이
  → 공유 Fence Signal

CUDA (Depth/Position reader가 소유한 하나의 컨텍스트와 stream)
  → Fence Wait
  → 세 텍스처를 CUDA array로 import해 읽기
  → array → GPU 선형 입력 버퍼 복사
  → ReconstructWorldPosition: Depth + inverse VP → GPU Position
  → BuildSurfaceInputs: Position + Normal + Albedo → GPU SurfaceInput[]
  → 검증용 pinned CPU 메모리로 출력 복사
  → 공유 Fence Signal

DX12 queue Wait
  → 세 리소스를 RENDER_TARGET 상태로 복구
  → DebugView 사용 / 다음 프레임
```

Surface 모드에서 입력 세 개와 출력 버퍼는 **동일한 CUDA 컨텍스트**에 속한다.
기존 별도 `CudaNormalReader`는 독립적인 원본 Normal 관찰 경로로 함께 실행한다.
Normal/Depth의 기존 수치 검사도 유지한다. G-buffer나 SurfaceInput을 CPU에서
조합해 GPU로 재업로드하지 않는다. 카메라 행렬만 CPU에서 GPU로 전달한다.

`CudaDepthPositionReader`는 GPU SurfaceInput 포인터, 소유 컨텍스트, stream을
getter로 제공한다. 이후 OptiX를 같은 컨텍스트에 연결할 수 있는 입력이 준비된
상태다. 이 포인터의 수명은 리더가 소유한 버퍼 수명에 묶인다. 현재 실행은 검증을
위해 프레임 끝에서 CPU가 기다리며, 실시간 다중 프레임 스케줄은 별도 작업이다.

## 어떻게 검증했는가

- **모든 픽셀의 Albedo 원본 4바이트**를 DX12 Readback과 CUDA Readback에서 비교한다.
  행 패딩은 DX12의 GetCopyableFootprints/RowPitch로 처리한다.
- 모든 픽셀의 묶음 Position이 기존 CUDA Position 버퍼와 비트 단위로 같은지 검사한다.
- Normal을 CPU에서도 원본 half 값에서 변환·정규화해 GPU 출력과 비교하고 단위 길이를
  확인한다. 성분 오차와 길이 오차의 허용값은 각각 0.00001이다.
- Albedo float가 원본 byte/255와 일치하는지 확인한다. 변환 오차 허용값은 0.0000001이다.
- 배경 전체 zero, 유효 표면 alpha=1, 유한한 값도 확인한다.
- 기존 CPU 최근접 ray/triangle 교차 표본에서 해당 **SceneObject의 Material.Albedo**와
  GPU 묶음의 Albedo를 비교한다. 위치와 재질이 같은 픽셀·같은 물체를 가리키는지 검사한다.

Material float와 UNORM 값의 비교는 양자화 때문에 `0.5/255 + 0.000001`의 오차를
허용한다. 반면 원본 RGBA8 전달 자체는 오차 없이 바이트 단위 일치를 요구한다.
기하 표본은 기존과 같이 삼각형 내부만 사용하며 네 SceneObject 모두 포함한다.
모든 픽셀의 최근접 삼각형을 CPU에서 전수 검증한 것은 아니다.

## 실제 결과

321×181과 1280×720에서 각각 4프레임을 검사했다. 카메라 위치 및 near/far를 번갈아
바꿔 이전 프레임 입력이 남는 경우도 검사한다. 두 해상도 모두 전체 Albedo가 바이트
단위로 일치했고, 묶음 데이터와 네 물체의 재질 대응 검사도 통과했다.

321×181 첫 프레임의 대표 입력:

| 물체 | 픽셀 (x,y) | GPU World Position | GPU 정규화 Normal | GPU Albedo RGB |
|---|---|---|---|---|
| 바닥 | (82,91) | (-3.707760,-0.999927,3.464984) | (0,1,0) | (0.450980,0.450980,0.450980) |
| 빨간 큐브 | (95,91) | (-2.092175,0.337126,-0.446115) | (-0.955345,0,-0.295493) | (0.850980,0.098039,0.078431) |
| 파란 큐브 | (173,80) | (0.538750,0.199674,1.075684) | (-0.230055,0.969064,-0.089384) | (0.078431,0.250980,0.898039) |
| 기울어진 평면 | (147,47) | (-0.439331,1.182198,3.165056) | (-0.358004,0.602209,-0.713566) | (0.098039,0.800000,0.200000) |

예를 들어 Material의 0.1과 관찰된 25/255≈0.098039는 UNORM 양자화 차이다.
광원이 어둡게 만든 색을 Albedo에 저장한 결과가 아니다.

기존 버퍼·Normal·Depth/Position 테스트와 새 Surface 두 해상도를 합쳐 CTest
**7/7 통과**, D3D12 debug layer warning/error 없음, 기본 렌더러 Debug 빌드 성공을
확인했다. 원래 Shader 출력 경로도 기존 DebugView 0~3 명령을 순환하며 검사한다.

## 실행과 CSV

기존 interop 옵션을 켜 빌드한 뒤 실행한다.

```powershell
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
& $cmakeExe -S . -B build -DHYBRIDTRACER_BUILD_INTEROP_TEST=ON
& $cmakeExe --build build --config Debug
.\build\Debug\HybridTracerNormalInteropTest.exe --surface-inputs --frames 4 --samples
.\build\Debug\HybridTracerNormalInteropTest.exe --surface-inputs --frames 4 --full-resolution
& (Join-Path (Split-Path $cmakeExe) 'ctest.exe') --test-dir build -C Debug --output-on-failure
```

CSV는 실행한 현재 폴더에 저장하며 CTest로 실행하면 `build`에 저장한다.

| 파일 | 내용 |
|---|---|
| SurfaceInputSamples.csv | 픽셀별 Position/Normal/Albedo, CPU Material, 위치 오차; 4프레임×4물체=16행 |
| SurfacePositionSamples.csv | GPU 위치와 CPU 최근접 교차 위치 |
| PackedSurfaceNormalSamples.csv | 원본 Normal의 DX12/CUDA 값과 half 비트 |

`--surface-inputs`는 Depth/Position 모드도 활성화한다. 기존 `--depth-position`과
Normal 전용 모드는 계속 사용할 수 있다. 기본 `HybridTracer.exe`는 아직 DX12
렌더러이며, 일반 초기화의 공유 옵션은 모두 false다.

## 다음 작업

Raster Primary Hit의 위치·방향·재질 입력에 이어 CPU Scene의 원본 삼각형을 OptiX에
올리고 GAS를 만드는 단계까지 검증했다. 다음은 pipeline/SBT/launch와 첫 shadow ray다.
아직 OptiX 교차 추적이나 최종 그림자/반사 합성은 없다.

공유 텍스처 규약은
[CUDA 12.0.1 interop 가이드](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/index.html#direct3d-12-interoperability),
half 변환은 설치된 CUDA Toolkit의 `cuda_fp16.h`를 사용한다. 실행 시 컴파일은
[NVRTC 문서](https://docs.nvidia.com/cuda/archive/12.0.1/nvrtc/index.html)를 참고한다.
