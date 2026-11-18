# Depth 공유와 CUDA World Position 복원

2026-10-30, 실제 Normal G-buffer 읽기 다음으로 구현한 단계다.
기존 `HybridTracerNormalInteropTest.exe`의 `--depth-position` 모드에서
**실제 R32_FLOAT Depth RT3를 CUDA로 공유하고, CUDA 커널에서 위치를 계산**한다.
기존 Normal 검사도 함께 수행한다. OptiX tracing 및 메인 Application 통합은 아직 없다.

후속으로 **Albedo 공유와 GPU Position/Normal/Albedo 입력 묶음**도 구현·검증했다.
[Surface 입력 문서](DX12_CUDA_Surface_Inputs.md)에 원본 수치와 MSVC 14.38 확인 결과를
정리했다. 아래 내용은 Depth/Position 검증 단계의 설명이다.

## 입력과 출력

| 데이터 | 규약 |
|---|---|
| Depth 입력 | Raster.hlsl의 SV_Position.z, R32_FLOAT, D3D 비선형 [0,1] 깊이 |
| 화면 좌표 | viewport 원점 (0,0), 너비 W/높이 H, 왼쪽 위 기준, MSAA 없음 |
| 카메라 | DirectX LH, row-vector: World × View × Projection |
| 행렬 입력 | inverse(View × Projection), row-major float 16개, 전치하지 않음 |
| GPU 출력 | 픽셀당 float 4개: World X/Y/Z와 Valid(표면 1, 배경 0) |
| 배경 출력 | Depth=1이면 (0,0,0,0) |

공유하는 Depth는 깊이 테스트용 D32_FLOAT DSV가 아니라 기존의 **별도 Depth Render
Target RT3**다. Depth 값을 카메라와 표면 사이의 선형 거리로 해석하지 않는다.
현재 LESS 깊이 테스트와 배경 Clear=1 계약에서는 far plane 끝의 hit도 배경과 구분하지
않는다. reversed-Z, viewport 부분 영역, MSAA, resize는 이번 검증 범위에 없다.

## 복원식

픽셀 인덱스 (x,y)의 중심을 사용한다.

```text
u = (x + 0.5) / W
v = (y + 0.5) / H

ndcX = 2u - 1
ndcY = 1 - 2v
ndcZ = Depth(x,y)

h = (ndcX, ndcY, ndcZ, 1) × inverse(View × Projection)
WorldPosition = h.xyz / h.w
```

화면 y가 아래로 증가하므로 NDC y에서는 방향을 뒤집는다. D3D z는 [0,1]이므로
`2*depth-1`로 바꾸지 않는다. 마지막 `h.w` 나눗셈이 원근 투영을 되돌리는 과정이다.

HLSL 상수 버퍼 업로드에서 사용하던 행렬 전치는 여기에는 적용하지 않는다.
CUDA 커널이 row-major 배열을 직접 인덱싱하므로 DirectXMath의 inverse 행렬을
그대로 보낸다. 입력 행렬은 매 프레임 갱신하고 pinned CPU 버퍼를 거쳐 GPU로 전송한다.
이 행렬 전송은 작은 카메라 파라미터 업로드이며, G-buffer를 CPU로 왕복시키는 경로가 아니다.

## 메모리와 실행 흐름

```text
DX12 RasterPass → Normal/Depth G-buffer 생성
  Normal, Depth: RENDER_TARGET → COPY_SOURCE
  DX12 CopyTextureRegion → 비교용 Readback
  Normal, Depth: COPY_SOURCE → COMMON

Normal reader: DX12/CUDA 공유 Fence로 기존 Normal 읽기

Depth/Position reader:
  DX12 Signal(rasterReady, n)
  CUDA Wait(rasterReady, n)
  CUDA Depth CUarray → GPU 선형 float 버퍼 (device-to-device copy)
  CUDA 커널: Depth + inverse VP → GPU WorldPosition 버퍼
  CUDA Depth/Position → pinned CPU 검증 버퍼
  CUDA Signal(cudaDone, n)
  DX12 Wait(cudaDone, n)

  Normal, Depth: COMMON → RENDER_TARGET
  DX12 DebugView의 4개 모드 순환
  CPU: 완료 대기 → 원본 수치/재투영/기하 검증
```

Depth 텍스처는 `CU_AD_FORMAT_FLOAT`, channel 1, color-attachment 플래그의 CUDA
array로 매핑한다. 커널은 CUDA 내부의 device-to-device 복사로 얻은 선형 버퍼를
읽는다. **CPU에서 위치를 계산해 GPU로 재업로드하지 않는다.** CPU Readback은
검증용 관찰 경로다. Normal과 Depth 모듈은 현재 각각 CUDA 컨텍스트를 소유하므로
각 API 진입 시 자신의 컨텍스트를 선택한다. 두 모듈은 동일한 MX450을 사용한다.
향후 OptiX와 함께 쓸 컨텍스트·stream·GPU 자원 소유 구조는 통합 단계에서 정리한다.

## NVRTC를 사용한 실제 CUDA 커널

`Shaders/ReconstructWorldPosition.cu`를 NVRTC 12.0으로 실행 시 PTX로 컴파일하고
CUDA Driver API로 로드·실행한다. MX450에서는 `compute_75`를 선택했다.
컴파일은 모듈 초기화 때 한 번 수행한다. NVRTC와 builtins DLL은 CMake가 실행 파일
옆에 복사한다. CPU 코드는 기존 MSVC로 빌드한다.

이 경로로 **사용자가 작성한 CUDA 커널의 실행도 확인**했다. 이후 사용자 PC에
MSVC 14.38.33130 설치를 확인했고, 이를 명시해 CUDA 12.0 NVCC로 두 커널을 PTX로
컴파일하는 데도 성공했다. NVCC 빌드에서는 14.38을 선택한다. 현재 진단 프로그램의
실행 경로는 NVRTC이며, 기존 C++ 빌드의 기본 툴셋은 14.44다. OptiX 모듈 자체의
컴파일/실행은 다음 단계에서 검증한다.

## 검증 방법

전송과 수학 계산을 다음 기준으로 각각 확인한다.

1. 모든 픽셀의 CUDA Depth와 DX12 Readback을 원본 float32 비트 단위로 비교한다.
2. 모든 픽셀에서 유한한 값, Depth [0,1], 배경의 zero/invalid 출력,
   Normal alpha와 Position Valid의 일치를 확인한다.
3. 모든 표면 픽셀의 GPU 위치를 View × Projection으로 다시 화면에 투영한다.
   원래 픽셀 중심과 Depth로 돌아와야 한다. 허용 오차는 화면 0.01픽셀,
   Depth 0.000002다. 이 검사는 inverse 복원식을 반복하지 않고 정방향 투영을 사용한다.
4. 격자 표본마다 Depth와 inverse VP를 사용하지 않는 CPU 카메라 광선을 만든다.
   원본 Scene 삼각형 전체와 double 정밀도 ray/triangle 교차를 계산하고,
   가장 가까운 hit 위치와 GPU 위치를 비교한다. 월드 거리 허용 오차는 0.002다.
5. 기하 표본이 바닥·빨간 큐브·파란 큐브·기울어진 평면을 모두 포함하는지 검사한다.

CPU 기준 광선은 inverse View의 카메라 축과 Projection의 x/y 스케일에서 구한다.
G-buffer Depth나 CUDA 출력으로 기준 교차점을 만들지 않는다. `Mesh`는 이를 위해
원본 CPU 정점/인덱스를 보관하고 getter를 제공한다. 이후 OptiX geometry 업로드에도
사용할 수 있는 데이터다.

삼각형 경계의 Raster coverage/top-left 규칙과 CPU 교차 판정은 차이가 있을 수 있다.
따라서 독립 기하 비교는 barycentric margin ≥0.025인 내부 표본만 사용한다.
전체 픽셀의 기하학적 최근접 hit를 증명한 것은 아니며, 전체 픽셀의 Depth 전송과
재투영 검사에 더해 내부 기하 표본을 독립 검증한 범위다.

## 실제 결과

두 해상도에서 각각 4프레임을 검사했다. 매 프레임 카메라 x 위치를 -0.45/+0.45로
바꾸고 near/far를 0.1/100과 0.25/40으로 번갈아 설정한다. Normal 이미지 변화도
확인하며, 모든 프레임에서 기존 DebugView 0~3을 순환한다.

| 항목 | 321 × 181 | 1280 × 720 |
|---|---:|---:|
| Depth 유효 행 바이트 / DX12 RowPitch | 1,284 / 1,536 | 5,120 / 5,120 |
| 한 프레임 전체 픽셀 | 58,101 | 921,600 |
| 표면 픽셀 수 (두 카메라) | 25,905 / 25,852 | 410,095 / 409,240 |
| 프레임당 CPU 내부 교차 표본 | 171 / 168 | 2,607 / 2,584 |
| 최대 화면 재투영 오차 | 0.00291614 px | 0.00641891 px |
| 최대 Depth 재투영 오차 | 2.85987e-7 | 1.93371e-7 |
| CPU 교차 대비 최대 월드 거리 오차 | 0.000212407 | 0.000221908 |

321×181 첫 프레임의 바닥 표본 (82,91):

```text
CUDA position = (-3.70775962, -0.999927461, 3.46498394)
CPU ray hit   = (-3.70784178, -1.000000000, 3.46513243)
distance error = 0.000184560
```

월드 단위가 미터라는 가정은 하지 않는다. float32 Depth·행렬·계산이므로 CPU 기하
기준과의 완전한 비트 일치는 요구하지 않는다. 반면 **공유 Depth 전송 자체는 모든
픽셀에서 비트 일치**했다. 기존 버퍼/Normal 테스트와 합쳐 CTest **5/5 통과**했고,
D3D12 debug layer warning/error는 없었다. 기존 렌더러의 Debug 빌드도 성공했다.

## 실행과 산출물

프로젝트 루트에서 실행한다.

```powershell
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
& $cmakeExe -S . -B build -DHYBRIDTRACER_BUILD_INTEROP_TEST=ON
& $cmakeExe --build build --config Debug
.\build\Debug\HybridTracerNormalInteropTest.exe --depth-position --frames 4 --samples
.\build\Debug\HybridTracerNormalInteropTest.exe --depth-position --frames 4 --full-resolution
& (Join-Path (Split-Path $cmakeExe) 'ctest.exe') --test-dir build -C Debug --output-on-failure
```

`--samples`는 실행한 현재 폴더에 `DepthPositionSamples.csv`와
`SurfaceNormalSamples.csv`를 쓴다. CTest에서는 `build` 폴더에 저장한다.
Position CSV는 프레임마다 네 SceneObject의 대표 표본을 하나씩 기록하며,
4프레임 기준 16행이다. Depth, 픽셀 좌표, GPU 위치, CPU 교차 위치와 거리 오차를 포함한다.

## 파일별 역할과 다음 작업

| 파일 | 역할 |
|---|---|
| RasterGBufferPass.h/.cpp | 선택적 Depth 공유 옵션; 기본값 false 유지 |
| CudaDepthPositionReader.h/.cpp | Depth import, NVRTC 컴파일, 커널 실행, Fence와 검증 Readback |
| ReconstructWorldPosition.cu | 픽셀 중심/Depth/inverse VP의 실제 GPU 복원 계산 |
| DepthPositionValidation.h/.cpp | 전체 Depth/재투영 검사 및 독립 CPU 최근접 삼각형 교차 |
| CudaNormalReader.cpp | Normal/Depth 리더를 함께 사용할 때 CUDA 컨텍스트 선택 |
| CudaNormalInteropTest.cpp | 기존 시나리오에 --depth-position 모드와 CSV 추가 |
| Mesh.h/.cpp | 원본 CPU 정점/인덱스 보관 |
| CMakeLists.txt | NVRTC 의존성·DLL/커널 복사·두 개의 Depth/Position CTest 등록 |

기존 메인 `HybridTracer.exe`는 계속 DX12 렌더러다. interop 옵션 기본값 OFF,
일반 Raster 초기화의 Normal/Depth 공유 기본값 false를 유지한다. CUDA 연결은 별도
검증 프로그램에서 실행하며 아직 여러 렌더 프레임을 겹쳐 처리하지 않는다.

후속 Surface 모드에서 Albedo까지 연결해 같은 픽셀의 Position/Normal/Albedo 입력을
완성했다. 다음은 OptiX용 geometry/GAS, pipeline/SBT/launch와 첫 2차 광선 효과다.
현재 GPU 입력 버퍼는 이 진단 모듈의 CUDA 컨텍스트에 속한다.

공식 참고:
[NVRTC 12.0.1](https://docs.nvidia.com/cuda/archive/12.0.1/nvrtc/index.html),
[CUDA 외부 리소스 API](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html),
[DirectXMath의 화면 좌표 역투영 규약](https://learn.microsoft.com/en-us/windows/win32/api/directxmath/nf-directxmath-xmvector3unproject).
