# 실제 Normal G-buffer → CUDA 읽기

2026-10-30, 공유 버퍼 왕복 테스트 다음으로 구현한 단계다.
`HybridTracerNormalInteropTest.exe`가 기존 `Scene`, `Camera`, `RasterGBufferPass`,
`Raster.hlsl`을 이용해 장면을 그린 뒤 **그 RasterPass가 생성한 Normal 텍스처 자체**를
CUDA에 연결한다. 별도의 가짜 Normal 배열이나 CPU 재업로드를 입력으로 사용하지 않는다.

후속으로 **Depth 공유와 CUDA 커널의 World Position 복원**도 완료했다.
[Depth/Position 검증 문서](DX12_CUDA_Depth_Position.md)에 좌표계 규약과 수치 결과를
정리했다. 아래 내용은 Normal 읽기 단계의 검증 범위다.

## 모듈별 입력과 출력

| 모듈 | 입력 | 출력과 역할 |
|---|---|---|
| RasterGBufferPass | 기존 Scene/Camera, shareNormalWithCuda=true | 공유 가능한 Normal RT2 생성·렌더링 |
| CudaNormalReader | DX12 장치, 실제 Normal ID3D12Resource | CUDA array 매핑, Fence 동기화, pinned CPU 메모리로 원본 half 비트 읽기 |
| CudaNormalInteropTest | 실제 Scene과 두 카메라 위치 | DX12/CUDA 전체 이미지 비교, 대표 픽셀·법선 검사, 선택적 CSV |
| InteropCommon.h | D3D12/CUDA API 결과 및 핸들 | LUID 일치 검사, 공유 Fence import, 오류 메시지, 리소스 전이 공통 함수 |

기존 버퍼 테스트도 `InteropCommon.h`를 사용하도록 정리했다. 기존 렌더러의
`RasterGBufferPass::Initialize()` 호출은 기본값 `shareNormalWithCuda=false`를 쓰므로
일반 실행에서 CUDA에 연결하거나 공유 메모리 옵션을 활성화하지 않는다.

## Normal의 의미와 메모리 표현

RT2는 `R16G16B16A16_FLOAT`이며, 한 픽셀은 8바이트다.

- RGB: **World Space Normal**의 x/y/z. 음수 성분도 그대로 저장한다.
- A: 현재 Raster.hlsl의 표면 픽셀은 1, Clear된 배경은 0.
- `(0, 0, 0, 0)`인 배경 법선은 단위 벡터로 검사하지 않는다.
- DebugView의 `normal * 0.5 + 0.5` 색 변환을 적용하지 않은 원본 데이터다.

Normal은 DX12에서 tiled texture로 관리하므로 CUDA 선형 포인터로 해석하지 않는다.
공유 committed resource를 external memory로 import한 후 다음 계약으로 매핑한다.

```text
DX12 format       R16G16B16A16_FLOAT
CUDA format       CU_AD_FORMAT_HALF, NumChannels=4
CUDA dimensions   Width=W, Height=H, Depth=0 (일반 2D)
CUDA flags        CUDA_ARRAY3D_COLOR_ATTACHMENT (DX12 Render Target)
Mip levels        1
```

`cuExternalMemoryGetMappedMipmappedArray()` → `cuMipmappedArrayGetLevel(0)`로
`CUarray`를 얻는다. `cuMemcpy2DAsync()`는 이 배열의 원본 비트를 pinned CPU
메모리로 옮긴다. **이 단계는 CUDA API를 통한 텍스처 읽기이며, 자체 CUDA 커널의
텍스처 샘플링이나 OptiX launch까지 검증한 것은 아니다.**

## 한 프레임의 흐름

```text
DX12 RasterPass: 실제 Scene → MRT, Normal RT2 쓰기
    ↓ Normal: RENDER_TARGET → COPY_SOURCE
DX12 CopyTextureRegion: Normal → DX12 Readback (비교 기준)
    ↓ Normal: COPY_SOURCE → COMMON
DX12 queue: Signal(rasterReady, n)
    ↓
CUDA stream: Wait(rasterReady, n)
CUDA stream: Normal CUarray → pinned CPU 메모리
CUDA stream: Signal(cudaReadDone, n)
    ↓
DX12 queue: Wait(cudaReadDone, n)
    ↓ Normal: COMMON → RENDER_TARGET
DX12 DebugViewPass: mode 0/1/2/3 순환
DX12 queue: Signal(completed, n)
    ↓
CPU: 완료 대기 → 두 경로의 모든 half 비트 비교 → 다음 프레임
```

두 공유 Fence는 각각 한 API에서만 Signal한다. CUDA 읽기 완료를 DX12 queue가
기다린 후에만 Normal을 다시 사용한다. API 사이에 CPU 대기를 넣지 않고,
마지막 수치 검사와 자원 재사용을 위해 프레임 끝에서만 CPU가 기다린다.

테스트는 한 프레임을 완료한 후 다음 프레임을 진행한다. 향후 실제 Application의
여러 프레임을 겹쳐 처리하는 스케줄은 별도로 설계해야 한다. 테스트의 RT0는 창의
SwapChain 대신 오프스크린 텍스처이며, Present나 키보드 조작은 검증하지 않는다.
실제 DebugView 셰이더의 SRV 읽기와 상태 전이 명령은 네 모드를 순환하며 실행한다.

## 왜 두 가지 해상도를 검사하는가

DX12의 텍스처 Readback은 행마다 패딩이 있을 수 있다. CUDA CPU 출력은 가로 픽셀
수 × 8바이트로 촘촘히 배치한다. DX12의 `GetCopyableFootprints()`가 알려 준
`RowPitch`로 각 행을 찾고, 유효한 픽셀 바이트만 비교해야 한다.

| 해상도 | 한 행의 픽셀 바이트 | DX12 RowPitch | 프레임 수 |
|---|---:|---:|---:|
| 321 × 181 | 2,568 | 2,816 | 8 |
| 1280 × 720 | 10,240 | 10,240 | 4 |

첫 해상도는 패딩 처리와 비정방형 좌표를, 두 번째는 현재 렌더러 해상도를 확인한다.
매 프레임 카메라 x 위치를 -0.45/+0.45로 번갈아 바꾸고 이미지가 달라지는지 검사한다.

## 수치 검사와 실제 결과

DX12와 CUDA가 읽은 **모든 픽셀의 RGBA 16비트 패턴이 정확히 일치**했다.
여기에 배경 0, 표면 alpha=1, 유한한 성분, 법선 길이 오차 ≤0.002를 검사한다.
DX12와 CUDA가 서로 일치한다는 사실만으로 렌더링 수학의 정확성이 증명되지는 않으므로
바닥 및 기울어진 평면의 예상 방향도 별도로 확인한다.

기울어진 평면의 CPU 기준값은 World 변환한 삼각형 꼭짓점의 두 변을 외적하고
정규화해 구한다. 셰이더의 inverse-transpose 계산식을 그대로 복제한 기준이 아니다.
현재 검증 Scene의 초록 Material을 가진 XZ 평면을 대상으로 하므로 Scene을 바꾸면
이 검증 조건도 함께 갱신해야 한다.

321×181 첫 프레임에서 관찰한 대표 픽셀:

| 분류 | 픽셀 (x,y) | CUDA와 DX12가 동일하게 읽은 원본 (x,y,z,a) |
|---|---|---|
| 배경 | (0,0) | (0, 0, 0, 0) |
| 바닥 | (211,83) | (0, 0.999511719, 0, 1) |
| 기울어진 평면 | (179,40) | (-0.357910156, 0.602050781, -0.713378906, 1) |
| 다른 기하 | (193,70) | (-0.229980469, 0.96875, -0.089355469, 1) |

기울어진 평면의 CPU 기하 기준은
`(-0.358045787, 0.602285624, -0.713481069)`이며 성분 오차 0.001 이내다.
이상적인 바닥 법선은 `(0,1,0)`이지만, 셰이더 연산과 half 저장을 거친 실제 값은
항상 정확히 1이라고 가정하지 않는다. 따라서 **전송 정확성은 비트 단위**, 기하
기준과 법선 길이는 **허용 오차**로 검사한다. 관찰한 최대 길이 오차는
0.00048828125였다. 일반 기하 픽셀의 방향을 개별 삼각형과 대조하는 검사는 아직 없다.

기존 버퍼 왕복 테스트와 새 Normal 두 해상도 테스트를 합쳐 CTest **3/3 통과**했고,
D3D12 debug layer의 warning/error는 없었다. 기본 래스터 렌더러도 Debug 빌드에 성공했다.

## 실행과 CSV

프로젝트 루트에서 기존 interop 옵션을 켜고 빌드한다.

```powershell
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
& $cmakeExe -S . -B build -DHYBRIDTRACER_BUILD_INTEROP_TEST=ON
& $cmakeExe --build build --config Debug
.\build\Debug\HybridTracerNormalInteropTest.exe --frames 8 --samples
.\build\Debug\HybridTracerNormalInteropTest.exe --frames 4 --full-resolution
& (Join-Path (Split-Path $cmakeExe) 'ctest.exe') --test-dir build -C Debug --output-on-failure
```

`--samples`는 **실행한 현재 폴더**에 `NormalInteropSamples.csv`를 쓴다. CTest로
실행하면 `build/NormalInteropSamples.csv`다. 매 프레임 배경·바닥·기울어진 평면·
다른 기하에서 하나씩 선택해 좌표, 두 경로의 float 해석값, 원본 half 비트를 저장한다.
half 비트 열은 16비트 패턴을 십진 정수로 표현한 값이며, 그 숫자 자체가 법선은 아니다.

## 다음 단계

Normal 읽기에 이어 Depth 공유와 CUDA World Position 복원을 검증했다.
다음 입력 확장은 Albedo이며, 이후 OptiX geometry/GAS와 Launch를 연결한다.
위치 커널은 NVRTC로 컴파일한다. 오프라인 NVCC–MSVC 툴셋 호환성과 OptiX 디바이스
코드의 빌드 방식은 별도로 결정해야 한다.

공유 텍스처 매핑과 해제 규약은
[CUDA 12.0.1 Driver API](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html),
[Direct3D 12 interop 가이드](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/index.html#direct3d-12-interoperability)를
참고했다.
