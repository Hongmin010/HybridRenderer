# DX12–CUDA 공유 버퍼 검증

2026-10-30에 MX450에서 검증한 첫 interop 단계다. 실행 파일은
`HybridTracerInteropTest.exe`이며, 기존 `HybridTracer.exe`와 별도로 실행한다.

후속 단계인 **실제 Normal G-buffer의 CUDA 읽기**도 구현·검증했다.
자세한 흐름과 원본 수치는 [Normal 텍스처 검증 문서](DX12_CUDA_Normal_Interop.md)를
참고한다. 아래 내용은 첫 번째 공유 버퍼 테스트의 범위를 설명한다.

## 이번에 확인하는 것

DX12와 CUDA가 **같은 GPU 메모리**를 읽고 쓰며, 공유 Fence로 접근 순서를
지킬 수 있는지 검사한다. 실제 G-buffer 공유, CUDA 자체 커널, OptiX 추적은
아직 포함하지 않는다.

기존 렌더러의 `DX12Context::CreateDevice()`를 그대로 호출한다. 따라서 테스트만
별도의 GPU를 선택해 성공하는 경우를 피한다. DX12 장치의 LUID와
`cuDeviceGetLuid()`가 반환한 CUDA 장치의 LUID를 바이트 단위로 대조하고,
일치하지 않으면 실패한다. 현재 구현은 단일 노드 어댑터를 대상으로 한다.

## 빌드와 실행

프로젝트 루트에서 PowerShell로 실행한다.

```powershell
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
& $cmakeExe -S . -B build -G 'Visual Studio 17 2022' -A x64 -DHYBRIDTRACER_BUILD_INTEROP_TEST=ON
& $cmakeExe --build build --config Debug
.\build\Debug\HybridTracerInteropTest.exe --iterations 256
```

등록된 테스트를 실행하려면 같은 CMake 설치 폴더의 `ctest.exe`를 사용한다.

```powershell
$ctestExe = Join-Path (Split-Path $cmakeExe) 'ctest.exe'
& $ctestExe --test-dir build -C Debug --output-on-failure
```

interop 빌드 옵션의 기본값은 `OFF`다. 옵션을 끄면 기존 래스터 렌더러만 빌드하며
CUDA Toolkit 탐색이나 CUDA 링크를 추가하지 않는다. CMake 캐시는 옵션 값을
유지하므로 기존 빌드 폴더에서 끄려면 `-DHYBRIDTRACER_BUILD_INTEROP_TEST=OFF`를
명시한다.

## CUDA 12.0과 MSVC 14.44

이 테스트는 일반 C++에서 CUDA **Driver API**(`cuda.h`, `CUDA::cuda_driver`)를
호출한다. `.cu` 파일이나 NVCC를 사용하지 않으므로 현재 설치된 MSVC로 빌드된다.

이것은 NVCC 호환성 문제가 해결됐다는 뜻은 아니다. 로컬 CUDA 12.0의
`include/crt/host_config.h`는 `_MSC_VER >= 1940`을 거부하며, 설치된 컴파일러는
MSVC 19.44다. 이후 CUDA/OptiX 디바이스 코드를 NVCC로 빌드할 때는 호환 가능한
툴셋 선택을 별도로 해결해야 한다. 이번 검증에 `-allow-unsupported-compiler`는
사용하지 않았다.

## 입력과 출력

공유 리소스는 `uint32_t[512]`, 즉 2,048바이트 버퍼 하나다. Windows/DX12의 실제
메모리 할당 크기는 이 PC에서 65,536바이트다. CUDA import에는
`GetResourceAllocationInfo()`가 반환한 **할당 크기**를 전달하고, CUDA 포인터로
매핑하는 범위는 버퍼의 2,048바이트다.

| 단계 | 수행 주체 | 데이터 |
|---|---|---|
| 입력 생성 | CPU | `0x12340000 ^ (iteration * 0x00010001) ^ (index * 17)` |
| 공유 버퍼 쓰기 | DX12 CopyBufferRegion | Upload Buffer → 공유 버퍼의 512개 원소 |
| 입력 관찰 | CUDA cuMemcpyDtoHAsync | 공유 버퍼 → pinned CPU 메모리, 512개 원소 |
| 공유 버퍼 수정 | CUDA cuMemsetD32Async | 뒤쪽 256개 원소를 `0xca000000 \| iteration`으로 변경 |
| 반환값 관찰 | DX12 CopyBufferRegion | 공유 버퍼 → Readback Buffer, 512개 원소 |
| 값 비교 | CPU | CUDA가 읽은 입력, DX12가 읽은 보존된 앞부분과 수정된 뒷부분 검사 |

반복 번호는 1부터 시작한다. 매회 입력과 출력 패턴이 달라지므로 이전 반복의
값이 그대로 남는 오류도 검출한다. GPU 포인터를 CPU에서 직접 역참조하지 않는다.
CPU 전송은 검증용 관찰 경로이며, CUDA 수정값을 DX12로 돌려줄 때 CPU를 통한
재업로드는 하지 않는다.

## 동기화 흐름

```text
DX12 queue                         CUDA stream
  CopyBufferRegion(upload → shared)
  shared: COPY_DEST → COMMON
  Signal(dxReady, n)  ────────────→ Wait(dxReady, n)
                                   cuMemcpyDtoHAsync(input 관찰)
                                   cuMemsetD32Async(뒤쪽 수정)
  Wait(cudaReady, n)  ←──────────── Signal(cudaReady, n)
  shared: COMMON → COPY_SOURCE
  CopyBufferRegion(shared → readback)
  shared: COPY_SOURCE → COMMON
  Signal(readbackDone, n)
        ↓
  CPU: 완료 대기 → 값 비교 → 다음 반복
```

| Fence | 생성 옵션 | Signal 주체 | Wait 주체 |
|---|---|---|---|
| dxReady | D3D12_FENCE_FLAG_SHARED | DX12 queue | CUDA stream |
| cudaReady | D3D12_FENCE_FLAG_SHARED | CUDA stream | DX12 queue |
| readbackDone | D3D12_FENCE_FLAG_NONE | DX12 queue | CPU |

두 API 사이에는 CPU 대기를 넣지 않는다. 각 공유 Fence는 Signal 주체를 하나로
제한하고 반복 번호를 단조 증가시키므로 값이 되돌아가는 문제를 피한다.
상대 API의 Signal 호출을 먼저 제출한 후 해당 값의 Wait를 제출한다.

CPU는 반환값을 읽을 때만 완료를 기다린다. 이 대기는 다음 반복에서 Upload
Buffer, 명령 할당자, 공유 버퍼를 안전하게 재사용하도록 보호한다. 현재 테스트는
한 버퍼의 순차 왕복이며, 렌더러의 여러 프레임을 겹쳐 실행하는 interop 스케줄은
검증하지 않는다.

리소스 상태 전이와 API 간 동기화는 역할이 다르다. DX12는 공유 버퍼를
`COMMON → COPY_DEST → COMMON`, `COMMON → COPY_SOURCE → COMMON`으로 전이하며,
CUDA에 넘기는 시점에는 `COMMON`이다. Fence는 다른 API의 쓰기가 끝날 때까지
기다리게 한다. 상태 전이만으로 CUDA 완료를 기다릴 수는 없다.

## 리소스 공유와 해제

공유 버퍼는 DEFAULT heap의 committed resource를 `D3D12_HEAP_FLAG_SHARED`로
생성한다. `CreateSharedHandle()`의 NT 핸들을 CUDA에
`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE`로 import하고,
`CUDA_EXTERNAL_MEMORY_DEDICATED`를 설정한다.

Fence도 공유 NT 핸들로 내보내고 `CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE`로
import한다. NT 핸들은 import 후 닫는다. 종료 시 CUDA stream 완료를 기다린 뒤
CUDA 매핑 포인터를 `cuMemFree()`로 해제하고 external memory/semaphore와 CUDA
context를 해제한다. DX12 리소스는 CUDA 참조를 해제한 뒤 파괴된다.

## 파일별 역할

| 파일 | 변경 이유 |
|---|---|
| CMakeLists.txt | 선택적 테스트 타깃과 CTest 등록; CUDA language는 활성화하지 않음 |
| Source/Renderer/DX12Context.h/.cpp | 창 없이 같은 GPU/queue를 생성하는 InitializeHeadless와 queue getter 제공 |
| Source/Interop/CudaInteropTest.cpp | GPU 일치, 공유 리소스, Fence 왕복, 원본 수치 검사 |

기존 `Initialize()`도 `InitializeHeadless()`로 장치/명령 객체를 만든 뒤 기존의
SwapChain/RTV 초기화를 수행한다. 기존 Application과 G-buffer/DebugView 셰이더는
이번 변경에서 수정하지 않았다.

## 로컬 검증 결과와 한계

- 환경: NVIDIA GeForce MX450 2 GiB, 드라이버 560.94, CUDA Toolkit 12.0.76,
  MSVC 19.44.35229, Windows SDK 10.0.26100.0.
- 기존 렌더러와 interop 테스트의 Debug 빌드 성공.
- DX12/CUDA LUID가 둘 다 `00000000:00013e68`로 일치. LUID는 이번 실행의 값이며
  다음 부팅에서도 같은 상수라고 가정하지 않는다.
- 256회 왕복 성공. 매회 입력 512개와 반환 512개를 정수값으로 전부 비교.
- 1회차: CUDA 입력 `[0] = 0x12350001`, DX12 반환 `[256] = 0xca000001`.
- 256회차: CUDA 입력 `[0] = 0x13340100`, DX12 반환 `[256] = 0xca000100`.
- D3D12 debug layer 활성화 상태에서 warning/error 없음. debug layer가 없는
  환경에서도 데이터 검사는 실행하지만 해당 검증 부재를 콘솔에 표시한다.
- 기존 1~4번 화면 전환은 이번 테스트로 재검증한 항목이 아니다.

후속 Normal 테스트에서는 실제 G-buffer 텍스처를 CUDA로 읽고 특정 픽셀과 전체
이미지의 원본 수치를 검증했다. 다음 입력 확장은 Depth와 World Position 복원이다.
OptiX GAS/Pipeline/SBT/Launch 및 여러 렌더 프레임의 interop 동기화는 별도 작업으로
남아 있다.

## 공식 참고 문서

- [CUDA 12.0.1 Programming Guide — Direct3D 12 Interoperability](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-c-programming-guide/index.html#direct3d-12-interoperability)
- [CUDA 12.0.1 Driver API — External Resource Interoperability](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html)
- [CUDA 12.0.1 Driver API — Device Management / cuDeviceGetLuid](https://docs.nvidia.com/cuda/archive/12.0.1/cuda-driver-api/group__CUDA__DEVICE.html)
- [Microsoft — D3D12_HEAP_FLAGS](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ne-d3d12-d3d12_heap_flags)
