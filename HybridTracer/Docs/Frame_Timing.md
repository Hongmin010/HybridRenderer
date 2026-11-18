# GPU 패스 시간 계측

2026-11-05. 기존 다섯 패스에 선택적 계측을 연결했다. 후속 TemporalAccumulationPass를 더해 현재는 여섯 패스다. 렌더 알고리즘/샘플 수는
계측 활성화 여부로 바뀌지 않는다. 기본 실행은 계측 OFF다.

## 실행

메인 창의 제목에 최근 완료 프레임의 GPU span/OptiX 시간을 표시한다. 종료 시
평균/중앙값/p95를 출력하고 현재 폴더에 `HybridTracerTimings.csv`를 저장한다.

```powershell
& 'C:\GitHub\HybridRenderer\HybridTracer\build\Debug\HybridTracer.exe' --profile
```

반복 가능한 기준 측정은 같은 production pipeline의 headless benchmark를 사용한다.
초기화/NVRTC/GAS와 워밍업을 측정 표본에서 제외하고, Present/vsync와 이미지 readback도 없다.

```powershell
Set-Location -LiteralPath 'C:\GitHub\HybridRenderer\HybridTracer\build'
.\Release\HybridTracerBenchmark.exe --warmup 60 --frames 180 --spp 4 --csv Baseline_4spp.csv
```

옵션: `--width`, `--height`(기본 1280/720), `--spp`(4), `--frames`(120), `--warmup`(30),
`--csv`, `--moving-camera`, `--check`(D3D debug), `--no-profile`(GPU 계측 없이 CPU wall 비교), `--no-accumulation`(누적 OFF).
`--no-profile`에서는 CSV를 생성하지 않는다. Windows Unicode 경로는 wmain/filesystem으로 받는다.
benchmark는 프레임마다 이미지를 readback하지 않고 완료 fence 뒤에 timestamp 56 bytes만 읽는다.

## 수치의 의미

| CSV 열 | 구간/해석 |
|---|---|
| raster_ms | DX12 Raster pass. 이후 COMMON 전환은 제외 |
| surface_input_ms | CUDA raster-ready wait 이후부터 SurfaceInput 완성까지 |
| surface_copies_ms | Depth/Normal/Albedo의 GPU 내부 복사와 작은 matrix upload |
| surface_kernels_ms | 위치 복원 + SurfaceInput packing kernel event 구간 |
| accumulation_ms | CUDA indirect mean kernel 구간. OFF에서는 인접 event 구간 |
| cpu_accumulation_submit_ms | 누적 Execute API wall. cpu_submit의 일부 |
| optix_trace_ms | parameter upload 뒤 OptiX launch를 감싼 CUDA event 구간 |
| shadow_copy_ms / indirect_copy_ms | 두 private output → DX12 공유 buffer GPU copy |
| composite_ms / tonemap_ms | DX12 합성/표시 색 변환 |
| debug_view_ms | 선택적 raw debug. Lit에서는 인접 timestamp 간 작은 계측 간격 |
| cuda_raster_wait_ms | CUDA stream에서 raster-ready semaphore wait를 감싼 event 구간 |
| dx_handoff_gap_ms | Raster의 COMMON 전환 뒤부터 Composite 직전까지 DX12 timestamp 간격 |
| gpu_span_ms | Raster 시작부터 ToneMap/optional Debug 종료까지 DX12 timestamp 간격 |
| cpu_submit_ms | 프레임 기록/제출 API들의 CPU wall. CPU/driver 대기가 포함됨 |
| cpu_end_frame_ms | 메인의 EndFrame API wall. command submit, Present/vsync와 fence 처리 포함 |
| cpu_reuse_wait_ms | RecordRaster의 기존 이전 프레임 완료 대기 API wall. cpu_submit의 일부 |
| cpu_completion_wait_ms | headless benchmark에서 마지막 제출 뒤 GPU 완료 fence를 기다리는 wall |
| cpu_surface_submit_ms | SurfaceInput Execute API wall. cpu_submit의 일부 |
| cpu_trace_submit_ms | OptiX LaunchIndirect API wall. cpu_submit의 일부 |
| cpu_publish_submit_ms | 두 공유 output Publish API wall. cpu_submit의 일부 |

`surface_input = surface_copies + surface_kernels`는 같은 CUDA stream의 연속 구간이다.
`dx_handoff_gap`에는 CUDA 계산·복사·fence·host 제출 간격이 **함께** 포함된다.
이를 순수 대기로 부르거나 CUDA 시간과 다시 더하면 중복 계산이 된다.
CPU submit 역시 실제 CPU 연산 시간만을 뜻하지 않는다. driver/검증/API 내부 대기도 들어간다.

DX12는 queue별 timestamp frequency로 UINT64 tick 차이를 ms로 변환한다.
CUDA는 같은 비기본 stream의 timing-enabled event와 cuEventElapsedTime을 쓴다.
두 API의 원시 timestamp를 빼거나 하나의 공통 clock으로 취급하지 않는다.
CUDA event 구간은 다른 GPU 작업/스케줄링/host 공급 간격에 영향을 받을 수 있으므로
순수 kernel instruction 실행 시간으로 단정하지 않는다. 정밀한 원인은 profiler trace가 필요하다.

`gpu_span`은 Present/vsync, query resolve, CPU completion wait를 제외한다. windowed CSV의
첫 프레임은 cold launch를 포함할 수 있고, debug view를 바꾼 세션의 summary는 여러 view를
섞는다. 성능 비교에는 고정 조건의 benchmark CSV를 사용한다. timestamp/CSV 회수는
프레임 GPU 완료 뒤에 하며 이를 위한 새 cuEventSynchronize/cuStreamSynchronize는 추가하지 않았다.

## 구현과 검증

- `Renderer/FrameProfiler.*`: DX12 query heap/readback, CUDA events, 완료 결과 회수/통계/CSV.
- `Interop/CudaTimingEvents.h`: 선택적 CUDA 계측 hook. OFF면 event 생성/기록을 하지 않는다.
- `HybridFramePipeline`: 기존 single-frame 자원 보호 뒤 결과 회수, 다음 frame에서 event 재사용.
- `Validation/FrameBenchmark.cpp`: 이미지 readback 없는 반복 실행과 워밍업 제외.
- `Application --profile`: 같은 메인 경로의 제목/CSV와 마지막 frame 회수.

기존 15개 검사에 `HybridCompositeProfiled`, `HybridTimingBenchmark`를 더해 현재 총 20개다. history 관련 검사는 Temporal_Accumulation.md에 있다.
profiled 통합 검사는 10프레임 전체 픽셀의 HDR/display/shared outputs와 8개 view를 검사하고,
timestamp의 frame/spp/view metadata, 유한/양수 값, SurfaceInput 구간 합과 DX12 구간 순서를 확인한다.
benchmark smoke는 워밍업 2개를 제외하고 정확히 4개 frame을 기록하며 D3D debug를 검사한다.
별도로 원본 `--smoke-test --profile`에서 창/Present/카메라 이동과 마지막 frame 회수를 검사한다.
CUDA/OptiX OFF DX12-only 빌드도 유지한다.

## 재현 조건과 근거

첫 기준 측정은 MX450 / driver 560.94 / CUDA Toolkit 12.0 / OptiX 8.0,
Release CPU 코드 / **OptiX validation ALL 유지** / D3D debug OFF, 정적 기본 장면이다.
카메라는 (-0.45,2.6,-7)에서 (0,0,0.5)를 본다. random seed=12345+frame이다.
전력/클럭을 고정하는 OS 설정은 변경하지 않았고 GPU clock/열/다른 작업에 따른 변동은 남는다.
평균 외에 중앙값과 nearest-rank p95를 함께 기록한다. GPU span의 역수를 실제 메인 FPS로 쓰지 않는다.
계측 ON/OFF의 전체 measured-loop wall(결과 회수/CSV 포함)도 비교한다.

측정/실패 시 중단 조건은 finite/metadata/validation 오류이며, 특정 GPU 시간 임계값을
테스트에 넣지 않는다. 알려진 OptiX launcher allocation notice는 이전과 같이 별도 기록한다.

API 근거: [Microsoft DX12 timing](https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing),
[CUDA 12.0 Driver event API](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-driver-api/group__CUDA__EVENT.html).
실측 CSV/해석/검증 로그는 workspace의 `verification-timing`에 보존한다.

## 첫 실측 결과

4 spp 첫 run의 중앙값: Raster 0.116 ms, SurfaceInput 7.547 ms, OptiX 23.446 ms,
두 공유 copy 합 약 0.85 ms, Composite 0.274 ms, ToneMap 0.05 ms, GPU span 33.531 ms.
1/4/8 spp trace 중앙값은 6.585/23.446/45.074 ms로 증가했다.
동일 4 spp 재측정에서는 clock 상태 변화와 함께 GPU span 48.819 ms, trace 33.245 ms가
관측됐다. 전/후 snapshot에서 memory clock 5000→3000 MHz가 기록됐다. clock을 고정하지
않았으므로 수치는 범위와 조건을 함께 해석한다. 두 상태 모두 trace가 가장 큰 구간이었다.
상세 표와 CSV 근거는 workspace verification-timing/Baseline_Report.md에 있다.

## 누적 추가 후 CSV/비교

현재 metadata는 frame/width/height/spp/view에 accumulation_enabled/history_frames/history_samples/
history_reset_mask를 더한 9개다. 시간 열은 22개로 총 31 columns다. CUDA events는 14개다.
benchmark 기본은 누적 ON이며 warmup 동안도 history가 쌓인다. 아래 첫 기준 실측은 누적을
추가하기 전 기록이므로 재현 비교에는 --no-accumulation을 명시한다. 현재 누적 비용은
workspace verification-temporal의 ON/OFF benchmark CSV와 기록을 참고한다.
