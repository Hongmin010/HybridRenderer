# 간접광 프레임 누적

검증한 1회 반사 indirect radiance를 같은 화면 픽셀에서 시간 평균한다.
메인은 기본 ON, 4 spp/frame이며 camera/light가 유지되면 표본이 계속 쌓인다.

```text
Raster → SurfaceInput → OptiX Trace → TemporalAccumulation → Composite → ToneMap → Present
```

## 계산과 자원

각 픽셀은 float RGB 평균/Valid 16 bytes와 uint count 4 bytes를 가진다.
매 프레임의 선형 HDR 간접광 x에 대해 `mean += (x - mean) / count`를 적용한다.
같은 epoch 안에서는 spp가 고정되므로 프레임 평균은 동일 가중치의 sample 평균이다.
직접광은 현재 프레임에서 계산하고, ToneMap/sRGB는 누적 후에 적용한다.
albedo/cosine/PDF를 누적 단계에서 다시 곱하지 않는다. 프레임 전체 합을 저장하지 않는다.

배경은 RGB/Valid/count=0으로 쓴다. 검정 표면의 유효한 0 radiance는 표본으로 센다.
픽셀의 Valid가 끊기면 해당 count를 0으로 돌려 새 유효 표본부터 평균한다.
현재 정적 장면/고정 카메라에서는 foreground의 count가 전체 history frame 수와 같다.
CPU는 전체 epoch의 frame 수와 최대 nominal spp를 표시하며 이미지 count를 읽지 않는다.

1280×720의 추가 history 메모리는 20 bytes/pixel, 약 18.4 MB다.
기존 CUDA context/stream을 빌리고 Trace → 누적 kernel → 공유 indirect copy 순서로 실행한다.
기존 출력 완료 fence가 history까지 보호한다. 프레임 자원은 현재처럼 한 세트다.
메인은 픽셀 readback이나 누적 전용 CPU/GPU 완료 대기를 추가하지 않는다.

## 초기화 조건

| reset mask | 이유 |
|---:|---|
| 1 | 첫 프레임 |
| 2 | View 또는 Projection 행렬 변경 |
| 4 | 정규화 광원 방향, linear RGB 또는 Intensity 변경 |
| 8 | 프레임당 spp 변경 |
| 16 | 수동 ResetAccumulation / R |
| 32 | 누적 ON/OFF 전환 |
| 64 | 2^20 history frames에 도달해 새 epoch 시작 |

이유는 bitmask로 합칠 수 있다. 첫/초기화 프레임은 과거 memory를 읽지 않고 현재 raw
radiance를 그대로 써서 stale history를 제거한다. 리소스를 재생성하면 첫 프레임에서 시작한다.
View/Projection 32 floats와 광원 7 floats를 정확히 비교하므로 아주 작은 변화도 초기화한다.
보기 모드, 노출, random seed 변경은 history를 유지한다. 기본 seed는 프레임마다 달라진다.
진단에서 명시적으로 seed를 고정하면 같은 표본을 반복하므로 nominal spp만 늘고 노이즈는 줄지 않는다.

OFF에서는 CUDA 누적 kernel을 생략하고 raw 추정치를 공유 buffer로 보낸다. GPU history 할당은
유지하며 다시 ON하면 첫 표본부터 시작한다. per-pixel count 진단 API는 ON 결과에만 유효하다.
history는 float32, per-pixel count는 uint32다. 장시간 float 평균과 count의 한계를 제한하기
위해 2^20 frames마다 초기화한다. spp≤4096이며 nominal total spp는 uint64로 표시한다.

## 조작과 API

```powershell
& 'C:\GitHub\HybridRenderer\HybridTracer\build\Debug\HybridTracer.exe'
& 'C:\GitHub\HybridRenderer\HybridTracer\build\Debug\HybridTracer.exe' --profile
& 'C:\GitHub\HybridRenderer\HybridTracer\build\Debug\HybridTracer.exe' --no-accumulation
```

기존 1~8 보기를 유지한다. 8=간접광만, T=누적 ON/OFF, R=수동 초기화다.
창 제목의 `Accum ON Nf / S spp`에서 진행 상황을 본다. 카메라를 멈추면 N이 증가한다.
7=직접광만 보기에서도 indirect는 계속 계산/누적한다. 현재 광선 경로의 실행 범위는 동일하다.

`HybridFramePipeline::SetAccumulationEnabled`, `ResetAccumulation`, `GetAccumulationInfo`로
제어한다. `GetDeviceIndirect()`는 현재 DX12에 전달한 결과이고 `GetDeviceRawIndirect()`는
이 프레임의 원본 OptiX 추정치다. `TemporalAccumulationPass`가 GPU mean/count를 소유한다.

benchmark 기본도 ON이다. 이전 raw 기준과 비교할 때 `--no-accumulation`을 명시한다.
warmup 동안에도 history가 쌓이므로 warmup=60이면 첫 측정 row의 history_frames는 61이다.
계측에는 `accumulation_ms`, `cpu_accumulation_submit_ms`와 enabled/frames/samples/reset mask를
추가했다. 현재 CSV는 31 columns, CUDA timing events는 14개다. 누적 OFF의 event 구간은
인접 event 기록 비용을 포함할 수 있으므로 이를 kernel 실행으로 해석하지 않는다.

## 검증과 실제 비교

기존 17개 검사에 수학/전체 프레임/움직이는 카메라 검사를 더해 CTest는 20개다.

- CudaTemporalAccumulation: 17×3, 140 frames. 독립 double sum/count, HDR, 검정/배경,
  Valid 전환, camera/light/spp/manual/toggle 초기화와 raw bypass를 검사한다.
- HybridTemporalAccumulation: 실제 pipeline의 raw GPU 출력 64개를 CPU에서 따로 평균한다.
  공유 buffer byte 일치, 누적 HDR/최종 sRGB, profiling metadata와 초기화를 검사한다.
- HybridTemporalMovingBenchmark: 카메라가 매번 바뀔 때 history가 항상 1인지 확인한다.
- 메인 hidden smoke: 12 frames에서 정지 중 1→4 증가, 이동/Projection 변경, T에 해당하는
  ON/OFF, R에 해당하는 수동 초기화와 8개 보기를 검사한다.

별도 Release 1280×720의 64 frames × 4 spp를 독립 seed의 256 frames × 16 spp
(4096 spp) 기준 추정치와 비교했다. 이는 유한 표본 Monte Carlo reference이며 정답 해석값은 아니다.

| 항목 | 선형 RGB 결과 |
|---|---:|
| 한 프레임 4 spp의 MSE | 4.11208e-4 |
| 64프레임/256 spp의 MSE | 6.81797e-6 |
| 누적 RMSE / 한 프레임 RMSE | 0.128765 |
| GPU 평균과 CPU 별도 평균의 최대 차이 | 8.69235e-8 |

오차는 배경을 제외한 foreground의 RGB에서 구한다. 표시 이미지의 gamma/클리핑을
정량 오차 계산에 사용하지 않는다. 실제 BMP/PNG, quality CSV, frame timing, 검증 로그는
workspace verification-temporal에 보존한다. PNG는 GPU BMP의 무손실 포맷 변환이다.

## 현재 범위

카메라가 움직이는 동안은 매번 초기화되어 한 프레임의 노이즈가 보인다. 멈추면 평균이
진행된다. 재투영/motion vector/denoise/subpixel anti-aliasing은 아직 없다. 현재 장면은
정적 GAS/material snapshot이며 동적 물체/재질 변경을 자동으로 추적하는 기능은 없다.
씬 리소스를 갱신하는 후속 구현은 GAS/material 갱신과 history 무효화를 함께 처리해야 한다.
다음 렌더 확장은 다중 bounce와 그에 따른 throughput/종료 조건 검증이다.

표본을 늘려 Monte Carlo 오차를 줄이는 배경은
[PBRT의 Improving Efficiency](https://pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency)를
참고했다. 구현과 수치 검사는 이 프로젝트의 GPU radiance 계약에 맞췄다.

## 누적 패스 비용

동일 1280×720/4 spp Release의 30 warmup + 120 measured frames에서 accumulation event
중앙값은 ON 0.7457 ms, OFF의 인접 event 간격은 0.0038 ms였다.
전체 GPU span 중앙값은 ON 34.2539 / OFF 33.5611 ms였다.
clock을 고정하지 않았으므로 전체 차이를 정확한 kernel 비용으로 단정하지 않는다.
자세한 조건/CSV는 workspace verification-temporal/Quality_And_Cost.md에 있다.
