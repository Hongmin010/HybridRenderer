# CUDA Diffuse BSDF: 샘플링·PDF·가중치와 정량 검증

후속 [SurfaceHit](Optix_Surface_Hit.md)과 [1회 반사 간접광](One_Bounce_Indirect.md)을
메인에 연결했다. 현재는 direct + hard shadow + 실제 diffuse indirect를 표시하고 상수
Ambient는 제거했다. 전체 CTest는 15개다. 아래는 BSDF 모듈 자체의 구현/검증 기록이다.

2026-11-03. 사용자 제공 Mitsuba 세미나의 tutorial3_BSDF와 hw3_sub2의 수식/검증 방법을
바탕으로 공용 CUDA device 함수를 구현했다. 다섯 렌더 패스는 유지하며 OptiX에서 재사용한다.

## 소스와 계약

| 파일 | 역할 |
|---|---|
| Shaders/DiffuseBSDFTypes.h | host/device 공통 Vector3 및 32-byte DiffuseSample |
| Shaders/DiffuseBSDF.cuh | 공용 CUDA SampleDiffuse / EvalDiffuseBRDF / PdfDiffuse |
| Shaders/DiffuseBSDFProbe.cu | 공용 함수를 호출해 기여값을 계산하는 진단 커널 |
| Shaders/DiffuseBSDFProbeData.h | 진단 input/output ABI, 정규 렌더 출력과 분리 |
| Source/Validation/DiffuseBSDFTest.cpp | CUDA Driver/NVRTC 실행, 독립적인 CPU 수식·분포 검사 |

함수 의미:

```text
EvalDiffuseBRDF(albedo, N, incoming, outgoing)
    -> raw albedo/pi

PdfDiffuse(N, incoming, outgoing)
    -> max(dot(normalize(N), incoming),0)/pi  [solid-angle density]

SampleDiffuse(albedo, N, outgoing, u1, u2)
    -> world direction / solid-angle PDF / BRDF*cos/PDF weight / Valid
```

벡터는 모두 표면에서 바깥으로 향한다. incoming은 앞으로 빛을 받아올 방향이며,
outgoing은 카메라 또는 이전 경로 정점 쪽 방향이다. N과 두 방향은 finite unit vector를
요구하며 length-squared 오차 1e-4 이내를 허용한다. 유효 N은 내부에서 다시 정규화한다.
같은 정상 쪽 반구의 반사만 지원한다. backside/horizon은 0이며 투과나 양면 BSDF는 없다.

Albedo는 finite linear RGB reflectance [0,1]이다. Sample의 검은 재질도 valid지만 Weight=0이다.
호출자가 독립적인 uniform u1/u2를 [0,1)에서 공급한다. 이 함수는 난수 생성기 자체가
아니다. 테스트에서는 고정 seed의 CPU mt19937 상위 24bit를 입력으로 공급하고, 방향
변환·BRDF/PDF·기여값 계산은 CUDA에서 수행한다. GI 단계의 pixel/frame/bounce별 GPU
난수 생성은 별도로 연결해야 한다.

SampleDiffuse는 한쪽 반구에서 다음 분포를 사용한다.

```text
r=sqrt(u1), phi=2*pi*u2
local direction = (r*cos(phi), r*sin(phi), sqrt(1-u1))
world direction = normalize(T*x+B*y+N*z)
pdf_omega = cos(theta)/pi
weight = (albedo/pi)*cos(theta)/(cos(theta)/pi) = albedo
```

T/B는 N과 직교하는 frame이다. 고정 축과 거의 평행한 정상에서도 안정적으로 구성할
수 있도록 보조 축을 선택한다. 이 weight=albedo 단순화는 **Lambert diffuse와 이 sampling
분포의 조합에만** 해당한다. 다른 재질/분포에 그대로 쓰지 않는다.

Sample.Weight에는 이미 BRDF*cos/PDF가 포함된다. 다음 경로의 throughput은 beta *= Weight다.
EvalDiffuseBRDF는 raw BRDF를 반환하며, Mitsuba eval()처럼 cosine을 포함하는 계약과
다르다. 이 차이를 이름과 주석에 명시했다. 여기에 cosine/PDF를 다시 적용하지 않는다.
유효하지 않은 Sample 입력은 모든 필드가 0인 record를 반환한다. Pdf는 재질 독립적인
sampling density이므로 Albedo 인자는 받지 않는다.

## 상수 radiance 검사

가려지지 않은 고립된 diffuse 표면에 정상 반구 전체에서 일정한 incoming radiance L이
들어오면 선형 outgoing radiance는 albedo*L이다. 이 단계는 환경 조명의 적분 검사이며
다른 물체를 거친 GI 검증과 구분한다. 기존 Ambient를 더하는 방식으로 검사하지 않는다.

- gray albedo=(0.8,0.8,0.8), environment=(5,5,5) → (4,4,4).
- colored albedo=(0.85,0.1,0.08), environment=(2,0.5,3) → (1.7,0.05,0.24).
- albedo=(0,1,0.25), environment=(0.0001,5,100) → (0,5,25).

진단 커널은 Sample.Weight 방식과 독립적인 Eval*cos/PDF 방식을 모두 계산한다. 출력에
ToneMap, gamma, [0,1] clipping을 적용하지 않는다. CPU에서는 내려받은 sample contribution을
double로 평균한다. 상수 환경 + cosine diffuse에서는 이론적으로 샘플링 분산이 0이며
부동소수점 반올림 오차만 남는다. 일반 GI도 무노이즈라는 의미는 아니다.

## 상수 가중치만 확인하지 않는 이유

weight를 albedo로 반환하는 구현은 상수 환경 검사만으로 방향 분포의 오류를 검출할 수
없다. 그래서 다음 독립 기준도 함께 검사한다.

1. 모든 GPU sample의 unit length / 정상 반구 / finite / PDF / raw BRDF / weight를 확인.
2. CPU double dot(N, direction)/pi로 PDF를 비교. CPU가 device 샘플러의 방향 생성식을
   그대로 재구현해 기준으로 삼지 않는다.
3. E[direction]=(2/3)N, E[direction_i^2]=(1+N_i^2)/4의 해석적 moment를 확인.
4. cos(theta) 8구간 × azimuth 8구간의 joint histogram을 확인. 각 bin 기대값은
   cos(theta)의 CDF=t^2와 균일 azimuth에서 구한다. CPU azimuth frame은 다른 축과
   Gram-Schmidt 방식으로 만들어 device frame 구현을 기준값으로 재사용하지 않는다.
5. world-space 방향에 따라 밝기가 달라지는 비상수 환경도 적분한다.

```text
L_r(w)=2+0.5*w.x       -> expected outgoing rho.r*(2+N.x/3)
L_g(w)=1+0.3*w.y       -> expected outgoing rho.g*(1+0.2*N.y)
L_b(w)=0.5+0.25*w.z^2  -> expected outgoing rho.b*(0.5+(1+N.z^2)/16)
```

moment와 비상수 적분 허용 오차는 샘플 수에 따른 이론적 표준오차의 6배와 작은
부동소수점 여유로 정했다. 64개 joint bin의 chi-squared는 150 미만을 요구한다.
이 기준은 선택한 고정 seed들에 대한 분포 검사이며 난수의 완전한 독립성 증명은 아니다.

## 조건과 실제 결과

13개 조건마다 131,072개, 총 **1,703,936개** GPU 샘플을 확인했다.

- ±X, ±Y, ±Z 정상.
- 기울어진 정상, pole 근처 정상, 보조 축 전환 양쪽 정상.
- grazing outgoing 방향.
- colored reflectance/environment와 black/white/HDR 조합.

추가로 u=0과 1 직전값, NaN/Inf/범위 밖 난수, 잘못된 정상·outgoing, horizon/backside,
잘못된 reflectance 등 **34개 경계/입력 사례**를 검사했다. Sample 결과와 상관없는
방향 조합 **16개**로 Eval/PDF를 따로 검사해, cosine 중복과 방향/반구 오류를 확인했다.

MX450 / CUDA 12.0 NVRTC compute_75 Debug 결과:

| 항목 | 결과 |
|---|---|
| gray의 상수 radiance 출력 | (4,4,4) |
| PDF 최대 절대 오차 | 5.787330332e-8 |
| explicit f*cos/PDF weight 최대 오차 | 1.192092896e-7 |
| 상수 radiance 최대 절대 오차 | 5.960464478e-8 |
| 최대 joint histogram chi-squared | 83.90609113 (기준 <150) |
| 방향 moment / 비상수 환경 적분 | 전 조건 통과 |
| CTest | 기존 12개 + CudaDiffuseBSDF, 13/13 통과 |
| CUDA/OptiX OFF DX12 전용 build-raster | configure/build 통과 |

## 업로드 순서 보강

대형 입력에서 pageable HtoD와 non-blocking stream 사이의 순서 의존이 실제로 드러났다.
HtoD 함수가 CPU에 반환했어도 staging 이후 device DMA까지 완료됐다고 보장되지는 않는다.
진단 입력은 pinned buffer에서 커널과 같은 stream에 HtoDAsync로 올려 순서를 보장했다.
CPU는 완료 event 뒤에만 검증 데이터를 읽는다.
[CUDA 12.0 동기화 규칙](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-driver-api/api-sync-behavior.html).

같은 유형의 초기화 순서 의존을 제거하기 위해 기존 GAS geometry 업로드 뒤와 SBT
업로드 뒤에도 명시적인 초기화 완료 동기화를 추가했다. 초기화 한 번에 적용하며
매프레임 단계에 새 CPU 대기를 추가하지 않는다. 기존 GPU fence handoff는 유지한다.

## 계측 범위

CUDA event를 커널 앞뒤에 기록해 `kernel_ms`를 CSV에 저장한다. input upload 뒤,
output readback 앞의 커널 구간이다. 첫 sample 커널은 warmup 후 측정한다.

이 커널에는 BSDF 함수 외에 여러 진단값 계산과 큰 output record 저장이 포함된다.
시간에는 GPU 스케줄링·클럭 등도 영향을 준다. 이 수치를 SampleDiffuse 단독 비용,
메인 렌더러의 OptiX 시간, GI 시간이나 FPS로 해석하지 않는다. 기존 다섯 패스의
GPU baseline 계측은 아직 별도 후속 작업이다.

## 빌드와 실행

기존 `HYBRIDTRACER_BUILD_INTEROP_TEST=ON`에서 새 진단 실행 파일을 빌드한다.
이 BSDF 검사 자체에는 OptiX SDK가 필요하지 않으며 CUDA Driver + NVRTC를 사용한다.
CUDA를 끈 기본 DX12 빌드 경로도 유지한다. 원본의 hybrid/interop 옵션은 ON이다.

```powershell
Set-Location -LiteralPath 'C:\GitHub\HybridRenderer\HybridTracer\build'
.\Debug\HybridTracerDiffuseBSDFTest.exe --samples
$LASTEXITCODE
```

최종 PASS와 exit 0을 확인한다. 현재 폴더의 DiffuseBSDFReport.csv에는 13개 조건의
평균·오차·분포·계측값이, DiffuseBSDFSamples.csv에는 조건마다 처음 4개 sample이
저장된다. CTest는 두 파일을 build 폴더에 쓴다.

CMake의 DiffuseBSDFRuntime target은 header-only 편집도 runtime shader에 반영되도록
매 build에서 copy_if_different를 수행한다. 소스에 .cu가 있어도 NVCC를 호출하지 않는다.
공용 함수는 CUDA device header이며 다음 OptiX NVRTC program에 header로 제공해 사용할
수 있다. 이번에는 실제 OptiX indirect shader에서 호출한 것은 아니다.

후속 작업은 secondary hit의 위치·법선·재질 접근을 확장하고 이 샘플러를 연결하는 것이다.
처음에는 1회 반사 indirect radiance를 별도 출력해 검증한 뒤 Composite의 Ambient를
교체한다. 임의로 기존 directional Intensity에 pi를 더하거나 빼서 밝기를 맞추지 않으며,
광원 물리량 계약과 기준 장면은 GI 조명 계산 단계에서 명확히 정한다.

참고: 사용자 제공 tutorial3_BSDF cell 5~8/20~22 및 hw3_sub2_assignment_clean cell 9~11.
[Mitsuba 3.6 BSDF API](https://mitsuba.readthedocs.io/en/v3.6.0/src/api_reference.html#mitsuba.BSDF),
[PBRT cosine-weighted reflection sampling](https://pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Sampling_Reflection_Functions).
