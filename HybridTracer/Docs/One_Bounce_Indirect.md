# 1회 반사 Diffuse 간접광

2026-11-03. Diffuse BSDF와 SurfaceHit을 메인에 연결했고 상수 Ambient를 제거했다.
2026-11-05 후속: [GPU/CPU 시간 계측](Frame_Timing.md)과 반복 benchmark를 추가했다. 후속 [프레임 누적](Temporal_Accumulation.md)도 연결했으며 현재 CTest는 20개다.

```text
Camera → A: DX12 rasterization + G-buffer 복원
A → B: cosine hemisphere BSDF sample + OptiX closest hit
B → directional light: Lambert evaluation + OptiX visibility
HDR(A) = direct(A) + mean(indirect samples)
```

## 물리량과 가중치

`E = LightColor * Intensity`는 광원 방향에 수직인 평면의 irradiance다.
`direct = rho_A/pi * E * max(N_A·L,0) * V_A`이다. 기본 Intensity는 pi로,
이전 `rho*cos*Intensity`의 Intensity=1과 직접광 밝기가 같다. Ambient 보정은 없다.

`pdf_A=cos_A/pi`, `Sample.Weight=rho_A`이므로 한 간접광 표본은
`rho_A * (rho_B/pi) * E * max(N_B·L,0) * V_B`다.
A의 albedo/cosine/PDF를 다시 적용하지 않는다. B는 원본 float material, A는
기존 linear RGBA8 UNORM 양자화값이다. 방향성 광원은 delta이므로 이 제한된 경로에
별도 light PDF/MIS가 필요하지 않다. 합성은 선형 HDR, 표시 변환은 ToneMap에서 수행한다.

## 구현과 범위

- `OptixIndirect.cuh`: BSDF sampling, secondary hit, B의 직접광/그림자.
- `IndirectLighting.h`: 16-byte RGB/Valid, 32-byte 설정과 선택적 진단 ABI.
- `OptixTracePass::LaunchIndirect`: primary shadow와 간접광을 한 launch에서 실행.
- `HybridFramePipeline`: 두 GPU 출력을 각각 공유 buffer로 copy하고 DX12 fence wait.
- `CompositePass`: direct + 이미 A의 BSDF가 포함된 indirect를 더한다.

raygen에서 순차 trace하므로 maxTraceDepth=1을 유지한다. B→light는 CH를 비활성화해
B의 hit record를 보존한다. blocker=1에서 시작하고 miss가 0으로 바꾼다.
GI는 pixel당 private radiance 16 bytes, scratch SurfaceHit 80 bytes, 공유 radiance
16 bytes를 추가한다. 메인은 진단 배열을 할당하거나 이미지 readback하지 않는다.
기존 single in-flight frame 보호는 유지한다.

pixel/frame seed/sample/dimension을 정수 mixer로 조합하고 24-bit [0,1)로 변환한다.
메인은 4 spp/매 프레임 새 seed이며 현재는 누적 기본 ON이다. 이 문서의 이전 capture는 고정 seed/32 spp다. 새 누적 검증/이미지는 Temporal_Accumulation.md를 참고한다. denoise는 아직 없다.

B는 winding의 Ng 기준 앞면만 반사한다. Ns를 Ng와 같은 hemisphere로 맞춰 BSDF를
계산하고 Ng로 offset한다. Ng·wo 또는 Ng·L≤0이면 배제한다. shadow는 양면 opaque다.
첫 표면은 G-buffer의 단일 normal로 offset한다. 현재 flat diffuse 장면이 대상이며
smooth surface의 shading-normal 보정/primary Ng 분리는 추가 검증이 필요하다.
offset=0.001, tMin=0.0001은 현재 장면 스케일의 고정값이다.

미스 radiance는 0이다. 파란 표시 배경은 환경광으로 쓰지 않는다. 환경/면광원, emissive
geometry, 다중 간접 반사, glossy/transmission, Russian roulette은 후속 범위다.

## 검증

`HybridTracerIndirectTest --samples`는 16,384개 합성 A에서 실제 GPU가 생성한 ray를
읽고 원본 Scene의 CPU double 삼각형 교차로 B와 B→light 차폐를 독립적으로 계산한다.
default scene의 flat faces에 대한 물리식과 비교한다. 기존 SurfaceQuery 검사는
비균일/음수 scale와 smooth normal 등의 hit 데이터 자체를 별도로 검증한다.

- seed=17: hit 4,522 / miss 11,861 / lit B 2,962 / shadowed B 248 (배경 1개 제외).
- seed=29: hit 4,516 / miss 11,867 / lit B 2,945 / shadowed B 260.
- 최대 선형 RGB 벡터 오차 약 4.63e-8. 같은 seed는 byte-exact 재현.
- 광원=0, A reflectance=0, 뒤를 향한 A normal, 배경은 모두 간접광 0.
- 별도 일정 radiance plane의 RGB=(0.2,0.6,0.15)를 1/4/64/128 spp에서 확인한다.
  이는 throughput과 평균의 분모가 spp에 따라 밝기를 바꾸지 않는지 검사한다.
- host API는 spp=0과 음수 irradiance를 거부한다.

통합 검사는 같은 production pipeline을 두 해상도에서 10프레임 실행한다. 전 픽셀의
공유 shadow/indirect bytes, primary CPU shadow, HDR 합성, 8개 보기와 sRGB를 확인한다.
GI 자체의 독립 기준은 위 standalone 검사다. 합성 검사는 readback GI에 A의 albedo가
다시 곱해지지 않는지 확인한다. 전체 15개 CTest, 원본 숨겨진 12프레임 smoke와
CUDA/OptiX OFF 빌드 결과는 workspace `verification-indirect/Verification_Summary.md`에 남긴다.
알려진 OptiX launcher allocation notice는 기록하고 예상 밖 warning/error는 실패한다.
검사 시간은 CPU/readback을 포함하므로 GPU FPS가 아니다.

## 실행

```powershell
& 'C:\GitHub\HybridRenderer\HybridTracer\build\Debug\HybridTracer.exe'
```

1: direct+indirect, 7: direct only, 8: indirect only. 6은 A의 direct shadow만 끄고
GI/B의 shadow는 유지한다. 2/3/4/5는 기존 진단 보기다. 8의 배경은 검정이다.
왼쪽 드래그/휠로 카메라를 조작한다. GPU 계측과 history 누적/초기화를 완료했고 다음은 다중 bounce다.
