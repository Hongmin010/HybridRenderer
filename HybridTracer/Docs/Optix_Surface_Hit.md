# OptiX SurfaceHit: 두 번째 표면의 셰이딩 데이터

2026-11-03. Diffuse BSDF Sample/Eval/PDF 준비 다음으로, OptiX가 맞힌 표면의 위치·법선·
재질색을 GPU에서 가져오는 경로를 구현했다. 기존 OptixTracePass에 별도의 GPU ray query
입출력 API를 추가하고, 같은 GAS/SBT/closest-hit을 재사용한다.

이 문서는 hit 조회 단계의 기록이다. 후속 [1회 반사 간접광](One_Bounce_Indirect.md)을
메인 Composite까지 연결했고 Ambient를 제거했다. 현재 전체 CTest는 15개다.

## 입력과 출력

```cpp
trace.Initialize(cudaContext, stream, gas, width, height);
trace.LaunchSurfaceQueries(deviceRays, deviceHits);
```

두 인자는 CUDA device pointer이며 각각 width*height개의 SurfaceRay / SurfaceHit 배열을
가리킨다. CPU readback을 패스 내부에서 수행하지 않는다. caller는 입력/출력을 별도로
할당하고 stream 완료까지 살아 있게 해야 한다. SurfaceRay와 SurfaceHit 타입은
`Shaders/SurfaceHit.h`에 host/device 공통으로 정의했다.

| 구조 | 크기 | 필드 |
|---|---:|---|
| SurfaceRay | 48 bytes | world origin, unit world direction, tmin/tmax, Valid, padding |
| SurfaceHit | 80 bytes | world position, distance, Ng, Ns, barycentric u/v, Albedo, object/primitive ID, Valid, FrontFace |

광선은 finite origin/direction과 0 <= tmin < tmax를 요구한다. direction의 length-squared
오차는 1e-4 이내다. Valid != 1이나 잘못된 입력은 trace하지 않고 zero hit record를 쓴다.
miss도 전체 record를 0으로 덮어쓴다. ObjectIndex=0은 실제 바닥 ID이므로 ID가 아니라
**Hit.Valid로 hit/miss를 판단**한다. t는 offset 이후 제공한 광선 원점 기준이며, 단위
방향에서 월드 거리와 같다.

기존 `Launch(deviceSurfaceInputs, shadowSettings)`는 16-byte ShadowResult를 그대로
출력한다. query API와 shadow API는 같은 실행 객체에서 번갈아 사용할 수 있다.
launch parameter는 매 호출 완전히 설정해 이전 mode의 pointer가 남지 않도록 한다.
기존 pinned parameter/event 재사용 계약대로 이전 launch 완료 후 다음 호출을 한다.
여러 launch를 CPU 대기 없이 겹치는 parameter ring은 이번 범위에 포함하지 않는다.

## geometry / normal / material 데이터

GAS가 소유하는 per-object GPU 데이터에 다음을 추가했다.

- 기존 world-space vertex와 uint3 index.
- 원본 vertex normal에 world inverse-transpose를 적용하고 정규화한 float4 normal 배열.
- 원본 Material의 linear float RGB reflectance snapshot.

host/device geometry 및 SBT data layout은 `Shaders/OptixSceneData.h`를 공유한다.
SceneHitData는 vertex/index/normal GPU 포인터와 object ID, triangle count, albedo를 담는다.
data 48 bytes + SDK header 32 bytes = hit record 80 bytes이며 SDK 16-byte 정렬을 지킨다.
GAS의 object별 build input / SBT record 대응은 유지한다.

현재 재질은 object당 하나의 Diffuse Albedo만 지원한다. 텍스처, UV, emission, roughness,
다중 material index는 아직 없다. material과 geometry 속성은 정적 snapshot이며 Scene의
정점·법선·색·Transform을 바꾸면 GAS/trace 초기화를 다시 해야 한다. 동적 update는 없다.
world transform은 finite하고 determinant 크기가 1e-12 이상이어야 하며 변환된 vertex normal도
finite/nonzero여야 한다. Diffuse reflectance는 RGB [0,1] 계약을 검사한다.

secondary hit의 Albedo는 RGBA8 G-buffer를 다시 읽는 값이 아닌 원본 material float다.
예를 들어 0.1은 primary UNORM에서 25/255가 될 수 있지만 secondary에서는 원본 float
0.1을 유지한다. 법선도 G-buffer half 대신 float normal 배열을 읽는다. 이 두 표현의
양자화 차이는 정상이며 CPU 기준 역시 원본 mesh/material을 사용한다.

## closest-hit 처리

`Shaders/OptixSurface.cuh`의 ReadCurrentHit는 closest-hit context에서 호출한다.

1. optixGetPrimitiveIndex로 **해당 build input 안의** primitive 번호를 얻는다.
2. SBT index buffer로 삼각형의 세 vertex index를 읽는다.
3. optixGetTriangleBarycentrics의 (u,v)와 w=1-u-v를 사용한다.
4. world vertex를 보간해 위치를 계산한다.
5. Ng는 world triangle edge의 cross product를 정규화한다.
6. Ns는 world vertex normal을 보간한 뒤 정규화한다. 보간 길이가 거의 0이면 Ng로 대체한다.
7. 원본 albedo와 object/primitive ID, distance, native front-face flag를 기록한다.

Ng는 winding 기준의 **기하 법선**, Ns는 vertex normal 기준의 **셰이딩 법선**이다.
둘을 구분하고 임의로 같은 방향으로 뒤집지 않는다. 음수 scale에서는 두 방향이
반대일 수도 있다. FrontFace는 OptiX native hit flag이며 단순 양면 shader 정책을 뜻하지
않는다. 향후 one-sided BSDF의 유효 면/normal 방향과 ray offset 정책을 이 정보로 정한다.
back-face hit 자체를 없애거나 Ns를 자동으로 face-forward한 것은 아니다.

API는 설치된 OptiX 8.0 header를 대조했으며,
[NVIDIA device API](https://raytracing-docs.nvidia.com/optix8/api/group__optix__device__api.html)도 참고했다.
그림자 mode에서는 추가 위치/법선/재질 출력을 수행하지 않는다.

## 검증 장면과 CPU 기준

새 `HybridTracerSurfaceQueryTest`는 headless DX12의 실제 GPU를 LUID로 매칭하고 같은
CUDA context/stream으로 GAS와 production OptixTracePass를 실행한다.

기존 바닥·두 큐브·기울어진 평면에 검사 전용 삼각형 2개를 추가한다.

- 서로 다른 vertex normal을 가진 매끄러운 삼각형 + 회전 + 비균일 scale.
- 같은 종류의 삼각형 + 회전 + 음수/비균일 scale.

Scene::AddObject는 mesh/material 소유권을 이전받는 일반 추가 API다. 검사 장면에만
이 삼각형을 추가하며 기본 메인 장면은 바꾸지 않았다. 검사 장면은 6 objects,
62 vertices, 30 triangles다.

원본 로컬 메시와 world matrix에서 CPU double 삼각형을 구성한다. CPU position은
scalar matrix 곱, normal은 독립적인 double cofactor inverse-transpose로 계산한다.
GPU 업로드 staging vector나 GPU normal buffer의 readback을 기준값으로 사용하지 않는다.
각 ray를 모든 원본 삼각형과 CPU Moller-Trumbore 검사해 nearest와 보간값을 비교한다.

각 배치의 입력:

- 바닥 위 합성 위치에서 생성한 cosine hemisphere secondary 방향 16,384개.
- 모든 primitive의 여러 내부 barycentric 위치를 앞뒤에서 겨냥한 광선 240개.
- tmin/tmax로 hit가 잘려야 하는 광선 4개.
- 명시적 miss 2개와 disabled/NaN/Inf/zero direction/잘못된 bounds 입력 9개.

이 입력들은 **진단용 CPU 생성 광선**이다. 현재 Raster SurfaceInput에서 GPU BSDF 샘플을
생성해 GI로 이어 붙인 것은 아니다. 일반 GPU ray query 입출력의 정확성을 먼저 검증했다.

4개 배치, 총 **66,556개 query**를 검사한다. 배치마다 input을 변경하고 output을 0xcd로
채워서 miss/invalid record에 이전 결과가 남으면 실패하게 한다. query 뒤에는 같은 trace
객체를 shadow mode로 실행해 parameter와 출력 mode 전환도 확인한다.

검사 항목:

- hit/miss, tmin/tmax, 잘못된 광선의 zero 출력.
- object 및 primitive가 CPU nearest를 설명하는지. 경계에서 복수 primitive가 수치상
  같은 거리에 있을 수 있어 reported primitive 자체도 독립적으로 교차 검사한다.
- ray origin + t*direction, 원본 삼각형 보간, GPU world position 일치.
- geometric normal과 보간·inverse-transpose shading normal의 별도 비교 및 unit length.
- 원본 float material color, barycentric u/v, native front/back flag.
- 배치마다 30개 primitive 전체 hit, smooth/mirrored normal 차이와 양면 hit가 실제 존재.
- D3D12 warning/error 및 예상 밖 OptiX warning/error가 없는지.

## 결과

MX450 / CUDA 12.0 / OptiX 8.0.0 Debug 결과:

| 항목 | 결과 |
|---|---|
| 전체 query | 66,556 (4 × 16,639) |
| hit / miss / invalid | 14,721 / 51,799 / 36 |
| primitive coverage | 매 배치 30/30 |
| 최대 distance 오차 | 2.190696039e-5 |
| 최대 world position 오차 | 2.208377967e-5 |
| 최대 Ng / Ns 벡터 오차 | 1.325661723e-7 / 1.042084638e-6 |
| 최대 barycentric 오차 | 5.395111931e-6 |
| 원본 Albedo 오차 | 0 |
| query → shadow → query mode 재사용 | 통과 |
| 기존 검사 포함 CTest | 14/14 통과 |

distance 허용 오차는 2e-5*max(1,t), 위치는 절대 3e-5, Ng/Ns와 barycentric은 2e-5다.
CPU double 원본 transform과 GPU float world vertex의 차이가 grazing ray에서 증폭될 수
있어 bit-exact 교차 위치를 요구하지 않는다. material float는 그대로 일치했다.
OptiX의 알려진 launcher allocation 알림 2회는 계속 로그/횟수를 남긴다.

## 빌드와 직접 실행

기존 `HYBRIDTRACER_BUILD_INTEROP_TEST=ON`, `HYBRIDTRACER_BUILD_OPTIX_GAS_TEST=ON`에서
새 진단 target과 CTest OptixSurfaceQueries를 빌드한다. shader header-only 편집도 새
query runtime target과 hybrid renderer runtime target이 copy_if_different로 반영한다.
NVRTC가 프로젝트 header를 이름 있는 메모리 include로 받고 PTX를 생성한다.

```powershell
Set-Location -LiteralPath 'C:\GitHub\HybridRenderer\HybridTracer\build'
.\Debug\HybridTracerSurfaceQueryTest.exe --samples
$LASTEXITCODE
```

마지막 PASS와 exit 0을 확인한다. OptixSurfaceHitSamples.csv는 현재 폴더에 각 object의
대표 hit와 miss를 저장하며 21 columns, 52 data rows다. 전체 query 비교는 표본 CSV와
관계없이 모두 수행한다. CUDA/OptiX OFF의 DX12-only configure/build도 통과했다.

후속 구현에서 첫 SurfaceInput의 BSDF 광선 생성과 두 번째 표면 조명을 연결했다.
직접광과 간접광은 별도 보기/출력으로 검사하며 상세는 [간접광 문서](One_Bounce_Indirect.md)에 있다.
