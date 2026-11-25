# 메인 화면의 DX12 / CUDA / OptiX 전체 패스

Lambert 확산 재질에서 직접광과 [최대 8회 간접 반사](Multi_Bounce_Indirect.md)를 계산한다.
방향광, 점광원, 사각형 면광원을 한 장면에서 함께 사용할 수 있다.
[프레임 누적](Temporal_Accumulation.md)과 [GPU/CPU 계측](Frame_Timing.md)을 지원한다.

```text
RasterGBufferPass (DX12) → Albedo RGBA8 / World Normal RGBA16F / D3D Depth R32F
SurfaceInputPass (CUDA) → Position / normalized Normal / Albedo / Valid (48-byte GPU packet)
OptixTracePass (OptiX) → 방향광 ShadowResult, 간접광, 추가 광원의 직접광
TemporalAccumulationPass (CUDA) → 간접광과 추가 광원의 직접광을 선형 공간에서 누적
CompositePass (DX12) → 방향광 직접광 + 추가 광원의 직접광 + 간접광 → linear HDR RGBA16F
ToneMapPass (DX12) → Exposure → Reinhard → exact piecewise sRGB → UNORM BackBuffer
Present
```

현재 여섯 핵심 패스다. GI는 OptixTracePass에서 생성하고 별도 CUDA pass에서 누적한다. primary visibility는
raster가 결정하며 OptiX는 shadow와 secondary ray를 처리한다. 정적 GAS는 초기화 때
한 번 만든다. 메인과 headless 검사는 같은 HybridFramePipeline을 사용한다.
Raster의 임시 RT0는 hybrid 최종 합성에서 읽지 않는다.

## 실행과 조작

```powershell
.\build\Release\HybridTracer.exe --scene .\Scenes\Studio.xml --bounces 2
```

창을 선택하고 숫자 키를 누른다. 왼쪽 드래그로 orbit, 휠로 zoom한다.

| 키 | 보기 |
|---:|---|
| 1 | 그림자 적용 직접광 + 간접광 |
| 2 | Albedo |
| 3 | World Normal |
| 4 | Depth 대비 표시 |
| 5 | 첫 표면의 광원 가시성. 혼합 광원은 그림자 적용 전후의 직접광 RGB 합의 비율 |
| 6 | A의 direct shadow만 끈 direct + 같은 indirect |
| 7 | shadowed Direct only |
| 8 | Indirect only, 검은 배경 |

기본은 4 spp/frame, 새 seed, 누적 ON이다. T로 누적 전환, R로 초기화한다. 카메라를 멈추면 노이즈가 줄고 이동 중에는 초기화된다. denoise는 아직 없다.
배경은 표시색일 뿐 환경광이 아니다. 2~5 raw 진단은 ToneMap/sRGB를 우회한다.

### 광원 선택과 장면 설정

앱과 `HybridTracerBenchmark.exe`는 `--scene`으로 씬 파일을 연다. 경로를 생략하면
실행 파일 옆의 `Scenes/Default.xml`을 읽는다. [기본 씬](../Scenes/Default.xml)과
[스튜디오 예제](../Scenes/Studio.xml)를 복사해서 씬별 물체와 광원을 편집할 수 있다.
실행할 때 지정한 XML을 읽으므로 파일을 편집한 뒤 앱을 다시 열면 된다.

```xml
<scene>
  <objects>
    <plane position="0 -1 0" scale="4 1 4" albedo="0.45 0.45 0.45"/>
    <cube scale="0.7 0.7 0.7" albedo="0.8 0.1 0.08"/>
  </objects>
  <lights>
    <point position="-2 3 -2" color="1 0.78 0.61" intensity="22"/>
    <rectangle center="1.5 4 -0.5" half_u="1.25 0 0" half_v="0 0 0.8" radiance="5"/>
  </lights>
</scene>
```

- `<objects>`에 기본 `cube`와 `plane`을 배치할 수 있다. `position`, `rotation`(라디안),
  `scale`, 선형 RGB `albedo`는 생략할 수 있다. 각 기본 도형의 로컬 크기는 2이고
  `plane`은 노멀이 +Y인 XZ 평면이다.
- `<lights>`에는 원하는 광원만 작성한다. 같은 종류를 여러 번 등록할 수 있고,
  `<lights/>`는 광원 0개다. 광원의 `color` 기본값은 선형 RGB 흰색이다.
- `directional`은 표면에서 광원을 향하는 `direction` 벡터와 `irradiance`를 받는다.
  `point`는 `position`과 `intensity`를 받는다.
  `rectangle`은 `center`, 두 반변 벡터 `half_u`, `half_v`, `radiance`를 받는다.
- 방향광 한 개를 제외한 나머지 광원은 최대 16개다. 점광원과 면광원의 그림자 광선은
  광원 위치까지 추적한다. 사각형은 `half_u × half_v` 방향으로 발광하는 가상 발광판이다.

### Bitterli Cornell Box

```powershell
.\build\Release\HybridTracer.exe --scene .\Scenes\CornellBox\scene_v3.xml --bounces 1
```

앱, 벤치마크, HDR 캡처 도구에서 같은 Mitsuba 3 XML을 열 수 있다. 현재 읽는 항목은
원근 카메라의 FOV와 필름 크기, 큐브와 사각형의 `to_world` 행렬, RGB 확산·양면
확산 재질, 사각형에 붙은 면광원이다. Cornell Box는 1024×1024 카메라에서 열리고,
천장 발광판이 카메라에 보이며 색 벽과 상자를 비춘다. 채택한 샘플 수와 반사 횟수는
HybridTracer 실행 설정을 따른다. 원본 XML 옆에 배포본 라이선스를 함께 보관한다.

## API 간 동기화와 수명

```text
CPU: 이전 프레임 완료 대기 (한 세트의 frame 자원 재사용 보호)
DX12: Raster, G-buffer RT → COMMON, ExecuteCommandLists, raster-ready Signal
CUDA: raster-ready Wait, G-buffer → SurfaceInput, OptiX launch
CUDA: raw indirect → temporal mean (same stream)
CUDA: private shadow → shared shadow, shadow-ready Signal
CUDA: selected raw/temporal indirect → shared indirect, indirect-ready Signal
DX12: 두 ready fence Wait, G-buffer / 두 buffer → PIXEL_SHADER_RESOURCE
DX12: Composite → ToneMap → optional DebugView → Present
DX12: 공유 structured buffers → COMMON 복귀
```

두 출력은 byte 크기를 받는 기존 SharedShadowBuffer 구현을 각각 재사용한다.
fence마다 signaler가 하나이고 값은 증가한다. SurfaceInput의 done은 OptiX 완료를
의미하지 않는다. 각 output ready fence는 해당 GPU copy 뒤에 signal한다.

메인의 SurfaceInput은 readback=false, GI 진단 buffer도 할당하지 않는다. 작은
카메라/launch parameter upload 외에 이미지 CPU 전송은 없다. G-buffer, SurfaceInput,
outputs, HDR와 pinned parameter가 한 세트여서 현재는 프레임 완료 뒤 재사용한다.
frame 자원 분리와 overlap은 후속 작업이다.

SubmitAndContinue는 raster 제출 뒤 command list만 reset한다. allocator reset은 기존
frame fence로 보호한다. 종료 시 DX12 queue의 CUDA wait까지 완료한 뒤 views → 공유
output mappings → trace → GAS → CUDA owner 순서로 해제한다.

## 합성과 검증

```text
E = LightColor * Intensity       // perpendicular-plane irradiance, default intensity=pi
direct = Albedo/pi * E * max(N·L,0) * visibility
HDR = direct + temporal_mean(IndirectLighting)  // indirect already includes A's BSDF throughput
display = sRGB(Reinhard(HDR * Exposure))
```

HybridTracerIndirectTest는 CPU 원본 삼각형 교차와 radiometry로 GI를 검증한다.
HybridTracerCompositeTest는 321×181 / 1280×720에서 10프레임씩 같은 pipeline을 실행한다.

- DX12와 CUDA private shadow/indirect 전 바이트 일치.
- 전 픽셀의 primary shadow hit/miss, 거리와 object를 CPU scene과 비교.
- 전 픽셀 HDR 합성, raw debug, Reinhard/sRGB와 alpha/finite 확인.
- 고정 seed의 shadow on/off, HDR>1, 어두운 sRGB 선형 구간과 8개 보기.
- D3D12 warning/error와 예상 밖 OptiX warning/error 발생 시 실패.

전체 CTest는 20개다. 누적 수학/전체 프레임/이동 초기화 검사까지 추가했다. 기존 single-frame 합성 검사는 누적 OFF로 유지한다. 알려진 내부 launcher allocation notice는 로그/횟수를 남긴다.
[BSDF](CUDA_Diffuse_BSDF.md), [SurfaceHit](Optix_Surface_Hit.md)와 기존 interop/GAS 검사를
유지한다. CPU/readback을 포함한 검사 시간은 GPU frame time이나 FPS가 아니다.

## 빌드와 재검증

기본 옵션은 CUDA/OptiX OFF의 DX12-only 빌드다. 원본 build는 hybrid ON이다.

```powershell
Set-Location -LiteralPath 'C:\GitHub\HybridRenderer\HybridTracer'
$cmakeExe = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$sdk = 'C:\Users\hongm\Documents\ChatGPT\졸업프로젝트\references\optix-sdk-8.0'
& $cmakeExe -S . -B build -DHYBRIDTRACER_ENABLE_HYBRID_RENDERER=ON -DHYBRIDTRACER_BUILD_INTEROP_TEST=ON -DHYBRIDTRACER_BUILD_OPTIX_GAS_TEST=ON "-DHYBRIDTRACER_OPTIX_ROOT=$sdk"
& $cmakeExe --build build --config Debug
& (Join-Path (Split-Path $cmakeExe) 'ctest.exe') --test-dir build -C Debug --output-on-failure
.\build\Debug\HybridTracer.exe --smoke-test
```

smoke-test는 숨겨진 창에서 12프레임, 8개 보기와 카메라 이동을 검사하고 자동 종료한다.

```powershell
Set-Location -LiteralPath 'C:\GitHub\HybridRenderer\HybridTracer\build'
.\Debug\HybridTracerCompositeTest.exe --full-resolution --captures
.\Debug\HybridTracerIndirectTest.exe --samples
```

capture는 실제 GPU 출력으로 고정 seed/32 spp를 쓴다. 현재 폴더에 HybridGI.bmp,
HybridDirect.bmp, HybridIndirect.bmp, HybridUnshadowed.bmp, HybridVisibility.bmp를 쓴다.
GI/direct/indirect는 같은 카메라와 광원이다. SDK는 외부에 유지하며 프로젝트 shader와
NVRTC runtime만 실행 파일 옆에 배치한다.
