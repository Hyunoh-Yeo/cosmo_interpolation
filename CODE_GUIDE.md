# 코드 안내 — 뭐가 뭐고 왜 있는가

프로젝트 목표: **시뮬레이션이 없는 우주론 (Ωm, w0, wa)의 스냅샷을 보간으로 만들기.**
56개 시뮬이 초기조건을 공유하므로 같은 `indx` = 같은 입자이고, 그 사이를 보간한다.

작업은 두 단계로 나뉜다.

| 단계 | 질문 | 상태 |
|---|---|---|
| **1. 전제 검증** | 같은 입자가 우주론이 달라도 비슷한 위치에 있나? | ✅ 완료 |
| **2. 보간** | 실제로 만들어서 맞는지 확인 | ⬜ 아직 |

지금까지 한 일은 대부분 **1단계**다. 2단계(본론)는 아직 시작 안 했다.

---

## 파이프라인

```
데이터 읽기 → 전제 검증 → 통계/그림 → (보간) → 평가
  gotpm      verify_shared_ic   plot_*      mvinterp/   power_spectrum
                                                        correlation
```

---

## 1. 데이터 읽기 — 모든 것의 기초

### `gotpm.py` (218줄) ★ 여기서 시작
GOTPM 바이너리 스냅샷을 numpy로 읽는 **유일한 리더**. 나머지 전부가 이걸 import한다.

- `read_records()` — 파일 → 레코드 배열 (stride/id_mod/subsample 지원)
- `decode_positions()` — **핵심**: `indx`에서 태초 격자 위치를 복원하고 저장된 변위를 더해 실제 위치를 만든다
- `list_subfiles()` — `/gpfs`와 `/multiverse` 양쪽을 뒤져 우주론 이름을 경로로 해석

**왜 이렇게 만들었나**: 처음엔 스크립트마다 디코딩을 복붙해서 4벌이 돌아다녔다. 버그를 고치면 4곳을 고쳐야 해서 하나로 합쳤다. 원본 C 코드(`samples/read.c`)의 `XofP` 매크로와 일치함을 확인했다.

### `mvinterp/inspect_snapshot.py`
포맷 진단용. 엔디안을 양쪽으로 무차별 대입해서 어느 쪽이 맞는지 검증한다.
**일부러 `gotpm.py`를 안 쓴다** — 검증 대상을 import하면 검증이 안 되니까.

---

## 2. 전제 검증 — 미팅에서 나온 질문들에 답하는 코드

### `verify_shared_ic.py` (375줄) ★ 핵심 도구
두 우주론에서 같은 `indx`를 찾아 위치 차이 |Δr|을 잰다. **박스 전체 86억 입자.**

**왜 어려운가**: subfile이 z-슬랩이라, 같은 입자가 우주론마다 다른 subfile에 들어간다.
같은 subfile끼리 비교하면 **경계를 넘어간 입자(=제일 많이 튄 놈)가 빠진다.**
→ `run_window()`가 A의 슬랩 j를 B의 슬랩 j±w에서 찾아 이 편향을 없앤다.

측정하는 것:
- |Δr| 히스토그램, 백분위, 임계값(>0.5/1/2/5/10) 초과 개수
- **축별 rms** — 이동이 등방인가 특정 축으로 치우치나
- **슬랩별 median** — 첫/마지막 슬랩이 이상하면 주기경계 처리 오류
- **cos(Δr, ψ)** — Δr이 입자 자신의 변위 ψ와 나란한가 (아래 설명)
- top-20 movers 좌표, `--save-movers`로 임계값 넘는 것 전부 저장

**cos(Δr, ψ)가 왜 중요한가**: ψ = 현재위치 − 태초위치(indx에서). 만약 우주론 차이가
단지 변위를 키우거나 줄이는 것이라면(ψ_B = (1+ε)ψ_A) Δr = −ε·ψ_A, 즉 **완전히 평행**해진다.
→ cos이 ±1에 가까우면 "성장 재스케일", 0이면 "무작위". **어느 방향으로 커지는지**와
**왜 커지는지**를 동시에 답한다.

### `compare_particles.py` (55줄)
같은 측정의 **첫 버전** — subfile 하나만 본다. 위의 편향 문제가 있어서 대체됐다.
비교용으로만 남겨뒀다.

### `check_subfiles.py` / `plot_subfiles.py`
subfile이 정말 순서대로 겹침 없이 z를 덮는지 확인. `run_window`가 그 가정 위에 서 있어서
먼저 검증해야 했다. **결과: 250개 전부, 틈 0.000, 겹침 0.000.**

### `calibrate_velocity.py` (101줄)
속도의 km/s 변환계수를 구한다. 원본 C 코드엔 `Fact1/Fact2/Pfact` 선언만 있고
공식이 없어서, **선형이론 v = aHf·ψ로 역산**한다. ψ를 이미 계산하니 가능.

---

## 3. 병렬 실행 — SLURM

### `submit_ic.sh` → `job_ic_array.sh` + `job_ic_merge.sh`
공유 IC 스캔을 **독립 job 16개**로 나눠 던진다.

```
워커 0  → A의 subfile   0~15
워커 1  → A의 subfile  16~31
...                          → 각자 .npz 저장 → merge가 더함
```

**왜 독립 job인가** (미팅 지적사항): 처음엔 job 하나 안에서 프로세스 16개를 `&`로 띄웠는데,
그러면 (a) 노드 하나에 갇히고 (b) 64코어를 잡고 16개만 쓰고 (c) 하나 죽으면 곤란하다.
독립 job이면 SLURM이 빈 자리 아무데나 배치하고, 죽은 것만 재실행한다.

**왜 MPI가 아닌가**: 워커끼리 주고받을 게 없다. 통계량이 전부 덧셈이라 파일로 합치면 끝.
병합 결과가 단일 실행과 **비트 단위로 같음**을 검증했다.

### `merge_stats.py`
워커들의 `.npz`를 더한다. 히스토그램·카운터는 덧셈, top-20은 합쳐서 다시 상위 20개.

### `submit_pk.sh` → `job_pk_array.sh`
P(k)를 **우주론마다 독립 job**으로. P(k) 하나는 모든 입자가 한 메시에 올라가야 해서
워커로 못 쪼갠다 — 대신 우주론을 쪼갠다. 목록은 `cosmologies_z0.txt`.

---

## 4. 물리량 측정

### `power_spectrum.py` (100줄)
pypower로 P(k). 전체 박스가 필요하고, `--stride`로 I/O를 줄인다
(1280이면 입자 1/1280, 읽는 양 1/10).

### `correlation.py` (114줄)
P(k) → ξ(r) (FFTLog). **직접 쌍세기는 r_max에 따라 6시간~800시간**이라
푸리에 변환이 첫 선택. 해석해 대비 정확도 1e-5, 실측 k범위로 잘라도 0.05~0.3%.

### `snapshot_view.py` (118줄)
z-슬랩 → 2D 밀도 지도 (pypower CatalogMesh, NGP/CIC/TSC).

### `mvinterp/cpl_growth.py` (33줄)
성장인자 D(z), f(z). cosmoprimo(CAMB) 래퍼.
**직접 짠 ODE와 0.03% 일치를 확인한 뒤 라이브러리로 교체**했다.

---

## 5. 보간 모델 — 아직 mock만

### `mvinterp/compare.py` (177줄)
linear / quadratic / RBF / GP 4개를 mock 데이터로 비교. scipy 사용.

### `mvinterp/gpu_interp.py` (211줄)
CuPy로 GPU 보간, 입자 축을 타일링해서 2048³ 처리.
**직접 짠 유일한 수치 코드** — scipy는 CPU 전용이라 타일링을 못 한다.

### `mvinterp/gp_interp.py` (137줄)
GPyTorch GP. 불확실도를 같이 준다.

### `mvinterp/make_mock.py` (143줄)
mock 스냅샷 생성. 실데이터가 오면 삭제될 임시 코드.

---

## 6. 그림 — 전부 `make_*` 또는 `plot_*`

| 파일 | 무엇 | 미팅 중요도 |
|---|---|---|
| `make_omladder_fig.py` | ΔΩm 사다리 → **Ωm 보간 가능** | ★★ |
| `make_scaling_fig.py` | 이동 ∝ 성장인자 (암흑에너지만) | ★★ |
| `make_movers_fig.py` | movers 위치, smoke vs 풀스캔 | ★ |
| `make_movers_proj_fig.py` | 위와 같되 투영만 | ★ |
| `make_dr_fig.py` | \|Δr\| 생존곡선 | ★ |
| `make_w0wa_fig.py` | (w0,wa) 격자 | ★ |
| `make_param_space_fig.py` | 격자 + σ8 vs Ωm | |
| `make_method_figs.py` | 보간 모델 4개 정확도 (mock) | |
| `plot_stats.py` | \|Δr\| 전체 히스토그램 + 슬랩 + 축 진단 | ★ |
| `plot_pk.py`, `plot_movers.py`, `plot_slabs.py`, `plot_subfiles.py` | 각 데이터 시각화 | |

`make_movers_fig_old.py`는 이전 버전(same vs different Ωm 2행) — 요청으로 남겨둠.

---

## 7. 문서

- **`meeting_notes.md`** — 미팅용 정리 (결과 + 근거 + caveat)
- **`FINDINGS.md`** (442줄) — 전체 기록. 데이터 포맷, 검증 결과, 함정 등
- **`CODE_GUIDE.md`** — 이 문서

---

## 실행됐나 vs 코드만 있나

| | 상태 |
|---|---|
| `verify_shared_ic` 기본 측정 | ✅ **6쌍 풀스캔 완료** |
| `verify_shared_ic` 방향(cos) 측정 | ⬜ 코드만 — 아직 안 돌림 |
| `calibrate_velocity` | ⬜ 코드만 |
| `correlation` | ⬜ 코드만 (P(k) 1개는 있음) |
| `submit_pk` | ⬜ 코드만 |
| 보간 (mvinterp) | ⚠️ **mock만**, 실데이터 미적용 |

**즉 지금 확실히 아는 것은 "전제가 성립한다"까지**이고, 나머지는 도구가 준비된 상태다.
