# FLU Model

## 개요

이 프로젝트는 지역 구조와 연령 구조를 함께 반영하는 동절기 인플루엔자 확산 시뮬레이션 코드입니다.
최종 제출 범위는 여러 counterfactual 실험을 모두 수행하는 것이 아니라, 하나의 calendar 기반 다지역·다연령 모델에서 두 가지 초기조건 방식이 결과를 어떻게 바꾸는지 비교하는 데 맞춰져 있습니다.

모델은 다음 7개 상태를 사용합니다.

- `S0`: 첫 감염 전 감수성 인구
- `E0`: 첫 감염 잠복기 인구
- `I0`: 첫 감염 감염 인구
- `R`: 회복 인구
- `S1`: 면역 소실 후 재감염 가능 인구
- `E1`: 재감염 잠복기 인구
- `I1`: 재감염 감염 인구

## 최종 실험 범위

최종 실험은 아래 두 초기조건만 비교합니다.

- `same_prevalence`: 모든 세부 지역·연령 셀에 동일한 초기 유병률을 적용합니다.
- `equal_absolute_seed`: 서울 상위 그룹(`Region_A`)과 원주 상위 그룹(`Region_B`) 각각에 대해 연령집단별 동일한 절대 seed 수를 부여한 뒤, 세부 지역 population 비율에 따라 seed를 분배합니다.

두 실험은 population, 연령 접촉행렬, 지역 접촉행렬, calendar, beta, reinfection, waning, susceptibility 설정을 모두 동일하게 두고, 초기 감염자 배정 방식만 다르게 둡니다.

## 공간 구조

최종 실행은 `data/population_by_region_age_detailed.csv`를 사용합니다.
이 파일은 총 50개 상태 지역을 포함합니다.

- 서울: 25개 구
- 원주: 25개 읍면동

다만 지역 접촉행렬의 원천 데이터는 `Region_A / Region_B` 수준의 2x2 coarse matrix입니다.
상세 population을 사용할 경우 코드는 이 2x2 행렬을 세부 지역의 population 비율에 따라 50x50 행렬로 확장합니다.

따라서 현재 모델은 실측 50x50 이동행렬 모델이 아니라, 50개 지역 상태공간 위에 2개 상위 그룹 접촉행렬을 population 비례로 확장해 적용한 모델입니다.

## 실행 환경 준비

아래 명령은 Windows PowerShell 기준입니다.

```powershell
cd C:\Users\dhkdd\Desktop\FLU_MODEL\flu-model
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PowerShell에서 가상환경 활성화가 막히면 현재 터미널에서만 실행 정책을 완화한 뒤 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

macOS 또는 Linux에서는 아래처럼 준비합니다.

```bash
cd /path/to/flu-model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

가상환경이 활성화되면 프롬프트 앞에 `(.venv)`가 표시됩니다.

## 실행 방법

가장 간단한 최종 실행:

```powershell
python main.py
```

명시적으로 config를 지정하는 최종 비교 실행:

```powershell
python -m src.cli --config configs\calendar_dec_feb_shape_detailed_compare_initial_conditions.yaml
```

그림 생성 없이 빠르게 확인:

```powershell
python -m src.cli --config configs\calendar_dec_feb_shape_detailed_compare_initial_conditions.yaml --run-name quick_initial_condition_check --no-plots
```

테스트 실행:

```powershell
python -m pytest
```

## 최종 결과 위치

최종 결과 폴더:

- `results/final_result_initial_condition_comparison`

주요 구조:

- `same_prevalence/`: 동일 초기 유병률 방식의 전체 결과
- `equal_absolute_seed/`: 동일 절대 seed 방식의 전체 결과
- `tables/initial_condition_overall_summary.csv`: 두 초기조건의 전체 요약 비교
- `tables/region_group_final_summary_by_initial_condition.csv`: 초기조건별 서울/원주 비교
- `tables/age_group_summary_by_initial_condition.csv`: 초기조건별 연령집단 비교
- `meta/batch_config_used.yaml`: 비교 실행에 사용된 설정
- `meta/batch_summary.json`: 하위 실행 폴더 요약

각 하위 실행 폴더에는 기존 단일 실행과 같은 표준 결과가 저장됩니다.

- `meta/config_used.yaml`
- `meta/summary_metrics.json`
- `tables/overall_daily_metrics.csv`
- `tables/region_group_final_summary.csv`
- `tables/age_group_summary.csv`
- `plots/region_comparison.png`
- `plots/growth_summary.png`
- `plots/seasonality_comparison.png`

## 해석 주의사항

- `same_prevalence`는 "동일 인구"가 아니라 "동일 초기 유병률" 방식입니다.
- `equal_absolute_seed`는 "동일 절대 seed" 방식입니다.
- `cross_region_flow_share`는 50지역 실행에서 자기 세부 지역 내부를 제외한 모든 세부 지역 간 감염 기여 비율입니다. 서울-원주 간 flow만 의미하지 않습니다.
- `no_cross_region`은 최종 실험에 포함하지 않습니다. 나중에 사용할 경우 서울-원주 간 혼합만 제거하는 것이 아니라 모든 세부 지역 간 off-diagonal 혼합을 제거합니다.
- `susceptibility_from_csv`와 calibration 기능은 코드에 남아 있지만, 최종 비교 설정에서는 `susceptibility_from_csv.enabled: false`입니다.
- 입력 자료와 target 자료는 수업 프로젝트용 synthetic/demo 자료이므로 실제 감시자료 분석 결과처럼 해석하면 안 됩니다.

## 주요 파일

- `main.py`: 최종 비교 실행을 기본값으로 연결한 진입점
- `src/cli.py`: CLI 실행 진입점
- `src/config.py`: 설정 로딩과 검증
- `src/data_loader.py`: 인구, 접촉행렬, calendar 로딩
- `src/model.py`: 감염압력과 상태 전이 수식
- `src/simulation.py`: 일 단위 시뮬레이션 루프
- `src/initial_condition_batch.py`: 최종 두 초기조건 비교 실행
- `src/metrics.py`: CSV/JSON 결과 생성
- `src/plotting.py`: 그림 생성
- `configs/calendar_dec_feb_shape_detailed_compare_initial_conditions.yaml`: 최종 비교 설정

## 의존성

주요 Python 패키지는 `requirements.txt`에 정리되어 있습니다.

- `numpy`
- `pandas`
- `PyYAML`
- `Pillow`
- `pytest`
