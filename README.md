# FLU Model

## Submission Note

- This repository is configured for course-project and demo use with synthetic inputs.
- `data/synthetic_target_influenza_2025_2026.csv` is a manually prepared synthetic target series, not official surveillance data.
- Region labels `Region_A` and `Region_B` are placeholders for a two-region toy model and should not be interpreted as real measured locations.
- Some legacy filenames still contain `observed` or `actualfit` for backward compatibility. For submission, prefer the `synthetic` and `demo_fit` config names listed in [DATA_SYNTHETIC_NOTE.md](DATA_SYNTHETIC_NOTE.md).

?쒖슱?밸퀎?쒖? ?먯＜?쒕? 鍮꾧탳?섎뒗 2吏??4?곕졊 寃곗젙濡좎쟻 李⑤텇??媛먯뿼蹂?紐⑤뜽 ??μ냼??  
湲곗〈 ?⑥씪 `main.py` 湲곕컲 coupled SEIR瑜??ㅼ젙/?곗씠??寃곌낵 ???援ъ“濡?遺꾨━?섍퀬, ?ㅼ쓬 湲곕뒫??異붽??덈떎.

- ?쒓컙媛蹂 regime: `I`, `II`, `III`, `IV`
- ?곹깭 ?뺤옣: `S0`, `E0`, `I0`, `R`, `S1`, `E1`, `I1`
- ?ш컧??異붿쟻, ?곕졊蹂?媛먯뿼 誘쇨컧?? regime蹂?beta multiplier
- `fixed regime mode`? `calendar mode`
- ?쇰퀎 source?뭪arget 媛먯뿼 ?먮쫫 ???- 湲곗〈 1李??ы쁽??legacy batch ?ㅽ뻾
- pytest 湲곕컲 蹂댁〈??flow ?쇨????뚯뒪??
## ?붾젆?곕━ 援ъ“

- `src/`: ?ㅼ젙 濡쒕뱶, ?곗씠??濡쒕뱶, 紐⑤뜽, ?쒕??덉씠?? 吏?? 洹몃┝, CLI
- `configs/`: 湲곕낯/legacy/誘쇨컧??counterfactual ?덉떆 ?ㅼ젙
- `data/`: ?멸뎄? ?묒큺?됰젹 CSV, winter calendar
- `results/{run_name}/`: 援ъ“?붾맂 ?ㅽ뻾 寃곌낵
- `outputs/`: `python main.py` legacy ?명솚 異쒕젰
- `tests/`: pytest ?뚯뒪??
## ?낅젰 ?뚯씪

湲곕낯 ?낅젰 ?뚯씪? `data/` 湲곗??대떎.

- `population_by_region_age.csv`
- `age_contact_period_I.csv`
- `age_contact_period_II.csv`
- `age_contact_period_III.csv`
- `age_contact_period_IV.csv`
- `region_contact_period_I.csv`
- `region_contact_period_II.csv`
- `region_contact_period_III.csv`
- `region_contact_period_IV.csv`
- `winter_calendar.csv` ?먮뒗 `sample_winter_calendar.csv`

`winter_calendar.csv` ?뺤떇:

```csv
date,regime
2025-12-01,I
2025-12-06,III
2025-12-25,III
2026-01-02,II
2026-02-16,IV
```

- `date`??`YYYY-MM-DD`
- `regime`??諛섎뱶??`I`, `II`, `III`, `IV`

## Period III ?됰젹 諛섏쁺

Period III(二쇰쭚) ?곕졊?됰젹怨?吏??뻾?ъ쓣 ?ㅼ젣 媛믪쑝濡?諛섏쁺?덈떎.

- ?곕졊?됰젹 ?대?吏??`??contact`, `??participant` ?뺥깭?湲??뚮Ц?? 肄붾뱶 ?뺤쓽 `A[a,b] = participant age a媛 contact age b瑜?留뚮굹??援ъ“`??留욊쾶 ?꾩튂?댁꽌 ??ν뻽??
- 吏??뻾?ъ? ?꾧뎅 沅뚯뿭 洹몃┝?먯꽌 `Metropolitan ??Gangwon` 釉붾줉留?異붿텧??2吏??紐⑤뜽??`Region_A ??Region_B` ?됰젹濡??ъ슜?쒕떎.
- 利?`Region_A = Metropolitan`, `Region_B = Gangwon`?쇰줈 留ㅽ븨?덈떎.

## Winter Calendar 媛??
?꾩옱 ??μ냼?먮뒗 [data/winter_calendar.csv](/abs/path/c:/Users/dhkdd/Desktop/FLU_MODEL/flu-model/data/winter_calendar.csv)? [data/sample_winter_calendar.csv](/abs/path/c:/Users/dhkdd/Desktop/FLU_MODEL/flu-model/data/sample_winter_calendar.csv)瑜??④퍡 ?ｌ뼱 ?먯뿀??

??baseline calendar???ㅼ쓬 洹쒖튃?쇰줈 留뚮뱾?덈떎.

- 湲곌컙: `2025-12-01` ~ `2026-02-28`
- ???? `III`
- `2025-12-25`, `2026-01-01`: `III`
- `2025-12-29` ~ `2026-02-27` ?됱씪: `II`
- `2026-02-16` ~ `2026-02-18` ???고쑕: `IV`
- ??議곌굔???대떦?섏? ?딅뒗 `2025-12-01` ~ `2025-12-26` ?됱씪: `I`

利????뚯씪? "二쇰쭚? 怨꾩궛?쇰줈 ?뺤젙, ???고쑕???ㅼ젣 ?좎쭨 諛섏쁺, 諛⑺븰 ?됱씪? ?곌뎄??媛???대씪???깃꺽??baseline calendar??  
?쒖슱/?먯＜ ?ㅼ젣 ?숈궗?쇱젙 湲곗? 遺꾩꽍???섎젮硫????뚯씪??吏곸젒 援먯껜?섎㈃ ?쒕떎.

`winter_calendar.csv`媛 ?놁쑝硫??ㅽ뻾湲곕뒗 `data/sample_winter_calendar.csv`瑜??덈궡?쒕떎.

異붽?濡?calibration ?ㅽ뿕??calendar???ы븿?섏뼱 ?덈떎.

- [data/winter_calendar_preseason.csv](/abs/path/c:/Users/dhkdd/Desktop/FLU_MODEL/flu-model/data/winter_calendar_preseason.csv): `2025-11-03` ?쒖옉
- [data/winter_calendar_fullseason.csv](/abs/path/c:/Users/dhkdd/Desktop/FLU_MODEL/flu-model/data/winter_calendar_fullseason.csv): `2025-09-01` ?쒖옉

## 愿痢?湲곕컲 蹂댁젙

愿痢?二쇨컙 諛쒕퀝瑜?CSV瑜?紐⑤뜽 ?곕졊吏묐떒?쇰줈 蹂?섑븯怨? calendar baseline?????媛꾨떒??grid calibration???섑뻾?섎뒗 ?ㅽ겕由쏀듃瑜?異붽??덈떎.

- 愿痢??뚯씪 ?덉떆: [data/synthetic_target_influenza_2025_2026.csv](/abs/path/c:/Users/dhkdd/Desktop/FLU_MODEL/flu-model/data/synthetic_target_influenza_2025_2026.csv)
- 蹂댁젙 ?ㅽ뻾湲? [src/calibrate_cli.py](/abs/path/c:/Users/dhkdd/Desktop/FLU_MODEL/flu-model/src/calibrate_cli.py)
- 蹂댁젙 濡쒖쭅: [src/calibration.py](/abs/path/c:/Users/dhkdd/Desktop/FLU_MODEL/flu-model/src/calibration.py)
- 愿痢??뚯꽌: [src/observations.py](/abs/path/c:/Users/dhkdd/Desktop/FLU_MODEL/flu-model/src/observations.py)

媛?뺤? ?ㅼ쓬怨?媛숇떎.

- 愿痢?CSV??33媛?二쇨컙 媛믪? `2025-W36` ~ `2026-W16`濡??댁꽍
- winter 鍮꾧탳 援ш컙? `2025-12-01` ~ `2026-02-23` 二쇱감
- `0-18`? `0??, `1-6??, `7-12??, `13-18??瑜??곕졊??`1:6:6:6`?쇰줈 媛以묓룊洹?- 愿痢≪튂??surveillance rate, 紐⑤뜽? simulated infection episode rate?대?濡??덈? ?섏?蹂대떎??二쇨컙 shape 鍮꾧탳??臾닿쾶瑜???
?ㅽ뻾 ?덉떆:

```bash
python -m src.calibrate_cli --config configs/calendar_baseline.yaml --target-csv data/synthetic_target_influenza_2025_2026.csv --run-name synthetic_target_calibration
python -m src.cli --config configs/calendar_synthetic_calibrated.yaml --mode calendar --run-name winter_calendar_synthetic_calibrated
```

鍮꾧탳 援ш컙??諛붽씀?ㅻ㈃:

```bash
python -m src.calibrate_cli --config configs/calendar_preseason_baseline.yaml --target-csv data/synthetic_target_influenza_2025_2026.csv --compare-start 2025-11-03 --compare-end 2026-02-23 --run-name synthetic_target_calibration_preseason
python -m src.calibrate_cli --config configs/calendar_fullseason_baseline.yaml --target-csv data/synthetic_target_influenza_2025_2026.csv --compare-start 2025-09-01 --compare-end 2026-02-23 --run-name synthetic_target_calibration_fullseason
```

?꾩옱 愿痢?湲곕컲 ?꾨낫 ?ㅼ젙? [configs/calendar_synthetic_calibrated.yaml](/abs/path/c:/Users/dhkdd/Desktop/FLU_MODEL/flu-model/configs/calendar_synthetic_calibrated.yaml)????ν빐 ?먯뿀??

## 洹몃┝ ?앹꽦 諛깆뿏??
????μ냼??湲곕낯 PNG/GIF 異쒕젰? Pillow 湲곕컲?쇰줈 援ы쁽?덈떎.  
?쇰? Windows ?섍꼍?먯꽌??`matplotlib` ?ㅼ씠?곕툕 DLL 濡쒕뵫???뺤콉?쇰줈 李⑤떒?????덉뼱, ?곌뎄??湲곕낯 洹몃┝? ?쒖닔 Python ?대?吏 ?앹꽦 寃쎈줈濡??좎??덈떎.

## ?ㅽ뻾 諛⑸쾿

湲곕낯 fixed regime ?ㅽ뻾:

```bash
python -m src.cli --config configs/default.yaml --mode fixed --regime I --days 180 --run-name fixed_I_180
```

calendar ?ㅽ뻾:

```bash
python -m src.cli --config configs/calendar_baseline.yaml --mode calendar --run-name winter_calendar_run
```

legacy 1李?鍮꾧탳???ㅽ뻾:

```bash
python main.py
```

?먮뒗 吏곸젒:

```bash
python -m src.cli --config configs/legacy.yaml --mode legacy_batch
```

counterfactual ?덉떆:

```bash
python -m src.cli --config configs/counterfactual_no_holiday.yaml --mode calendar
python -m src.cli --config configs/counterfactual_no_cross_region.yaml --mode fixed --regime II
python -m src.cli --config configs/counterfactual_no_reinfection.yaml --mode fixed --regime II
python -m src.cli --config configs/legacy_period_beta.yaml --mode fixed --regime IV
```

## 寃곌낵 ?뚯씪

援ъ“?붾맂 ?ㅽ뻾 寃곌낵??`results/{run_name}/` ?꾨옒????λ맂??

- `states_long.csv`
- `node_daily_metrics.csv`
- `overall_daily_metrics.csv`
- `flow_long.csv`
- `summary_metrics.json`
- `config_used.yaml`
- `timeseries_overview.png`
- `region_comparison.png`
- `age_group_comparison.png`
- `regime_timeline.png`
- `network_snapshots/`
- `network_animation.gif`

legacy batch 寃곌낵??`outputs/` ?꾨옒????λ맂??

- `summary_table.csv`
- `total_infectious_I.png`, `total_infectious_II.png`, `total_infectious_IV.png`
- `age_cumulative_I.png`, `age_cumulative_II.png`, `age_cumulative_IV.png`
- `age_matrix_*.csv`, `region_matrix_*.csv`

## 吏???뺤쓽

- `ever_infected_rate`: 理쒖냼 1???댁긽 媛먯뿼 鍮꾩쑉, `1 - S0 / N`
- `infection_episode_rate`: ?꾩쟻 泥?媛먯뿼 + ?꾩쟻 ?ш컧???먰뵾?뚮뱶 / `N`
- `cross_region_flow_share`: ?뱀씪 ?꾩껜 媛먯뿼 ?먮쫫 以?吏??媛??먮쫫 鍮꾩쨷
- `elderly_burden_summary`: 65+ 吏묐떒???쇳겕/?꾩쟻 遺???붿빟

## ?뚯뒪??
```bash
python -m pytest
```

?ы븿???뚯뒪??

- 珥앹씤援?蹂댁〈
- 珥덇린 媛먯뿼 0 ?덉젙??- no cross region ??吏??媛?flow 0
- fixed/calendar ?쇱튂??- no reinfection 異뺤빟
- source-target flow ???쇱튂??- seoul seed + no cross region 寃⑸━ sanity check

