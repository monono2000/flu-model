import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================
# 0. 기본 설정
# ======================================
AGE_LABELS = ["0-18", "19-49", "50-64", "65+"]
SCENARIO_NAMES = {
    "I": "Semester Weekday",
    "II": "Vacation Weekday",
    "IV": "Lunar New Year"
}

DAYS = 180

# 질병 파라미터
LATENT_DAYS = 2.0
INFECTIOUS_DAYS = 5.0
SIGMA = 1.0 / LATENT_DAYS
GAMMA = 1.0 / INFECTIOUS_DAYS

# 전파계수
BETA = {
    "I": 0.040,
    "II": 0.055,
    "IV": 0.060
}

# 초기 감염자 수: 도시와 무관하게 연령집단별 동일 절대값
# 순서: 0-18 / 19-49 / 50-64 / 65+
INITIAL_INFECTED = np.array([100.0, 100.0, 100.0, 100.0], dtype=float)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ======================================
# 1. 도시별 인구 벡터
# 순서: 0-18 / 19-49 / 50-64 / 65+
# ======================================
N_SEOUL = np.array([1098911, 4151588, 2134796, 1918834], dtype=float)
N_WONJU = np.array([54787, 142180, 92214, 74488], dtype=float)

CITY_POP = {
    "Seoul": N_SEOUL,
    "Wonju": N_WONJU
}

# ======================================
# 2. 기간별 연령 접촉행렬 A
# 행 = participant age
# 열 = contact age
# ======================================
A_I = np.array([
    [5.844, 2.461, 0.528, 0.109],
    [0.878, 2.490, 1.047, 0.411],
    [0.312, 1.733, 1.902, 0.950],
    [0.087, 0.926, 1.292, 2.771]
], dtype=float)

A_II = np.array([
    [3.879, 2.257, 0.447, 0.127],
    [0.805, 2.111, 0.982, 0.434],
    [0.264, 1.626, 1.802, 0.967],
    [0.102, 0.977, 1.315, 2.950]
], dtype=float)

A_IV = np.array([
    [1.555, 2.533, 0.644, 1.042],
    [0.904, 1.414, 0.869, 0.735],
    [0.381, 1.439, 1.523, 0.743],
    [0.837, 1.655, 1.010, 1.912]
], dtype=float)

AGE_MATRICES = {
    "I": A_I,
    "II": A_II,
    "IV": A_IV
}

# ======================================
# 3. 기간별 2지역 residential matrix W
# 행 = contact region [Seoul, Wonju]
# 열 = participant region [Seoul, Wonju]
# Seoul = Metropolitan
# Wonju = Gangwon
# ======================================
W_I = np.array([
    [4.697, 0.124],
    [0.005, 4.382]
], dtype=float)

W_II = np.array([
    [4.066, 0.273],
    [0.004, 4.222]
], dtype=float)

W_IV = np.array([
    [3.381, 0.822],
    [0.063, 3.087]
], dtype=float)

REGION_MATRICES = {
    "I": W_I,
    "II": W_II,
    "IV": W_IV
}

# ======================================
# 4. 보조 함수
# ======================================
def normalize_rows(mat: np.ndarray) -> np.ndarray:
    row_sums = mat.sum(axis=1, keepdims=True)
    return np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

def normalize_cols(mat: np.ndarray) -> np.ndarray:
    col_sums = mat.sum(axis=0, keepdims=True)
    return np.divide(mat, col_sums, out=np.zeros_like(mat), where=col_sums != 0)

def make_initial_state(N: np.ndarray, initial_infected: np.ndarray = INITIAL_INFECTED):
    # 혹시 인구보다 초기감염자가 더 큰 경우를 막기 위해 clip
    I0 = np.minimum(initial_infected, N * 0.99).astype(float)
    E0 = np.zeros_like(N)
    R0 = np.zeros_like(N)
    S0 = N - I0
    return S0, E0, I0, R0

# ======================================
# 5. coupled SEIR 하루 갱신
# ======================================
def step_seir_coupled(
    S_seoul, E_seoul, I_seoul, R_seoul, N_seoul,
    S_wonju, E_wonju, I_wonju, R_wonju, N_wonju,
    A_period, W_region, beta, sigma=SIGMA, gamma=GAMMA
):
    # 연령별 총 접촉수
    c_age = A_period.sum(axis=1)      # shape (4,)
    # 연령 혼합 분포
    P_age = normalize_rows(A_period)  # shape (4,4)
    # 지역 혼합 분포 (participant region 기준)
    Q_region = normalize_cols(W_region)  # shape (2,2)

    # 연령별 유병률
    prev_seoul = I_seoul / N_seoul
    prev_wonju = I_wonju / N_wonju

    # 서울 참가자가 만나는 contact region 혼합
    mixed_prev_seoul = (
        Q_region[0, 0] * prev_seoul +   # Seoul contact x Seoul participant
        Q_region[1, 0] * prev_wonju     # Wonju contact x Seoul participant
    )

    # 원주 참가자가 만나는 contact region 혼합
    mixed_prev_wonju = (
        Q_region[0, 1] * prev_seoul +   # Seoul contact x Wonju participant
        Q_region[1, 1] * prev_wonju     # Wonju contact x Wonju participant
    )

    # 연령 혼합 반영
    age_mixed_prev_seoul = P_age @ mixed_prev_seoul
    age_mixed_prev_wonju = P_age @ mixed_prev_wonju

    # 감염압력
    force_seoul = beta * c_age * age_mixed_prev_seoul
    force_wonju = beta * c_age * age_mixed_prev_wonju

    p_inf_seoul = 1.0 - np.exp(-force_seoul)
    p_inf_wonju = 1.0 - np.exp(-force_wonju)

    newE_seoul = S_seoul * p_inf_seoul
    newE_wonju = S_wonju * p_inf_wonju

    newI_seoul = sigma * E_seoul
    newI_wonju = sigma * E_wonju

    newR_seoul = gamma * I_seoul
    newR_wonju = gamma * I_wonju

    S_seoul_next = S_seoul - newE_seoul
    E_seoul_next = E_seoul + newE_seoul - newI_seoul
    I_seoul_next = I_seoul + newI_seoul - newR_seoul
    R_seoul_next = R_seoul + newR_seoul

    S_wonju_next = S_wonju - newE_wonju
    E_wonju_next = E_wonju + newE_wonju - newI_wonju
    I_wonju_next = I_wonju + newI_wonju - newR_wonju
    R_wonju_next = R_wonju + newR_wonju

    return (
        S_seoul_next, E_seoul_next, I_seoul_next, R_seoul_next,
        S_wonju_next, E_wonju_next, I_wonju_next, R_wonju_next
    )

# ======================================
# 6. 시뮬레이션 실행
# ======================================
def run_simulation_coupled(
    N_seoul, N_wonju,
    A_period, W_region, beta,
    days=DAYS
):
    S_seoul, E_seoul, I_seoul, R_seoul = make_initial_state(N_seoul)
    S_wonju, E_wonju, I_wonju, R_wonju = make_initial_state(N_wonju)

    history = {
        "Seoul": {"S": [], "E": [], "I": [], "R": [], "N": N_seoul.copy()},
        "Wonju": {"S": [], "E": [], "I": [], "R": [], "N": N_wonju.copy()}
    }

    for city, S, E, I, R in [
        ("Seoul", S_seoul, E_seoul, I_seoul, R_seoul),
        ("Wonju", S_wonju, E_wonju, I_wonju, R_wonju)
    ]:
        history[city]["S"].append(S.copy())
        history[city]["E"].append(E.copy())
        history[city]["I"].append(I.copy())
        history[city]["R"].append(R.copy())

    for _ in range(days):
        (
            S_seoul, E_seoul, I_seoul, R_seoul,
            S_wonju, E_wonju, I_wonju, R_wonju
        ) = step_seir_coupled(
            S_seoul, E_seoul, I_seoul, R_seoul, N_seoul,
            S_wonju, E_wonju, I_wonju, R_wonju, N_wonju,
            A_period, W_region, beta
        )

        for city, S, E, I, R in [
            ("Seoul", S_seoul, E_seoul, I_seoul, R_seoul),
            ("Wonju", S_wonju, E_wonju, I_wonju, R_wonju)
        ]:
            history[city]["S"].append(S.copy())
            history[city]["E"].append(E.copy())
            history[city]["I"].append(I.copy())
            history[city]["R"].append(R.copy())

    for city in ["Seoul", "Wonju"]:
        history[city]["S"] = np.array(history[city]["S"])
        history[city]["E"] = np.array(history[city]["E"])
        history[city]["I"] = np.array(history[city]["I"])
        history[city]["R"] = np.array(history[city]["R"])

    return history

# ======================================
# 7. 요약 지표 계산
# ======================================
def summarize_results(result: dict):
    S = result["S"]
    I = result["I"]
    N = result["N"]

    total_pop = N.sum()
    total_I = I.sum(axis=1)

    peak_day = int(np.argmax(total_I))
    peak_I_per_100k = float(total_I.max() / total_pop * 100000.0)

    cumulative_infected_by_age = N - S[-1]
    cumulative_infected_total = float(cumulative_infected_by_age.sum())
    cumulative_rate_total = float(cumulative_infected_total / total_pop * 100.0)
    cumulative_rate_by_age = cumulative_infected_by_age / N * 100.0

    elderly_share = 0.0
    if cumulative_infected_total > 0:
        elderly_share = float(cumulative_infected_by_age[-1] / cumulative_infected_total * 100.0)

    return {
        "peak_day": peak_day,
        "peak_I_per_100k": peak_I_per_100k,
        "cumulative_rate_total": cumulative_rate_total,
        "cumulative_rate_by_age": cumulative_rate_by_age,
        "elderly_share": elderly_share
    }

# ======================================
# 8. 표 생성
# ======================================
def build_summary_table(all_results: dict):
    rows = []

    for city_name, scenario_results in all_results.items():
        for scenario_code, result in scenario_results.items():
            summary = summarize_results(result)

            rows.append({
                "City": city_name,
                "Scenario": scenario_code,
                "Scenario_Name": SCENARIO_NAMES[scenario_code],
                "Peak_Day": summary["peak_day"],
                "Peak_I_per_100k": round(summary["peak_I_per_100k"], 2),
                "Cumulative_Rate_%": round(summary["cumulative_rate_total"], 2),
                "Elderly_Share_%": round(summary["elderly_share"], 2),
                "Age_0_18_%": round(summary["cumulative_rate_by_age"][0], 2),
                "Age_19_49_%": round(summary["cumulative_rate_by_age"][1], 2),
                "Age_50_64_%": round(summary["cumulative_rate_by_age"][2], 2),
                "Age_65plus_%": round(summary["cumulative_rate_by_age"][3], 2),
            })

    return pd.DataFrame(rows)

# ======================================
# 9. 그래프
# ======================================
def plot_total_infectious(all_results: dict):
    days = np.arange(DAYS + 1)

    for scenario_code in ["I", "II", "IV"]:
        plt.figure(figsize=(10, 6))

        for city_name in ["Seoul", "Wonju"]:
            result = all_results[city_name][scenario_code]
            total_I_per_100k = result["I"].sum(axis=1) / result["N"].sum() * 100000.0
            plt.plot(days, total_I_per_100k, label=city_name)

        plt.xlabel("Day")
        plt.ylabel("Infectious per 100,000")
        plt.title(f"{SCENARIO_NAMES[scenario_code]}: Seoul vs Wonju")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"total_infectious_{scenario_code}.png", dpi=200)
        plt.show()

def plot_age_cumulative_bar(all_results: dict):
    for scenario_code in ["I", "II", "IV"]:
        x = np.arange(len(AGE_LABELS))
        width = 0.35

        seoul_summary = summarize_results(all_results["Seoul"][scenario_code])
        wonju_summary = summarize_results(all_results["Wonju"][scenario_code])

        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, seoul_summary["cumulative_rate_by_age"], width=width, label="Seoul")
        plt.bar(x + width / 2, wonju_summary["cumulative_rate_by_age"], width=width, label="Wonju")

        plt.xticks(x, AGE_LABELS)
        plt.ylabel("Cumulative infection rate (%)")
        plt.title(f"{SCENARIO_NAMES[scenario_code]}: Age-specific cumulative infection rate")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"age_cumulative_{scenario_code}.png", dpi=200)
        plt.show()

# ======================================
# 10. 행렬 저장
# ======================================
def save_matrices():
    for scenario_code in ["I", "II", "IV"]:
        pd.DataFrame(
            AGE_MATRICES[scenario_code],
            index=AGE_LABELS,
            columns=AGE_LABELS
        ).to_csv(OUTPUT_DIR / f"age_matrix_{scenario_code}.csv", encoding="utf-8-sig")

        pd.DataFrame(
            REGION_MATRICES[scenario_code],
            index=["Seoul_contact", "Wonju_contact"],
            columns=["Seoul_participant", "Wonju_participant"]
        ).to_csv(OUTPUT_DIR / f"region_matrix_{scenario_code}.csv", encoding="utf-8-sig")

# ======================================
# 11. 메인 실행
# ======================================
def main():
    all_results = {
        "Seoul": {},
        "Wonju": {}
    }

    for scenario_code in ["I", "II", "IV"]:
        history = run_simulation_coupled(
            N_SEOUL, N_WONJU,
            AGE_MATRICES[scenario_code],
            REGION_MATRICES[scenario_code],
            BETA[scenario_code]
        )

        all_results["Seoul"][scenario_code] = history["Seoul"]
        all_results["Wonju"][scenario_code] = history["Wonju"]

    summary_df = build_summary_table(all_results)

    print("\n===== Summary Table =====")
    print(summary_df)

    summary_df.to_csv(
        OUTPUT_DIR / "summary_table.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\n저장 완료: {OUTPUT_DIR / 'summary_table.csv'}")

    save_matrices()
    plot_total_infectious(all_results)
    plot_age_cumulative_bar(all_results)

if __name__ == "__main__":
    main()