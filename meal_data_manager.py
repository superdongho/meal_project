"""급식 데이터 관리 모듈.

monthly_data.json 파일을 로드하고, 급식 분석에 필요한
데이터 조회·집계 함수들을 제공합니다.
"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd

# 데이터 파일 경로
_BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(_BASE_DIR, "monthly_data.json")


def _load_raw() -> list[dict]:
    """monthly_data.json 파일을 파이썬 리스트로 로드합니다."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"급식 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_df() -> pd.DataFrame:
    """전체 급식 데이터를 평탄화(flatten)한 DataFrame을 반환합니다.

    컬럼: 날짜, total_kcal, total_carbs, total_protein, total_fat,
           rice, soup, side1, side2, side3 (음식명),
           rice_rate, soup_rate, side1_rate, side2_rate, side3_rate (섭취율)
    """
    records = _load_raw()
    rows = []
    for rec in records:
        row = {
            "날짜": pd.to_datetime(rec["date"]),
            "total_kcal": rec["calculated_nutrition"]["total_kcal"],
            "total_carbs": rec["calculated_nutrition"]["total_carbs"],
            "total_protein": rec["calculated_nutrition"]["total_protein"],
            "total_fat": rec["calculated_nutrition"]["total_fat"],
        }
        for slot, info in rec.get("food_results", {}).items():
            row[f"{slot}_name"] = info.get("name", "")
            row[f"{slot}_rate"] = info.get("intake_rate", 0)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.sort_values("날짜", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_recent_days(days: int = 7) -> pd.DataFrame:
    """최근 N일간의 데이터만 필터링하여 반환합니다."""
    df = get_all_df()
    cutoff = df["날짜"].max() - timedelta(days=days - 1)
    return df[df["날짜"] >= cutoff].copy()


def get_summary_text(days: int = 7) -> str:
    """최근 N일간 급식 현황을 텍스트로 요약합니다.

    에이전트가 첫 데이터 파악 단계에서 사용할 수 있습니다.
    """
    df = get_recent_days(days)
    if df.empty:
        return "급식 데이터가 없습니다."

    lines = [f"[최근 {days}일 급식 현황 요약]", f"- 총 {len(df)}일 데이터"]
    lines.append(f"- 평균 칼로리: {df['total_kcal'].mean():.0f} kcal")
    lines.append(f"- 평균 탄수화물: {df['total_carbs'].mean():.1f} g")
    lines.append(f"- 평균 단백질: {df['total_protein'].mean():.1f} g")
    lines.append(f"- 평균 지방: {df['total_fat'].mean():.1f} g")
    lines.append("")
    lines.append("[날짜별 칼로리]")
    for _, row in df.iterrows():
        lines.append(f"  {row['날짜'].strftime('%Y-%m-%d')}: {row['total_kcal']} kcal")

    return "\n".join(lines)


def get_food_intake_df() -> pd.DataFrame:
    """음식별 섭취율 평균을 계산한 DataFrame을 반환합니다.

    반환 컬럼: food_name, avg_intake_rate, count
    """
    records = _load_raw()
    food_stats: dict[str, list[int]] = {}

    for rec in records:
        for slot, info in rec.get("food_results", {}).items():
            name = info.get("name", "")
            rate = info.get("intake_rate", 0)
            if name:
                food_stats.setdefault(name, []).append(rate)

    rows = []
    for name, rates in food_stats.items():
        rows.append({
            "food_name": name,
            "avg_intake_rate": round(sum(rates) / len(rates), 1),
            "count": len(rates),
        })

    return pd.DataFrame(rows).sort_values("avg_intake_rate", ascending=False).reset_index(drop=True)
