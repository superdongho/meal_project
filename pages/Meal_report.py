"""급식 데이터 분석 & AI 보고서 페이지."""

import os

import streamlit as st
from dotenv import load_dotenv

import meal_data_manager

load_dotenv()

# ─── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(page_title="급식 분석 & AI 보고서", layout="wide")

st.title("🍱 급식 데이터 분석 & AI 보고서")
st.write("monthly_data.json에 저장된 급식 기록을 분석하고 AI가 보고서를 생성합니다.")

# ─── 데이터 로드 ─────────────────────────────────────────────
try:
    df_all = meal_data_manager.get_all_df()
    data_ok = not df_all.empty
except FileNotFoundError:
    df_all = None
    data_ok = False

# ═══════════════════════════════════════════════════════════
# 1. 데이터 현황
# ═══════════════════════════════════════════════════════════
st.subheader("📋 일자별 급식 기록")

if not data_ok:
    st.info("아직 기록된 급식 데이터가 없습니다. monthly_data.json을 확인해주세요.")
else:
    # 조회 일수 슬라이더
    total_days = len(df_all)
    view_days = st.slider(
        "최근 몇 일 데이터를 볼까요?",
        min_value=5,
        max_value=total_days,
        value=min(14, total_days),
        step=1,
    )
    df_view = meal_data_manager.get_recent_days(view_days)

    # ─ 요약 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📅 총 데이터", f"{total_days}일")
    col2.metric("🔥 평균 칼로리", f"{df_view['total_kcal'].mean():.0f} kcal")
    col3.metric("💪 평균 단백질", f"{df_view['total_protein'].mean():.1f} g")
    col4.metric("🌾 평균 탄수화물", f"{df_view['total_carbs'].mean():.1f} g")

    st.divider()

    # ─ 데이터 테이블
    display_df = df_view.copy()
    display_df["날짜"] = display_df["날짜"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        display_df[["날짜", "total_kcal", "total_carbs", "total_protein", "total_fat"]].rename(
            columns={
                "total_kcal": "칼로리(kcal)",
                "total_carbs": "탄수화물(g)",
                "total_protein": "단백질(g)",
                "total_fat": "지방(g)",
            }
        ),
        use_container_width=True,
    )

    # CSV 다운로드
    csv_bytes = display_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 CSV 다운로드",
        data=csv_bytes,
        file_name="meal_data.csv",
        mime="text/csv",
    )

    # ═══════════════════════════════════════════════════════
    # 2. 기본 차트 (Streamlit 내장 차트)
    # ═══════════════════════════════════════════════════════
    st.divider()
    st.subheader("📈 영양소 추이 차트")

    chart_df = df_view.copy()
    chart_df.index = chart_df["날짜"].dt.strftime("%m/%d")

    tab1, tab2, tab3 = st.tabs(["칼로리 추이", "영양소 구성", "음식별 섭취율"])

    with tab1:
        st.write("날짜별 총 칼로리 섭취량 추이입니다.")
        st.line_chart(chart_df[["total_kcal"]], height=300)

    with tab2:
        st.write("날짜별 탄수화물·단백질·지방 구성입니다.")
        st.bar_chart(
            chart_df[["total_carbs", "total_protein", "total_fat"]],
            height=300,
        )

    with tab3:
        st.write("전체 기간 음식별 평균 섭취율 순위입니다.")
        food_df = meal_data_manager.get_food_intake_df()
        food_df = food_df.set_index("food_name")
        st.bar_chart(food_df[["avg_intake_rate"]], height=350)


# ═══════════════════════════════════════════════════════════
# 3. AI 급식 분석 보고서
# ═══════════════════════════════════════════════════════════
st.divider()
st.subheader("🤖 AI 급식 분석 보고서 생성")
st.write("AI 영양사가 급식 데이터를 분석하고 Word 보고서와 차트를 자동으로 만들어 드립니다.")

# 분석 기간 선택
report_days = st.selectbox(
    "분석 기간 선택",
    options=[7, 14, 21, 31],
    format_func=lambda x: f"최근 {x}일",
    index=0,
)

# 추가 요청사항 입력
extra_request = st.text_input(
    "추가 요청사항 (선택)",
    placeholder="예: 단백질 섭취 부족한 요일을 집중적으로 분석해줘",
)

if st.button("📊 AI 보고서 생성", type="primary", disabled=not data_ok):
    with st.spinner("AI 영양사가 급식 데이터를 분석 중입니다... (약 30~60초)"):
        try:
            from llm_util import MealAgentBot

            bot = MealAgentBot()

            prompt = (
                f"최근 {report_days}일간의 급식 데이터를 분석하고 "
                "칼로리 추이 차트, 영양소 구성 차트, 음식별 섭취율 차트를 만든 뒤 "
                "급식 분석 보고서를 Word 파일로 작성해주세요."
            )
            if extra_request.strip():
                prompt += f" 추가로 다음 사항도 반영해주세요: {extra_request.strip()}"

            result = bot.chat(prompt)

            # WORD_PATH 마커 분리
            display_text = result
            word_path = None
            if "WORD_PATH:" in result:
                parts = result.split("WORD_PATH:")
                display_text = parts[0].strip()
                word_path = parts[1].strip()

            st.success("✅ 보고서가 완성되었습니다!")

            if display_text:
                st.markdown(display_text)

            # Word 파일 다운로드 버튼 (즉시 제공)
            if word_path and os.path.exists(word_path):
                with open(word_path, "rb") as f:
                    st.download_button(
                        label="📄 Word 보고서 다운로드",
                        data=f.read(),
                        file_name=os.path.basename(word_path),
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
            else:
                st.info("보고서 관리 페이지에서 Word 파일을 다운로드할 수 있습니다.")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
            st.info("OPENAI_API_KEY가 .env 파일에 올바르게 설정되어 있는지 확인해주세요.")

elif not data_ok:
    st.warning("급식 데이터가 없어 보고서를 생성할 수 없습니다.")

# ═══════════════════════════════════════════════════════════
# 4. 생성된 보고서 목록
# ═══════════════════════════════════════════════════════════
st.divider()
st.subheader("📁 생성된 급식 보고서 목록")

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

meal_reports = sorted(
    [f for f in os.listdir(REPORT_DIR) if f.startswith("meal_report") and f.endswith(".docx")],
    reverse=True,
)

if not meal_reports:
    st.info("아직 생성된 급식 보고서가 없습니다. 위에서 AI 보고서를 생성해보세요.")
else:
    st.success(f"총 {len(meal_reports)}개의 보고서가 있습니다.")
    for i, filename in enumerate(meal_reports):
        filepath = os.path.join(REPORT_DIR, filename)
        file_size_kb = round(os.path.getsize(filepath) / 1024, 1)

        # 파일명에서 날짜 추출: meal_report_20260402_abc123.docx
        date_part = filename.replace("meal_report_", "").split("_")[0]
        if len(date_part) == 8:
            display_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
        else:
            display_date = date_part

        left, middle, right = st.columns([4, 2, 2])
        with left:
            st.markdown(f"**{filename}**")
        with middle:
            st.caption(f"작성일: {display_date} | {file_size_kb} KB")
        with right:
            with open(filepath, "rb") as f:
                st.download_button(
                    label="다운로드",
                    data=f.read(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key=f"meal_dl_{i}",
                )
        st.divider()
