import streamlit as st
import pandas as pd


st.markdown("""
<style>
[data-testid="stSidebarNav"] ul li:nth-child(4) {
    display: none;
}
</style>
""", unsafe_allow_html=True)
st.title("🍱 식단표")
st.divider()
st.subheader("오늘의 식단을 확인해보세요 ✅")

df = pd.read_csv("meal_plan.csv")

# 메뉴 합치기
df["메뉴"] = df[["밥", "국", "반찬1", "반찬2", "반찬3"]] \
    .apply(lambda x: "\n".join(x.dropna()), axis=1)

# 카드 함수
def meal_card(day, menu):
    return f"""
    <div style="
        background-color:#ffffff;
        padding:12px;
        border-radius:15px;
        box-shadow:2px 2px 8px rgba(0,0,0,0.1);
        min-height:220px;
        border-left:5px solid #4CAF50;
        margin-bottom:15px;
    ">
        <h5>📅 {day}일</h5>
        {menu.replace(chr(10), "<br>")}
    </div>
    """

for i in range(0, len(df), 4):
    cols = st.columns(4)
    chunk = df.iloc[i:i+4]

    for j in range(len(chunk)):
        with cols[j]:
            row = chunk.iloc[j]
            st.markdown(meal_card(row["날짜"], row["메뉴"]), unsafe_allow_html=True)