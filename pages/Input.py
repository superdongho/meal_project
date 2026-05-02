import streamlit as st
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import cv2
import json
import os

st.markdown("""
<style>
[data-testid="stSidebarNav"] ul li:nth-child(4) {
    display: none;
}
</style>
""", unsafe_allow_html=True)

df = pd.read_csv("./meal_plan.csv")
st.title("데이터 입력 페이지")

from yolo_detector import detect_meal

def compare_food(before, after):
    result = {}
    for food in before:
        before_area = before.get(food, 0)
        after_area = after.get(food, 0)
        eaten = float(max(0, before_area - after_area))
        percent = float(eaten / before_area) if before_area > 0 else 0
        result[food] = {
            "eaten_area": eaten,
            "percent_eaten": percent
        }
    return result

# --- 0. 날짜와 식단 확인 ---
st.subheader("날짜를 선택하세요")
day = st.selectbox(" ", df["날짜"])
result = df[df["날짜"] == day]

if result.empty or result['날짜'].iloc[0] == '---':
    st.warning("유효한 날짜를 선택해주세요.")
    st.stop()

st.subheader(f"{day}일의 식단")
st.dataframe(result[["밥", "국", "반찬1", "반찬2", "반찬3"]].T)

menu_list = [result["반찬1"].iloc[0], result["반찬2"].iloc[0], result["반찬3"].iloc[0]]

# --- 1. 식전 사진 과정 ---
st.subheader("1. 배식받은 사진을 올려주세요!")
start_file = st.file_uploader("식전 사진", type=["jpg", "jpeg", "png"], key="start")

if start_file is not None:
    img1 = Image.open(start_file)
    img1_np = np.array(img1)
    
    with st.spinner("식전 사진 분석 중..."):
        area1, dishes1, annotated_img1 = detect_meal(img1_np, "./best (9).pt")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img1_np, caption="원본 식전 사진")
    with col2:
        st.image(annotated_img1, caption="분석된 식전 사진")
         
    dish_mapping = {}
    if dishes1:
        st.write("반찬(Dish)에 해당하는 1~3번 번호와 실제 반찬 이름을 매칭시켜주세요:")
        for i in range(len(dishes1)):
            unique_id = f"dish_{i+1}"
            index_to_select = i if i < len(menu_list) else 0
            dish_mapping[unique_id] = st.selectbox(f"{i+1}번 반찬:", menu_list, index=index_to_select, key=f"select_{i}")
            
        has_duplicates = len(dish_mapping.values()) != len(set(dish_mapping.values()))
        if has_duplicates:
            st.error("중복된 반찬이 선택되었습니다. 각각 고유한 반찬을 선택해주세요.")
        else:
            if st.button("저장"):
                st.session_state['meal_saved'] = True
                
            if st.session_state.get('meal_saved', False):
                st.info("맛있게 드세요!^^")
            
    else:
        st.warning("사진에서 반찬을 찾지 못했습니다.")
        
    st.divider()

    # --- 2. 식후 사진 과정 ---
    st.subheader("2. 맛있는 식사 되셨나요? 식사 후 사진을 올려주세요!")
    end_file = st.file_uploader("식후 사진", type=["jpg", "jpeg", "png"], key="end")
    
    if end_file is not None:
        img2 = Image.open(end_file)
        img2_np = np.array(img2)
        
        with st.spinner("식후 사진 분석 및 계산 중..."):
            area2, dishes2, annotated_img2 = detect_meal(img2_np, "./best (9).pt")

        col3, col4 = st.columns(2)
        with col3:
            st.image(img2_np, caption="원본 식후 사진")
        with col4:
            st.image(annotated_img2, caption="분석된 식후 사진")
             
        comparison = compare_food(area1, area2)
        
        st.divider()
        
        # --- 3. 정리 및 기록 ---
        st.subheader("3. 오늘 식사한 내용을 정리해서 보여줘")
        
        display_results = []
        for food_key, data in comparison.items():
            display_name = food_key
            if food_key in dish_mapping:
                display_name = dish_mapping[food_key]
            
            percent_val = round(data["percent_eaten"] * 100, 1)
            display_results.append(f"- **{display_name}**: {percent_val}% 섭취")
            
        for line in display_results:
            st.markdown(line)
            
        if st.button("결과를 데이터에 기록하기"):
            NUTRITION_DB = {
                "흰쌀밥": {"kcal": 300, "carbs": 65, "protein": 6, "fat": 0.5},
                "흑미밥": {"kcal": 290, "carbs": 63, "protein": 7, "fat": 1.0},
                "현미밥": {"kcal": 280, "carbs": 60, "protein": 8, "fat": 1.5},
                "볶음밥": {"kcal": 400, "carbs": 60, "protein": 8, "fat": 12}, 
                "된장국": {"kcal": 80, "carbs": 10, "protein": 4, "fat": 2},
                "미역국": {"kcal": 90, "carbs": 8, "protein": 6, "fat": 3},
                "김치찌개": {"kcal": 150, "carbs": 12, "protein": 10, "fat": 7},
                "부대찌개": {"kcal": 250, "carbs": 15, "protein": 12, "fat": 15},
                "제육볶음": {"kcal": 250, "carbs": 10, "protein": 20, "fat": 15},
                "돈까스": {"kcal": 350, "carbs": 25, "protein": 15, "fat": 20},
                "소불고기": {"kcal": 220, "carbs": 12, "protein": 18, "fat": 10},
                "치킨너겟": {"kcal": 200, "carbs": 15, "protein": 10, "fat": 12},
                "고등어구이": {"kcal": 180, "carbs": 2, "protein": 18, "fat": 12},
                "계란말이": {"kcal": 120, "carbs": 2, "protein": 10, "fat": 8},
                "시금치나물": {"kcal": 40, "carbs": 5, "protein": 3, "fat": 2},
                "콩나물무침": {"kcal": 45, "carbs": 4, "protein": 3, "fat": 2},
                "어묵볶음": {"kcal": 130, "carbs": 15, "protein": 6, "fat": 5},
                "애호박볶음": {"kcal": 50, "carbs": 6, "protein": 2, "fat": 3},
                "멸치볶음": {"kcal": 110, "carbs": 5, "protein": 10, "fat": 5},
                "배추김치": {"kcal": 15, "carbs": 3, "protein": 1, "fat": 0},
                "깍두기": {"kcal": 18, "carbs": 4, "protein": 1, "fat": 0},
                "백김치": {"kcal": 12, "carbs": 2, "protein": 1, "fat": 0}
            }
            
            menu_cols = ["밥", "국", "반찬1", "반찬2", "반찬3"]
            key_names = ["rice", "soup", "side1", "side2", "kimchi"]
            
            today_food_results = {}
            total_kcal = total_carbs = total_protein = total_fat = 0
            
            # 선택한 실제 반찬 이름을 dish 번호 키로 역추적
            reverse_dish_mapping = {v: k for k, v in dish_mapping.items()}
            
            for col, key_name in zip(menu_cols, key_names):
                food_name = result[col].iloc[0]
                rate = 0
                
                if food_name in reverse_dish_mapping:
                    dish_id = reverse_dish_mapping[food_name]
                    if dish_id in comparison:
                        rate = comparison[dish_id]["percent_eaten"] * 100
                else:
                    for check_name in [food_name, col, key_name, "rice", "soup"]:
                        if check_name in comparison:
                            rate = comparison[check_name]["percent_eaten"] * 100
                            break
                        
                today_food_results[key_name] = {"name": food_name, "intake_rate": round(rate, 1)}
                
                if food_name in NUTRITION_DB:
                    nutrition = NUTRITION_DB[food_name]
                    total_kcal += nutrition["kcal"] * (rate / 100)
                    total_carbs += nutrition["carbs"] * (rate / 100)
                    total_protein += nutrition["protein"] * (rate / 100)
                    total_fat += nutrition["fat"] * (rate / 100)
                    
            today_daily_data = {
                "date": day,
                "food_results": today_food_results,
                "calculated_nutrition": {
                    "total_kcal": round(total_kcal),
                    "total_carbs": round(total_carbs, 1),
                    "total_protein": round(total_protein, 1),
                    "total_fat": round(total_fat, 1)
                }
            }
            
            file_path = 'monthly_data.json'
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        monthly_results = json.load(file)
                    except json.JSONDecodeError:
                        monthly_results = []
            else:
                monthly_results = []
                
            monthly_results.append(today_daily_data)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(monthly_results, file, ensure_ascii=False, indent=2)
                
            st.success("데이터가 성공적으로 기록되었습니다.")