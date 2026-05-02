"""YOLO 기반 객체 탐지 모듈 (안전모 감지 / 가위바위보 등)."""


import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import os


@st.cache_resource
def load_model(model_path: str = "rsp_best.pt") -> YOLO:
    """YOLO 모델을 로드하고 캐시합니다.
    
    model_path: 모델 파일 경로 (예: 'best_helmet.pt', 'rsp_best.pt')
    """
    if not os.path.exists(model_path):
        st.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    return YOLO(str(model_path))

def detect(image, conf: float = 0.25, model_path: str = "rsp_best.pt") -> dict | None:
    """일반 객체 탐지 (가위바위보 등)."""
    model = load_model(model_path)
    if model is None:
        return None
    results = model.predict(source=image, conf=conf, verbose=False)
    result = results[0]
    
    if len(result.boxes) == 0:
        return None

    xyxy_data = result.boxes.xyxy
    names_data = [result.names[cls.item()] for cls in result.boxes.cls.int()]
    confs_data = result.boxes.conf
    annotated_image = draw_boxes_on_image(image, xyxy_data, names_data, confs_data)

    
    # 감지된 고유 클래스 확인
    unique_classes = set(c for c in names_data)

    if len(unique_classes) > 1:
        return {
            "error": "multiple",
            "detected": unique_classes,
            "annotated_image": annotated_image,
        }

    best_idx = confs_data.argmax()

    return {
        "choice": result.names[int(confs_data[best_idx])],
        "confidence": float(confs_data[best_idx]),
        "annotated_image": annotated_image,
    }


def detect_helmet(image, conf: float = 0.25, model_path: str = "best_helmet.pt") -> dict | None:
    """안전모 전용 탐지 함수.
    
    Returns:
        dict: helmet_count, no_helmet_count, total, compliance_rate, annotated_image
        None: 모델 로드 실패 시
    """
    model = load_model(model_path)
    if model is None:
        return None

    results = model.predict(source=image, conf=conf, verbose=False)
    result = results[0]

    # 결과 이미지 (바운딩 박스가 그려진 이미지)
    xyxy_data = result.boxes.xyxy
    names_data = [result.names[cls.item()] for cls in result.boxes.cls.int()]
    confs_data = result.boxes.conf
    res_rgb = draw_boxes_on_image(image, xyxy_data, names_data, confs_data)

    # 클래스별 카운트 (data_helmet.yaml 기준: 0=Helmet, 1=NO_Helmet)
    class_ids = result.boxes.cls.cpu().numpy()
    helmet_count = int(np.count_nonzero(class_ids == 0))    # Helmet (착용)
    no_helmet_count = int(np.count_nonzero(class_ids == 1)) # NO_Helmet (미착용)
    total = helmet_count + no_helmet_count

    compliance_rate = 0.0
    if total > 0:
        compliance_rate = round((helmet_count / total) * 100, 1)

    return {
        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "total": total,
        "compliance_rate": compliance_rate,
        "annotated_image": res_rgb,
    }

def draw_boxes_on_image(image, xyxy, names, confs):

    for i in range(len(names)):
        # 여기에 코딩
        x1, y1, x2, y2 = map(int, xyxy[i].tolist())
        name = names[i]
        conf = confs[i].item() # confs is a tensor, need to get item
        color = (255, 0, 0)  # Green color
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
        label = f"{name}: {conf:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        # Put background rectangle for text
        cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)

        # Put text on the image, changed color to white (255, 255, 255)
        cv2.putText(image, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return image


def detect_meal(image, model_path: str = "best (9).pt") -> tuple:
    """급식(식판) 음식 전용 탐지 함수.
    
    Returns:
        tuple, (food_dict, dishes_list, annotated_image)
    """
    model = load_model(model_path)
    if model is None:
        return {}, [], image

    # 이미지가 4채널(RGBA)일 경우 3채널(RGB)로 변환
    if hasattr(image, 'shape') and len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    results = model.predict(source=image, verbose=False)
    result = results[0]

    xyxy_data = result.boxes.xyxy
    names_data = [result.names[cls.item()] for cls in result.boxes.cls.int()]
    
    items = []
    for i in range(len(xyxy_data)):
        x1, y1, x2, y2 = xyxy_data[i].tolist()
        items.append({
            "box": (x1, y1, x2, y2),
            "name": names_data[i],
            "area": (x2 - x1) * (y2 - y1),
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2
        })
        
    dishes = [item for item in items if item["name"].lower() == "dish"]
    # 위치 기준 정렬 (세로 기준 대략적으로 묶고, 가로로 정렬)
    dishes = sorted(dishes, key=lambda d: (round(d["center_y"] / 50), d["center_x"]))
    
    other_foods = [item for item in items if item["name"].lower() != "dish"]
    
    food_dict = {}
    for i, d in enumerate(dishes):
        d["unique_id"] = f"dish_{i+1}"
        food_dict[d["unique_id"]] = d["area"]
        
    for o in other_foods:
        name = o["name"]
        if name in food_dict:
            food_dict[name] += o["area"]
        else:
            food_dict[name] = o["area"]
            
    annotated_image = result.plot()
    for i, d in enumerate(dishes):
        x1, y1 = int(d["box"][0]), int(d["box"][1])
        cv2.putText(annotated_image, str(i+1), (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        
    return food_dict, dishes, annotated_image