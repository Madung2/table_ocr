import cv2
import matplotlib.pyplot as plt
import numpy as np
import prrocr

def run_preprocess(image):
    # 이미지 사이즈 조정만 하고 싶어
    scale_percent = 110  # 200% 크기로 확대
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    return resized

def run_ocr(img_path):
    image = cv2.imread(img_path)

    image = run_preprocess(image)

    # 이미지의 색상 체계를 BGR에서 RGB로 변환 (matplotlib는 RGB 이미지를 필요로 함)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # OCR 수행
    ocr = prrocr.ocr(lang="ko")
    result = ocr(image, detail=True)

    # 이미지와 OCR 결과를 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)

    # OCR 결과를 이미지 위에 표시
    for text, bbox in zip(result['description'], result['bounding_poly']):
        vertices = bbox['vertices']
        print(text)

        # 좌표 추출
        x1, y1 = vertices[0]['x'], vertices[0]['y']
        x2, y2 = vertices[2]['x'], vertices[2]['y']

        # OCR 결과 텍스트를 이미지 위에 표시
        plt.text(x1, y1 - 10, text, fontsize=12, color='red', bbox=dict(facecolor='yellow', alpha=0.5))

        # OCR 결과 박스를 이미지 위에 그리기
        plt.plot([x1, x2], [y1, y1], color='green', linewidth=2)  # 위쪽 경계선
        plt.plot([x1, x2], [y2, y2], color='green', linewidth=2)  # 아래쪽 경계선
        plt.plot([x1, x1], [y1, y2], color='green', linewidth=2)  # 왼쪽 경계선
        plt.plot([x2, x2], [y1, y2], color='green', linewidth=2)  # 오른쪽 경계선

    plt.axis('off')  # 축을 숨김
    plt.show()

def run_ocr_only_res(image):
    # image = cv2.imread(img_path)

    image = run_preprocess(image)

    # 이미지의 색상 체계를 BGR에서 RGB로 변환 (matplotlib는 RGB 이미지를 필요로 함)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # OCR 수행
    ocr = prrocr.ocr(lang="ko")
    result = ocr(image, detail=True)
    return result

# run_ocr("4.png")