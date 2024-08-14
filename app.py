import streamlit as st
import os
from ocr import run_ocr_only_res
from PIL import Image
import numpy as np
import cv2
from cut_boxes import process_image_and_generate_table
# Streamlit 앱 제목
st.title("OCR Web Application")

# 이미지 업로드
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 이미지를 열고 화면에 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # # 이미지 처리
    image = np.array(image)
    process_image_and_generate_table(image)


    # saved_images 폴더에서 파일들 읽어오기
    output_folder = 'saved_images'
    result_list = []

    if os.path.exists(output_folder):
        for filename in sorted(os.listdir(output_folder)):
            file_path = os.path.join(output_folder, filename)
            file_num = filename.split('.')[0]
            print(file_path, file_num)
            img = cv2.imread(file_path)
            result = run_ocr_only_res(img)
            ocr_text = ' '.join(result['description'])
            with open('table_output.html', 'r', encoding='utf-8') as file:
                html_content = file.read()

            # <<file_num>>을 ocr_text로 대체
            new_html_content = html_content.replace(f'{{{file_num}}}', ocr_text)

            # 수정된 내용을 다시 파일에 기록
            with open('table_output.html', 'w', encoding='utf-8') as file:
                file.write(new_html_content)

            # # 결과 리스트에 추가
            # result_list.append(ocr_text)
    # result = run_ocr_only_res(image)
    st.write("OCR processing completed and HTML file updated.")
    with open('table_output.html', 'r', encoding='utf-8') as file:
        updated_html_content = file.read()
    
    st.components.v1.html(updated_html_content, height=600, scrolling=True)


    # # OCR 결과를 이미지에 표시
    # if result:
    #     for text, bbox in zip(result['description'], result['bounding_poly']):
    #         vertices = bbox['vertices']
    #         x1, y1 = vertices[0]['x'], vertices[0]['y']
    #         x2, y2 = vertices[2]['x'], vertices[2]['y']
    #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    #     # 결과 이미지 출력
    #     st.image(image, caption="OCR Processed Image", use_column_width=True)

    #     # OCR 텍스트 결과 출력
    #     st.write("OCR Results:")
    #     for text in result['description']:
    #         st.write(text)
