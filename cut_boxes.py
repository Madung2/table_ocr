import cv2
import os
import pytesseract  # pytesseract 임포트 추가
import numpy as np
def preprocess_image(img):
    # 이미지 불러오기
    # img = cv2.imread(image_path)
    print('type', type(img))
    print('Image shape:', img.shape, 'Image dtype:', img.dtype)
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    # 이미지의 채널 수 확인
    if len(img.shape) == 2:
        gray = img  # 이미 그레이스케일 이미지인 경우
    elif len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format: {}".format(img.shape))
    
    # 이진화
    _, img_bin = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    return img, img_bin

def find_lines(img_bin):
    # 수평과 수직 선 검출을 위한 커널
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_bin.shape[0]//30))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_bin.shape[1]//30, 1))
    
    # 수직선 검출
    vertical_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # 수평선 검출
    horizontal_lines = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # 수직선과 수평선을 결합하여 테이블의 교차점 찾기
    grid = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    
    return grid

def find_contours_and_save(img, grid, output_folder):
    # 컨투어 검출
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    # 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 컨투어를 돌면서 사각형으로 자르기
    i = 0
    for j, contour in enumerate(contours):
        if j == 0:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # 너무 작은 박스는 무시 (노이즈 필터링)
        if w > 50 and h > 20:
            print(j)
            cells.append((x, y, w, h))
            cropped_img = img[y:y+h, x:x+w]
            img_filename = os.path.join(output_folder, f'{i}.png')
            cv2.imwrite(img_filename, cropped_img)
            i += 1
    
    cells = sorted(cells, key=lambda c: (c[1], c[0]))
    return cells, i
def generate_html_table(cells, img, output_html, len_nums):
    # 셀 간의 간격을 기준으로 행과 열 개수 추정
    rows = []
    current_row = []
    last_y = None
    
    for (x, y, w, h) in cells:
        if last_y is None or abs(y - last_y) < 10:  # 새로운 행이 아니라면
            current_row.append((x, y, w, h))
        else:  # 새로운 행이면
            rows.append(current_row)
            current_row = [(x, y, w, h)]
        
        last_y = y
    
    if current_row:
        rows.append(current_row)
    
    # HTML 테이블 생성
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write("<table border='1'>\n")
        i =0
        
        for row in rows:
            f.write("<tr>\n")
            for (x, y, w, h) in row:
                i+=1
                cropped_img = img[y:y+h, x:x+w]
                text = pytesseract.image_to_string(cropped_img, config='--psm 7').strip()
                f.write(f"<td>{{{len_nums-i}}}</td>\n")
            f.write("</tr>\n")
        
        f.write("</table>\n")


def clear_saved_images_folder(output_folder):
    # 폴더 내 모든 파일 삭제
    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
# 실행 부분
def process_image_and_generate_table(image):
    output_folder = 'saved_images'  # static한 출력 폴더 이름
    output_html = 'table_output.html'  # static한 출력 HTML 파일 이름

    # 폴더 내의 기존 파일 삭제
    clear_saved_images_folder(output_folder)

    # 1. 이미지 전처리
    img, img_bin = preprocess_image(image)

    # 2. 줄 찾기 (수평선과 수직선 검출)
    grid = find_lines(img_bin)

    # 3. 컨투어를 찾고, 개별 이미지를 저장
    cells, len_nums = find_contours_and_save(img, grid, output_folder)

    # 4. HTML 테이블 생성
    generate_html_table(cells, img, output_html, len_nums)

    print(f'Cropped images have been saved to the folder: {output_folder}')
    print(f'HTML table has been generated: {output_html}')

# # 이제 이 함수를 호출할 수 있습니다.
# process_image_and_generate_table('easy_table.jpg')