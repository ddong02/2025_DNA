import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from collections import Counter
import numpy as np

def create_sample_image(filename='sample_image.png'):
    """Demonstration을 위한 샘플 PNG 이미지를 생성합니다."""
    img = Image.new('RGBA', (200, 150), color = '#F0F0F0') # Light gray background
    draw = ImageDraw.Draw(img)
    # 다양한 색상의 사각형 그리기
    draw.rectangle([10, 10, 70, 50], fill='red')      # Red
    draw.rectangle([80, 10, 140, 50], fill='blue')     # Blue
    draw.rectangle([150, 10, 190, 50], fill='green')  # Green
    draw.rectangle([10, 60, 70, 100], fill='#FFA500') # Orange
    draw.rectangle([80, 60, 140, 100], fill='red')      # Another Red
    draw.rectangle([150, 60, 190, 140], fill='#008080') # Teal
    draw.rectangle([10, 110, 70, 140], fill='blue')    # Another Blue
    img.save(filename)
    print(f"샘플 이미지 생성 완료: '{filename}'")
    return filename

def plot_color_distribution(image_path, top_n=20):
    """
    PNG 이미지의 색상 분포를 분석하고 플롯합니다.

    Args:
        image_path (str): PNG 이미지 파일 경로.
        top_n (int): 표시할 상위 색상 개수.
    """
    try:
        img = Image.open(image_path)
        img = img.convert("RGBA")
    except FileNotFoundError:
        print(f"오류: '{image_path}' 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return

    # 픽셀 데이터에서 색상 카운트
    pixels = list(img.getdata())
    color_counts = Counter(pixels)

    # 가장 흔한 색상 추출
    most_common_colors = color_counts.most_common(top_n)

    if not most_common_colors:
        print("이미지에서 색상을 찾을 수 없습니다.")
        return

    # 플롯 데이터 준비
    plot_colors = []
    counts = []
    hex_labels = []
    for color, count in most_common_colors:
        # Matplotlib는 0-1 범위의 RGB 색상을 사용합니다.
        plot_colors.append(tuple(c/255.0 for c in color))
        counts.append(count)
        # HEX 코드로 변환 (알파값 제외)
        hex_labels.append('#%02x%02x%02x' % color[:3])


    # 플롯 생성
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    bars = ax.bar(range(len(counts)), counts, color=plot_colors, edgecolor='black', linewidth=0.5)

    # 라벨 및 제목 설정 (한글 폰트가 없을 경우 깨질 수 있습니다)
    plt.xlabel("Colors (HEX Code)")
    plt.ylabel("Pixel Count")
    plt.title(f"Top {len(most_common_colors)} Color Distribution for '{image_path}'")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(hex_labels, rotation=45, ha="right")

    plt.tight_layout()

    # 플롯 파일로 저장
    plot_filename = 'color_distribution.png'
    plt.savefig(plot_filename)

    print(f"색상 분포 플롯이 '{plot_filename}'으로 저장되었습니다.")


# --- 코드 실행 부분 ---

# 1. 샘플 이미지 생성
# sample_image_file = create_sample_image()

# 2. 생성된 샘플 이미지의 색상 분포 분석 및 플롯 생성
# plot_color_distribution(sample_image_file)

# 3. (선택) 다른 이미지를 분석하려면 아래 코드의 주석을 해제하고 파일명을 수정하세요.
my_image_file = '/Users/eogus/Desktop/Dataset/2025DNA/SemanticDatasetTest/image/test/set1/000001_leftImg8bit.png'
plot_color_distribution(my_image_file)