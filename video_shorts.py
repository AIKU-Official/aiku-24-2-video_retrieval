from moviepy.editor import VideoFileClip, ColorClip, CompositeVideoClip
import os

### 경로 설정 ###
BASE_PATH = '/home/aikusrv02/aiku/video_retrieval'
base_data_path = f"{BASE_PATH}/data/home_alone" ### 여기만 바꾸면 됨##




def add_margin_to_short(input_video_path, output_video_path):
    # 비디오 클립 불러오기
    clip = VideoFileClip(input_video_path)
    
    # Shorts 비율 (9:16) 계산
    target_aspect_ratio = 9 / 16
    video_aspect_ratio = clip.w / clip.h
    
    # 목표 크기 계산 (9:16 비율에 맞춤)
    target_width = clip.w
    target_height = int(target_width / target_aspect_ratio)
    
    # 위아래 여백 추가 여부 결정
    if video_aspect_ratio > target_aspect_ratio:
        # 가로가 더 긴 경우, 위아래에 여백 추가
        margin_height = (target_height - clip.h) // 2
        # 위아래 여백을 추가한 배경 생성
        background = ColorClip(size=(clip.w, target_height), color=(0, 0, 0))
        background = background.set_duration(clip.duration)
        
        # 클립을 배경 중앙에 배치
        video_with_margin = CompositeVideoClip([background, clip.set_position(("center", "center"))])
    else:
        # 비율이 맞으면 여백 추가 없이 원본 유지
        video_with_margin = clip

    # 출력 파일 저장
    video_with_margin.write_videofile(output_video_path, codec="libx264", fps=clip.fps)


def print_video_dimensions(video_path):
    # 비디오 파일 불러오기
    clip = VideoFileClip(video_path)
    
    # 가로 및 세로 길이 출력
    width, height = clip.w, clip.h
    print(f"비디오 가로 길이: {width} 픽셀")
    print(f"비디오 세로 길이: {height} 픽셀")
    

if __name__ == "__main__":
    final_video_path = f"{base_data_path}/final_video/"
    shorts_output_dir = f"{base_data_path}/shorts_trimmed/"
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(shorts_output_dir):
        os.makedirs(shorts_output_dir)
    
    for file in os.listdir(final_video_path):
        if file.endswith('.mp4'):  # mp4 파일만 처리
            input_video_path = os.path.join(final_video_path, file)
            # 파일 이름 뒤에 _shorts 추가
            base_name = os.path.splitext(file)[0]  # 확장자 제외한 파일 이름
            output_video_path = os.path.join(shorts_output_dir, f"{base_name}_shorts_black.mp4")
            add_margin_to_short(input_video_path, output_video_path)