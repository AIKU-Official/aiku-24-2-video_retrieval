from moviepy.editor import VideoFileClip, AudioFileClip
from PIL import ImageFont
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import cv2
import os
import json
import random
import numpy as np
from PIL import ImageDraw, Image

### 경로 설정 ###
BASE_PATH = '/home/aikusrv02/aiku/video_retrieval'
font_path = '/home/aikusrv02/aiku/video_retrieval/data/이서윤체.ttf'
base_data_path = f"{BASE_PATH}/data/home_alone"

def extract_frames(video_path, num_frames=16):
    """비디오에서 균일한 간격으로 프레임 추출 및 리사이즈"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"비디오 파일을 읽을 수 없습니다: {video_path}")
        return None
        
    interval = max(total_frames // num_frames, 1)
    target_size = (224, 224)  # VideoMAE 모델의 기대 입력 크기
    
    try:
        for i in range(num_frames):
            frame_idx = min(i * interval, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"프레임 {frame_idx} 읽기 실패")
                continue
                
            # 프레임 전처리
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
    except Exception as e:
        print(f"프레임 추출 중 오류 발생: {e}")
    finally:
        cap.release()
    
    if len(frames) != num_frames:
        print(f"추출된 프레임 수가 부족합니다. (예상: {num_frames}, 실제: {len(frames)})")
        return None
        
    return np.array(frames)  # numpy 배열로 변환

def analyze_video_content(video_path):
    """비디오 내용 분석하여 행동/장면 설명 생성"""
    try:
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        model = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        
        frames = extract_frames(video_path)
        if frames is None:
            print("프레임 추출 실패")
            return None
            
        try:
            # 입력 형태 확인 및 출력
            print(f"프레임 배열 형태: {frames.shape}")
            
            # 프레임 정규화 및 모델 입력 준비
            inputs = processor(list(frames), return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                top_k = 3
                topk_values, topk_indices = torch.topk(logits, top_k)
                predictions = []
                
                for idx in topk_indices[0]:
                    label = model.config.id2label[idx.item()]
                    predictions.append(label)
            
            return predictions
            
        except Exception as e:
            print(f"모델 처리 중 오류 발생: {e}")
            print(f"입력 텐서 크기: {inputs['pixel_values'].shape}")  # 디버깅용
            return None
        
    except Exception as e:
        print(f"비디오 분석 중 오류 발생: {e}")
        return None

def generate_shorts_title(actions):
    """쇼츠 스타일의 매력적인 한글 제목 생성"""
    
    # 행동 한글 매핑 (더 감각적인 표현으로 수정)
    action_mapping = {
        'fighting': '격파',
        'running': '도주',
        'falling': '낙사',
        'laughing': '빵터진',
        'screaming': '절규하는',
        'dancing': '춤추는',
        'jumping': '점프',
        'eating': '먹방',
        'playing': '플레이',
        'hitting': '격파',
        'throwing': '던지기',
        'catching': '캐치',
        'sliding': '슬라이딩',
        'hiding': '스텔스',
        'chasing': '추격',
        'pranking': '몰카',
        'surprising': '충격적',
        'tricking': '속임수'
    }
    
    # 쇼츠 스타일 이모지
    emoji_mapping = {
        'fighting': ['🔥', '👊'],
        'running': ['💨', '🏃'],
        'falling': ['💫', '😱'],
        'laughing': ['🤣', '😂'],
        'screaming': ['😱', '⚡'],
        'dancing': ['🕺', '💃'],
        'jumping': ['⬆️', '🦘'],
        'eating': ['🍽️', '😋'],
        'playing': ['🎮', '🎯'],
        'hitting': ['💥', '👊'],
        'throwing': ['🎯', '🎪'],
        'catching': ['🎯', '🙌'],
        'sliding': ['💨', '🌪️'],
        'hiding': ['🙈', '👻'],
        'chasing': ['🏃', '💨'],
        'pranking': ['😈', '🎭'],
        'surprising': ['😲', '❗'],
        'tricking': ['🎭', '🃏']
    }
    
    main_action = actions[0].lower()
    emoji = random.choice(emoji_mapping.get(main_action, ['✨', '🔥']))
    
    # 쇼츠 스타일 제목 템플릿
    templates = [
        # 궁금증 유발형
        [
            f"케빈이 {action_mapping.get(main_action, '미쳤다')}.. {emoji}",
            f"이게 되네?! {action_mapping.get(main_action, '천재')} 케빈 {emoji}",
            f"도둑들 실제 반응 {emoji} (ft. {action_mapping.get(main_action, '충격')})",
        ],
        
        # 임팩트형
        [
            f"역대급 {action_mapping.get(main_action, '레전드')} 순간 {emoji}",
            f"케빈 매운맛 {action_mapping.get(main_action, '복수')} {emoji}",
            f"이 장면 실화임 {emoji} {action_mapping.get(main_action, '충격')}",
        ],
        
        # 밈형
        [
            f"이때부터 전설이었다 {emoji} #{action_mapping.get(main_action, '레전드')}",
            f"케빈 진짜 무섭네 {emoji} #{action_mapping.get(main_action, '실화')}",
            f"도둑들 멘탈 붕괴 {emoji} #{action_mapping.get(main_action, '파괴')}",
        ]
    ]
    
    # 랜덤 선택 + 해시태그 추가
    category = random.choice(templates)
    title = random.choice(category)
    
    # 50% 확률로 트렌디한 해시태그 추가
    if random.random() < 0.5:
        hashtags = ["#홈얼론", "#나홀로집에", "#케빈", "#추억의명장면", "#레전드"]
        title += f" {random.choice(hashtags)}"
    
    return title

def generate_thumbnail_text(title):
    """썸네일용 간단한 텍스트 추출"""
    # 이모지와 해시태그 제거
    clean_title = ''.join(char for char in title if not char in '🔥👊💨🏃😱🤣😂⚡🕺💃⬆️🦘🍽️😋🎮🎯💥👊🎪🙌🌪️👻😈🎭😲❗🃏✨ #')
    
    # 긴 제목은 첫 부분만 사용
    if len(clean_title) > 15:
        clean_title = clean_title[:15] + '...'
    
    return clean_title.strip()
'''
def find_korean_font():
    """SSH 환경에서 사용 가능한 한글 폰트 찾기"""
    # 서버 환경에서 자주 사용되는 폰트 경로들
    font_paths = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf',
        '/usr/share/fonts/truetype/fonts-korean-gothic/gothic.ttf'
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                # 폰트 로드 테스트
                font = ImageFont.truetype(font_path, 14)
                print(f"사용 가능한 한글 폰트 발견: {font_path}")
                return font_path
            except Exception as e:
                continue
    
    # 폰트를 찾지 못한 경우, 현재 디렉토리에서 폰트 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(current_dir, 'fonts')
    
    if os.path.exists(font_dir):
        for file in os.listdir(font_dir):
            if file.endswith(('.ttf', '.otf')):
                font_path = os.path.join(font_dir, file)
                try:
                    font = ImageFont.truetype(font_path, 14)
                    print(f"로컬 폰트 발견: {font_path}")
                    return font_path
                except:
                    continue
    
    print("경고: 한글 폰트를 찾을 수 없습니다. 폰트 설치가 필요할 수 있습니다.")
    print("다음 명령어로 나눔폰트를 설치할 수 있습니다:")
    print("sudo apt-get install fonts-nanum")
    return None
'''

def add_title_to_video(video_path, title, output_path):
    """OpenCV를 사용하여 비디오에 제목 오버레이 추가하고 오디오 결합"""
    try:
        # 비디오 읽기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("비디오를 열 수 없습니다")
            return False

        # 비디오 속성 가져오기
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 출력 비디오 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output_path = output_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        # 폰트 설정
        font_size = int(width / 15)
        font = ImageFont.truetype(font_path, font_size)

        # 첫 해시태그에서 줄바꿈 처리
        if "#" in title:
            split_pos = title.index("#")
            title_lines = [title[:split_pos].strip(), title[split_pos:].strip()]
        else:
            title_lines = [title]

        # 텍스트 위치를 화면의 아래쪽으로 조정
        y_offset = int(height * 0.75)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV 프레임을 PIL Image로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)

            for i, line in enumerate(title_lines):
                # 각 줄의 너비 계산
                line_bbox = draw.textbbox((0, 0), line, font=font)
                line_width = line_bbox[2] - line_bbox[0]
                x_pos = (width - line_width) // 2
                y_pos = y_offset + (i * (font_size + 10))  # 줄 간격 10픽셀

                # 텍스트 그림자 효과
                shadow_color = (0, 0, 0)
                shadow_offset = int(font_size * 0.05)

                # 전방향 그림자로 더 굵은 외곽선 효과
                for dx in range(-shadow_offset, shadow_offset + 1, 2):
                    for dy in range(-shadow_offset, shadow_offset + 1, 2):
                        draw.text((x_pos + dx, y_pos + dy), line, font=font, fill=shadow_color)

                # 메인 텍스트 (흰색) - 해시태그와 이모지 포함
                draw.text((x_pos, y_pos), line, font=font, fill=(255, 255, 255))

            # PIL Image를 다시 OpenCV 형식으로 변환
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            out.write(frame)

            # 진행률 표시
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = int((current_frame / total_frames) * 100)
            if progress % 10 == 0:
                print(f"처리 진행률: {progress}%")

        # 리소스 정리
        cap.release()
        out.release()

        # MoviePy로 비디오에 원본 오디오 추가
        video_with_audio = VideoFileClip(temp_output_path)
        original_audio = VideoFileClip(video_path).audio
        final_video = video_with_audio.set_audio(original_audio)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # 임시 파일 삭제
        os.remove(temp_output_path)

        print("제목 추가 및 오디오 유지 완료")
        return True

    except Exception as e:
        print(f"비디오 제목 추가 중 오류 발생: {e}")
        return False

def save_video_metadata(video_path, title, actions, output_dir):
    """비디오 메타데이터를 JSON 파일로 저장"""
    try:
        video = VideoFileClip(video_path)
        metadata = {
            'video_path': video_path,
            'title': title,
            'actions': actions,
            'duration': video.duration
        }
        video.close()
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(output_dir, f"{base_name}_metadata.json")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
            
        return True
    except Exception as e:
        print(f"메타데이터 저장 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    print("프로그램 시작")  # 디버깅용 print
    
    # 입력/출력 경로 설정
    final_video_path = f"{base_data_path}/shorts_trimmed/"
    metadata_dir = f"{base_data_path}/shorts_metadata/"
    
    # 메타데이터 디렉토리 생성
    os.makedirs(metadata_dir, exist_ok=True)
    
    # 입력 디렉토리 확인
    if not os.path.exists(final_video_path):
        print(f"오류: 입력 디렉토리가 존재하지 않습니다: {final_video_path}")
        exit(1)
    
    # 처리할 비디오 파일 목록
    video_files = [f for f in os.listdir(final_video_path) if f.endswith('.mp4')]
    print(f"발견된 비디오 파일들: {video_files}")  # 디버깅용 print
    
    if not video_files:
        print(f"오류: 입력 디렉토리에 MP4 파일이 없습니다: {final_video_path}")
        exit(1)
    
    print(f"처리할 비디오 파일 수: {len(video_files)}")
    
    for file in video_files:
        video_path = os.path.join(final_video_path, file)
        print(f"\n처리 중인 파��: {file}")
        
        # 1. 비디오 내용 분석
        actions = analyze_video_content(video_path)
        
        if actions:
            # 2. 제목 생성
            title = generate_shorts_title(actions)
            thumbnail_text = generate_thumbnail_text(title)
            
            # 3. 제목이 있는 비디오 생성
            output_video_path = os.path.join(
                metadata_dir,
                f"titled_{os.path.basename(video_path)}"
            )
            
            if add_title_to_video(video_path, title, output_video_path):
                print(f"제목이 추가된 비디오 저장 완료: {output_video_path}")
            else:
                print("제목 추가 실패")
            
            # 4. 메타데이터 저장
            if save_video_metadata(video_path, title, actions, metadata_dir):
                print(f"생성된 제목: {title}")
                print(f"썸네일 텍스트: {thumbnail_text}")
                print(f"감지된 행동: {', '.join(actions)}")
                print(f"메타데이터 저장 완료")
            else:
                print("메타데이터 저장 실패")
        else:
            print(f"비디오 분석 실패")
        
        print("-" * 50)
    
    print("\n모든 처리 완료!")