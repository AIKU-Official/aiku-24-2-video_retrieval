import torch
from transformers import (
    AutoProcessor, 
    AutoModelForVideoClassification,
    pipeline
)
import whisper
import cv2
import os
from moviepy.editor import VideoFileClip
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

# 기본 경로 설정
BASE_PATH = '/home/aikusrv02/aiku/video_retrieval'
SHORTS_PATH = f"{BASE_PATH}/data/home_alone/shorts_trimmed"
OUTPUT_PATH = f"{BASE_PATH}/data/home_alone/shorts_metadata"

class MultiModalAnalyzer:
    def __init__(self):
        print("모델 초기화 중...")
        try:
            # 비디오 분석 모델
            self.video_processor = AutoProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            self.video_model = AutoModelForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            
            # STT 모델 (Whisper)
            self.stt_model = whisper.load_model("base")
            
            # 캡셔닝 파이프라인
            self.caption_pipeline = pipeline(
                "video-classification", 
                model="facebook/timesformer-base-finetuned-k400",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # GPU 설정
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.video_model.to(self.device)
            
            print(f"모델 초기화 완료. 사용 장치: {self.device}")
            
        except Exception as e:
            print(f"모델 초기화 중 오류 발생: {e}")
            raise

    def extract_video_features(self, video_path: str) -> list:
        """비디오 프레임 분석"""
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(total_frames // 16, 1)
            
            for i in range(16):
                frame_idx = min(i * interval, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()
            
            if not frames:
                return []
            
            # 비디오 특징 추출
            inputs = self.video_processor(list(frames), return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.video_model(**inputs)
                video_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                _, indices = torch.topk(video_probs, k=5)
                
            return [self.video_model.config.id2label[idx.item()] for idx in indices[0]]
            
        except Exception as e:
            print(f"비디오 특징 추출 중 오류 발생: {e}")
            return []

    def extract_stt(self, video_path: str) -> dict:
        """비디오에서 음성 추출 및 STT 수행"""
        try:
            # 임시 오디오 파일 경로
            temp_audio = "temp_audio.wav"
            
            # 오디오 추출
            video = VideoFileClip(video_path)
            if video.audio is None:
                print("오디오 트랙이 없습니다.")
                return None
            
            # 오디오 저장
            video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
            video.close()
            
            # STT 수행
            result = self.stt_model.transcribe(temp_audio)
            
            # 임시 파일 삭제
            os.remove(temp_audio)
            
            return result
            
        except Exception as e:
            print(f"STT 추출 중 오류 발생: {e}")
            return None

    def generate_caption(self, video_features: list, stt_result: dict) -> str:
        """비디오 특징과 STT 결과를 결합하여 캡션 생성"""
        try:
            caption_elements = []
            
            # 1. 비디오 행동/장면 설명 추가
            if video_features:
                main_action = video_features[0]
                caption_elements.append(f"영상에서 '{main_action}' 행동이 감지되었으며")
            
            # 2. STT 텍스트 추가
            if stt_result and 'text' in stt_result and stt_result['text'].strip():
                caption_elements.append(f"다음과 같은 대사가 포함되어 있습니다: '{stt_result['text'].strip()}'")
            
            # 3. 캡션 조합
            if caption_elements:
                return " ".join(caption_elements)
            else:
                return "캡션을 생성할 수 없습니다."
                
        except Exception as e:
            print(f"캡션 생성 중 오류 발생: {e}")
            return "캡션 생성 중 오류가 발생했습니다."

    def generate_shorts_title(self, video_features: list, stt_result: dict, caption: str) -> dict:
        """캡션을 바탕으로 쇼츠 스타일의 제목 생성"""
        
        # 감정/상황별 이모지 매핑
        emotion_emoji = {
            'funny': ['😂', '🤣', '😆'],
            'exciting': ['🔥', '⚡', '💥'],
            'surprising': ['😱', '😲', '🤯'],
            'cute': ['😊', '🥰', '💖'],
            'cool': ['😎', '🆒', '✨'],
            'scary': ['😨', '🙀', '👻'],
            'action': ['💪', '🏃', '👊'],
            'clever': ['🧠', '💡', '🤓']
        }
        
        # 행동/키워드별 감정 매핑
        action_emotion = {
            'running': 'action',
            'jumping': 'exciting',
            'falling': 'funny',
            'laughing': 'funny',
            'fighting': 'action',
            'dancing': 'cool',
            'screaming': 'surprising',
            'hiding': 'scary',
            'planning': 'clever'
        }
        
        try:
            # 1. 주요 감정/상황 파악
            emotion = 'exciting'  # 기본값
            
            # 비디오 특징에서 감정 파악
            if video_features:
                main_action = video_features[0].lower()
                for action, emo in action_emotion.items():
                    if action in main_action:
                        emotion = emo
                        break
            
            # STT 텍스트에서 감정 파악
            if stt_result and 'text' in stt_result:
                text = stt_result['text'].lower()
                for action, emo in action_emotion.items():
                    if action in text:
                        emotion = emo
                        break
            
            # 2. 이모지 선택
            emoji = np.random.choice(emotion_emoji[emotion])
            
            # 3. 제목 템플릿
            templates = {
                'funny': [
                    f"케빈이 또 대작전 성공했습니다 {emoji}",
                    f"이게 되네?ㅋㅋㅋ {emoji}",
                    f"도둑들 빵터진 순간 {emoji}"
                ],
                'exciting': [
                    f"케빈 레전드 순간 {emoji}",
                    f"이걸 생각해내다니.. {emoji}",
                    f"도둑잡기 대작전 {emoji}"
                ],
                'surprising': [
                    f"상상도 못한 반전 {emoji}",
                    f"도둑들 경악한 순간 {emoji}",
                    f"케빈의 놀라운 작전 {emoji}"
                ],
                'clever': [
                    f"천재 케빈의 수학 {emoji}",
                    f"이 꼬마 뭔가 다르다 {emoji}",
                    f"케빈의 똑똑한 함정 {emoji}"
                ]
            }
            
            # 4. 제목 생성
            title_templates = templates.get(emotion, templates['exciting'])
            main_title = np.random.choice(title_templates)
            
            # 5. 해시태그 추가 (30% 확률)
            hashtags = ["#홈얼론", "#나홀로집에", "#케빈", "#레전드"]
            if np.random.random() < 0.3:
                main_title += f" {np.random.choice(hashtags)}"
            
            # 6. 썸네일용 간단 텍스트 생성
            thumbnail_text = main_title.split('#')[0].strip()  # 해시태그 제외
            
            return {
                'title': main_title,
                'thumbnail_text': thumbnail_text,
                'emotion': emotion,
                'emoji_used': emoji
            }
            
        except Exception as e:
            print(f"제목 생성 중 오류 발생: {e}")
            return {
                'title': f"케빈의 레전드 순간 ✨",
                'thumbnail_text': "케빈의 레전드 순간",
                'emotion': 'exciting',
                'emoji_used': '✨'
            }

def process_video(video_path: str, analyzer: MultiModalAnalyzer) -> dict:
    """단일 비디오 처리"""
    try:
        print(f"\n비디오 분석 중: {Path(video_path).name}")
        
        results = {
            'video_path': video_path,
            'video_features': [],
            'stt_result': None,
            'caption': ''
        }
        
        # 1. 비디오 특징 추출
        print("비디오 특징 추출 중...")
        results['video_features'] = analyzer.extract_video_features(video_path)
        
        # 2. STT 수행
        print("음성 인식 수행 중...")
        results['stt_result'] = analyzer.extract_stt(video_path)
        
        # 3. 캡션 생성
        print("캡션 생성 중...")
        results['caption'] = analyzer.generate_caption(
            results['video_features'],
            results['stt_result']
        )
        
        # 4. 쇼츠 스타일의 제목 생성
        print("제목 생성 중...")
        results['shorts_title'] = analyzer.generate_shorts_title(
            results['video_features'],
            results['stt_result'],
            results['caption']
        )
        
        # 결과 출력
        print("\n=== 분석 결과 ===")
        print("\n비디오 특징:")
        for feat in results['video_features']:
            print(f"- {feat}")
            
        if results['stt_result']:
            print("\nSTT 결과:")
            print(results['stt_result']['text'])
            
        print("\n생성된 캡션:")
        print(results['caption'])
        
        print("\n제목 생성 결과:")
        print(results['shorts_title'])
        
        return results
        
    except Exception as e:
        print(f"비디오 처리 중 오류 발생: {e}")
        return None

def save_results(results: dict, output_dir: str):
    """결과를 JSON 파일로 저장"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(
            output_dir,
            f"{Path(results['video_path']).stem}_analysis.json"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print(f"\n결과 저장 완료: {output_path}")
        
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")

def main():
    """메인 실행 함수"""
    print("\n=== 멀티모달 비디오 분석 시작 ===\n")
    
    # 비디오 파일 목록 가져오기
    if not os.path.exists(SHORTS_PATH):
        print(f"오류: 입력 디렉토리가 존재하지 않습니다: {SHORTS_PATH}")
        return
        
    video_files = [f for f in os.listdir(SHORTS_PATH) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"오류: 입력 디렉토리에 비디오 파일이 없습니다: {SHORTS_PATH}")
        return
    
    print(f"총 {len(video_files)}개의 비디오 파일을 처리합니다.")
    
    try:
        # 분석기 초기화
        analyzer = MultiModalAnalyzer()
        
        # 각 비디오 처리
        for file in tqdm(video_files, desc="비디오 처리 중"):
            video_path = os.path.join(SHORTS_PATH, file)
            results = process_video(video_path, analyzer)
            
            if results:
                save_results(results, OUTPUT_PATH)
            print("-" * 50)
        
        print("\n모든 처리가 완료되었습니다!")
        
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main() 