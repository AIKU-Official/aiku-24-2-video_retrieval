import json
import subprocess
import os
import datetime
from prompt import get_funny_timestamps, load_json, convert_timestamp_to_seconds

def cut_video(timestamp_pair, input_video_path, clip_output_dir, title):
    """시작-끝 타임스탬프 쌍을 사용하여 비디오 자르기"""
    # HH:MM:SS 형식의 시작/끝 시간을 초 단위로 변환
    start_time = convert_timestamp_to_seconds(timestamp_pair[0])
    end_time = convert_timestamp_to_seconds(timestamp_pair[1])
    
    # 출력 디렉토리 생성 (없을 경우)
    os.makedirs(clip_output_dir, exist_ok=True)

    # 파일 이름 포맷팅 및 저장 경로 설정
    output_filename = f"home_alone_{start_time}_{end_time}.mp4"
    output_filepath = os.path.join(clip_output_dir, output_filename)

    # ffmpeg 명령어 생성
    command = [
        'ffmpeg',
        '-i', input_video_path,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-c', 'copy',
        output_filepath
    ]

    # ffmpeg 명령어 실행
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"비디오가 {output_filepath}에 저장되었습니다.")
        return output_filepath
    except subprocess.CalledProcessError as e:
        print(f"비디오 자르기 실패: {e}")
        return None

def process_funny_timestamps(funny_timestamps_json, input_video_path, clip_output_dir):
    """funny_timestamps.json 파일을 처리하여 비디오 클립 생성"""
    try:
        # JSON 파일 로드
        with open(funny_timestamps_json, 'r', encoding='utf-8') as f:
            timestamps_data = json.load(f)
        
        # full_timestamps 리스트 가져오기
        full_timestamps = timestamps_data.get('full_timestamps', [])
        
        if not full_timestamps:
            print("타임스탬프 데이터가 없습니다.")
            return []
        
        # 각 타임스탬프 쌍에 대해 비디오 자르기 실행
        output_files = []
        for pair in full_timestamps:
            output_file = cut_video(
                pair,
                input_video_path,
                clip_output_dir,
                title
            )
            if output_file:
                output_files.append(output_file)
        
        print(f"총 {len(output_files)}개의 클립이 생성되었습니다.")
        return output_files
        
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {funny_timestamps_json}")
        return []
    except json.JSONDecodeError:
        print(f"JSON 파일 형식이 잘못되었습니다: {funny_timestamps_json}")
        return []
    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        return []

