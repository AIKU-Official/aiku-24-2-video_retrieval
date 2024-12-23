import os
import torch
import json
import asyncio
import time
from video_caption import process_video
from audio_caption import transcribe_video_to_json
from concat import concat_captions
from prompt import get_funny_timestamps, load_json
from final_video import cut_video, process_funny_timestamps

## 데이터 경로 ##
BASE_PATH = '/home/aikusrv02/aiku/video_retrieval'
base_data_path = f"{BASE_PATH}/data/home_alone" ## 여기 바꿔 여기랑 아랫줄만 바꾸면됨!!!!!!!!##
input_video_path = f"{base_data_path}/home_alone_4800_end.mp4"   ## 여기 바꿔!!!!!!!!##
video_title = "home_alone"
api_key = '' ### chatgpt api key 입력

def setup_paths():
    frame_folder = f"{base_data_path}/video_caption/frames"
    video_caption_output = f"{base_data_path}/video_caption/video_caption_output.json"
    audio_output_dir = f"{base_data_path}/audio_caption"
    audio_output_m4a = f"{audio_output_dir}/audio_caption_output.m4a"
    audio_output_json = f"{audio_output_dir}/audio_caption_output.json"
    merged_output_json = f"{base_data_path}/merged_caption/merged_caption.json"
    data_clips = f"{base_data_path}/clips"
    final_video_output = f"{base_data_path}/final_video"
    funny_timestamps_json = f"{base_data_path}/funny_timestamps.json"
    return {
        'input_video_path': input_video_path,
        'frame_folder': frame_folder,
        'video_caption_output': video_caption_output,
        'audio_output_m4a': audio_output_m4a,
        'audio_output_json': audio_output_json,
        'merged_output_json': merged_output_json,
        'data_clips': data_clips,
        'final_video_output': final_video_output,
        'funny_timestamps_json': funny_timestamps_json
    }

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치 사용 중: {device}")
    return device

def generate_video_captions(input_video_path, frame_folder, video_caption_output):
    try:
        video_captions = process_video(input_video_path, frame_folder, video_caption_output)
        print("비디오 캡션이 생성되어 저장되었습니다.")
        return video_captions
    except Exception as e:
        print(f"비디오 캡션 생성 중 오류 발생: {e}")
        return None

def generate_audio_captions(input_video_path, audio_output_m4a, audio_output_json, device):
    try:
        transcribe_video_to_json(input_video_path, audio_output_m4a, audio_output_json, device)
        print("오디오 캡션이 생성되어 저장되었습니다.")
    except Exception as e:
        print(f"오디오 캡션 생성 중 오류 발생: {e}")

async def main():
    start_time = time.time()  # 시작 시간 기록

    paths = setup_paths()
    device = setup_device()

    # generate_video_captions(paths['input_video_path'], paths['frame_folder'], paths['video_caption_output'])
    # print("비디오 캡션 생성 완료")
    # generate_audio_captions(paths['input_video_path'], paths['audio_output_m4a'], paths['audio_output_json'], device)
    # print("오디오 캡션 생성 완료")
    # # 비디오 캡션과 오디오 캡션을 병합
    # concat_captions(paths['audio_output_json'], paths['video_caption_output'], paths['merged_output_json'])
    # print("병합 완료")
    
    # funny_timestamps.json 생성 후
    await get_funny_timestamps(
        paths['merged_output_json'], 
        video_title, 
        api_key, 
        paths['funny_timestamps_json'],
        num_clips=5  # 원하는 클립 개수 지정
    )

    # 새로운 방식으로 비디오 클립 생성
    output_files = process_funny_timestamps(
        paths['funny_timestamps_json'],
        paths['input_video_path'],
        paths['final_video_output']
    )

    if output_files:
        print("모든 클립이 성공적으로 생성되었습니다.")
    else:
        print("클립 생성 중 문제가 발생했습니다.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"총 소요 시간: {elapsed_time:.2f}초")


if __name__ == "__main__":
    asyncio.run(main())