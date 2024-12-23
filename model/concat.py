import json
import os
from datetime import datetime, timedelta

def to_datetime(time_str):
    try:
        # 부동 소수점 숫자로 제공되는 경우
        if isinstance(time_str, (float, int)) or time_str.replace('.', '', 1).isdigit():
            return datetime(1900, 1, 1) + timedelta(seconds=float(time_str))
        # 문자열로 제공되는 경우
        return datetime.strptime(time_str, '%H:%M:%S.%f')
    except ValueError:
        return datetime.strptime(time_str, '%H:%M:%S')

def merge_captions(audio_caption, video_caption):
    merged_data = []

    for audio in audio_caption:
        start_time = to_datetime(str(audio['start_formatted']))  # 수정된 부분
        end_time = to_datetime(str(audio['end_formatted']))      # 수정된 부분

        relevant_videos = [
            video['caption']
            for video in video_caption
            if start_time <= to_datetime(str(video['time']))     # 수정된 부분
            <= end_time
        ]

        merged_entry = {
            'time': audio['start_formatted'],
            'audio_caption': audio['text'],
            'video_caption': relevant_videos
        }

        merged_data.append(merged_entry)

    return merged_data

def save_merged_data(merged_data, output_path):
    with open(output_path, 'w') as file:
        json.dump(merged_data, file, indent=4)
        
def concat_captions(audio_caption_path, video_caption_path, output_path):
    with open(audio_caption_path, 'r') as file:
        audio_caption_data = json.loads(file.read())
        audio_caption = [
            {
                'start_formatted': entry['start'],  # 수정된 부분
                'end_formatted': entry['end'],      # 수정된 부분
                'text': entry['text']
            }
            for entry in audio_caption_data['segments']
        ]
    
    with open(video_caption_path, 'r') as file:
        video_caption = json.loads(file.read())

    merged_data = merge_captions(audio_caption, video_caption)
    save_merged_data(merged_data, output_path)
    print(f"병합된 데이터가 {output_path}에 저장되었습니다.")
    
    
    

