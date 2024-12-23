import whisper_timestamped as whisper
import whisper
import subprocess
import json
import torch

def transcribe_video_to_json(input_file, output_audio_file, output_json_file, language="en", device=None):
    """
    비디오 파일에서 오디오를 추출하고, Whisper 모델을 사용해 트랜스크립션을 진행한 후 결과를 JSON 파일로 저장하는 함수.

    Parameters:
    input_file (str): 입력 비디오 파일 경로
    output_audio_file (str): 추출된 오디오 파일 경로 (기본값: 'output.m4a')
    output_json_file (str): 결과 JSON 파일 경로 (기본값: 'transcription_result.json')
    language (str): 트랜스크립션에 사용할 언어 (기본값: 'en')
    
    Returns:
    None (결과는 JSON 파일로 저장)
    """
    print(f"Input file path: {input_file}")
    
    try:
        subprocess.run(['ffmpeg', '-i', input_file, '-vn', '-acodec', 'aac', '-b:a', '192k', output_audio_file], check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("FFmpeg 오류:", e.stderr.decode())
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for audio transcription: {device}")
    
    model = whisper.load_model("base", device=device)
    result = model.transcribe(output_audio_file)
    
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Transcription saved to {output_json_file}")


# input_file = '/home/aikusrv02/aiku/video_retrieval/data/brooklyn/brooklyn_nine-nine.mp4'
# transcribe_video_to_json(input_file, output_audio_file="output_audio.m4a", output_json_file="output_transcription.json")
