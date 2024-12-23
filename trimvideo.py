from pytube import YouTube
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def cut_video(input_path, start_time, end_time, output_path):
    # 비디오 전체 길이 구하기
    video = VideoFileClip(input_path)
    total_duration = video.duration
    video.close()
    
    # end_time이 None이면 비디오 끝까지
    if end_time is None:
        end_time = total_duration
        
    ffmpeg_extract_subclip(input_path, start_time, end_time, targetname=output_path)

if __name__ == "__main__":
    download_path = "/home/aikusrv02/aiku/video_retrieval/data/home_alone/home_alone.mp4"
    trimmed_path = "/home/aikusrv02/aiku/video_retrieval/data/home_alone/home_alone_4800_end.mp4"

    # 60분부터 끝까지
    start_time = 80*60  # 시작 시간(초) - 60분
    end_time = None     # 비디오 끝까지
    cut_video(download_path, start_time, end_time, trimmed_path)