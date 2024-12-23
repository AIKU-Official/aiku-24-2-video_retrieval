from pytube import YouTube
from pytubefix import YouTube
from pytubefix.cli import on_progress
 
url = 'https://www.youtube.com/watch?v=L7wCy1IOwOw&t=468s'
 
yt = YouTube(url, on_progress_callback = on_progress)
print(yt.title)
 
ys = yt.streams.get_highest_resolution()
ys.download()

# from pytube import YouTube
# from pytubefix.cli import on_progress

# url = 'https://www.youtube.com/watch?v=noLK78Hgq0A&list=PLNbBj4TorBWdIU_4qVNKGWcpWuqDo_4GZ'

# # YouTube 객체 생성
# yt = YouTube(url, on_progress_callback=on_progress)

# # 영상 제목 출력
# print(yt.title)

# # 영상과 소리가 포함된 스트림 선택 (progressive=True)
# ys = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()

# # 다운로드
# ys.download()

# print("Download completed.")
