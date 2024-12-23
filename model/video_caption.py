import os
import json
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def initialize_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
    return processor, model


def downsample_and_save_frames(input_video_path, output_folder, downsample_rate_seconds):
    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    downsample_rate = int(fps * downsample_rate_seconds)

    frame_idx = 0
    saved_frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % downsample_rate == 0:
            frame_filename = os.path.join(output_folder, f'frame_{saved_frame_idx}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_frame_idx += 1

        frame_idx += 1

    cap.release()


def generate_caption(img_path, processor, model):
    raw_image = Image.open(img_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


def process_video(video_path, frame_folder, output_json_path, downsample_rate_seconds=1):
    # frame_folder가 존재하지 않으면 생성
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)
        print(f"프레임 폴더가 생성되었습니다: {frame_folder}")

    # output_json_path의 디렉토리가 존재하지 않으면 생성
    output_dir = os.path.dirname(output_json_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 JSON 폴더가 생성되었습니다: {output_dir}")

    processor, model = initialize_model()
    
    downsample_and_save_frames(video_path, frame_folder, downsample_rate_seconds)
    
    frames = sorted(os.listdir(frame_folder))
    captions = []

    for idx, frame in enumerate(frames):
        img_path = os.path.join(frame_folder, frame)
        time_str = f"{idx // 3600:02}:{(idx % 3600) // 60:02}:{idx % 60:02}"
        caption = generate_caption(img_path, processor, model)
        data = {
            "time": time_str,
            "caption": caption
        }
        captions.append(data)

    with open(output_json_path, 'w') as json_file:
        json.dump(captions, json_file, indent=4)
        print(f"캡션이 저장되었습니다: {output_json_path}")

    return captions