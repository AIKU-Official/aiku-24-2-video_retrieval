import openai
import json
import asyncio

def convert_timestamp_to_seconds(timestamp):
    """HH:MM:SS 형식의 타임스탬프를 초 단위로 변환"""
    parts = timestamp.split(':')
    if len(parts) == 2:  # MM:SS 형식
        m, s = map(float, parts)
        return m * 60 + s
    elif len(parts) == 3:  # HH:MM:SS 형식
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    else:
        raise ValueError("타임스탬프는 'MM:SS' 또는 'HH:MM:SS' 형식이어야 합니다")

def humorous_timestamps_prompt_brooklyn(video_data, video_title, objective="Identify the funniest moments"):
    prompt_template = """
    You are an expert in comedy analysis. Your task is to review the video segments of the American TV show, "Brooklyn Nine-Nine", titled "{}".
    The objective is to select the 10 most humorous or entertaining moments based on the provided timestamps, audio captions, and video captions.

    **Show Description:**
    "Brooklyn Nine-Nine" is a popular American sitcom that follows the lives of quirky detectives in the fictional 99th precinct of the NYPD.
    The show is known for clever humor blended with exaggerated reactions. Key comedic elements include:

    - **Deadpan Delivery:** Captain Holt’s serious tone amidst absurd situations.
    - **Goofy Antics:** Jake’s impulsive behavior.
    - **Recurring Gags:** Terry’s love for yogurt and fitness, Gina’s witty self-centeredness.

    ***Characteristics of Humorous Scenes:***
    1. **Contrasting Character Interactions**: Scenes where characters with different personalities clash or complement each other, such as Jake’s carefree attitude against Holt’s seriousness.
    2. **Exaggerated Reactions**: Over-the-top responses to mundane situations, often shown by characters like Terry or Boyle.
    3. **Repetitive Gags**: Recurring jokes, like the Halloween Heist, or Terry’s obsession with yogurt.
    4. **Unexpected Situations**: Funny misunderstandings or scenarios where the characters end up in unusual predicaments.
    5. **Deadpan Humor**: Especially from Captain Holt, where he delivers lines with a serious tone that contrasts with the absurdity of the situation.

    ***Objective:*** {}

    ***Instructions:***
    - Review each segment's timestamp, image caption, and audio caption.
    - The data is presented in chronological order; consider the context from previous and following segments to understand the full humor potential of each moment.
    - Choose segments spread out across the 10-minute video to capture humor from various points in time. Aim for diversity in timing, ensuring the funniest moments aren’t clustered too closely together.
    - Select the segments that are likely to evoke laughter or be perceived as funny by viewers, especially those aligning with the characteristics of humor commonly found in "Brooklyn Nine-Nine".
    - Rank the top 10 segments in terms of humor and provide **only their timestamps** in order of funniness, starting from the funniest. Do not include descriptions or explanations, only the timestamps.

    ***Input Data Format:***
    The data is provided as a list of segments with the following fields:
    {{
        "time": Timestamp of the video segment,
        "audio_caption": Description of the audio content,
        "video_caption": Visual description of the scene
    }}

    ***Video Segments:***
    {}

    ***Output Format:***
    List only the timestamps of the top 10 funniest moments, ranked from 1 (funniest) to 10:
    - Funniest Moment #1: [Timestamp]
    - Funniest Moment #2: [Timestamp]
    - Funniest Moment #3: [Timestamp]
    ...
    - Funniest Moment #10: [Timestamp]

    ***Output Example:***
    {{
        "funniest_timestamps": [
            "00:02:30",
            "00:05:15",
            "00:10:42",
            ...
        ]
    }}
    """.format(
        video_title,
        objective,
        "".join([f"Timestamp: {segment['time']}\nAudio Caption: {segment['audio_caption']}\nVideo Caption: {segment['video_caption']}\n\n" for segment in video_data])
    )

    return prompt_template

def humorous_timestamps_prompt_home_alone_with_caption(video_data, video_title, num_clips=10, objective="Identify the funniest segments with start and end timestamps for YouTube Shorts, plus separate start timestamps within a 1-minute limit"):
    """
    Generates a prompt for identifying funny moments with titles for each segment.
    
    Args:
        video_data: Video segment data
        video_title: Video title
        num_clips: Number of clips to extract (default: 10)
        objective: Custom objective (default: None)
    """
    if objective is None:
        objective = f"Identify the {num_clips} funniest segments with start and end timestamps for YouTube Shorts"

    prompt_template = """
    You are an expert in comedy analysis. Your task is to review the video segments of the movie, "Home Alone", titled "{}".
    The objective is to select the {num} funniest or most entertaining moments based on the provided timestamps, audio captions, and video captions.

    **Movie Description:**
    "Home Alone" is a classic comedy film about a young boy, Kevin, who is left alone at home during Christmas and hilariously defends his house from two burglars using clever tricks and pranks. The film is known for slapstick humor and exaggerated reactions. Key comedic elements include:

    - **Kevin’s Clever Pranks:** Ingenious tricks Kevin uses to thwart the burglars.
    - **Slapstick Humor:** Physical comedy from the burglars encountering various traps.
    - **Exaggerated Reactions:** Over-the-top reactions from characters like the burglars facing Kevin's traps.
    
    ***Characteristics of Humorous Scenes:***
    1. **Physical Comedy**: Scenes where characters, especially the burglars, face physical humor due to Kevin’s traps.
    2. **Exaggerated Reactions**: Characters react in extreme ways to otherwise simple or mundane events, often shown by the burglars or Kevin himself.
    3. **Unexpected Situations**: Moments when Kevin or the burglars end up in unexpected or unusual predicaments.
    4. **Cleverness and Ingenuity**: Kevin’s quick thinking and clever setups to thwart the burglars.

    ***Objective:*** {}

    ***Instructions:***
    - Review each segment's timestamp, image caption, and audio caption.
    - The data is presented in chronological order; consider the context from previous and following segments to understand the full humor potential of each moment.
    - Choose segments spread out across the movie to capture humor from various points in time, ensuring the funniest moments aren’t clustered too closely together.
    - Select segments that:
        * Have a maximum duration of 1 minute 30 seconds
        * Start and end at natural scene transitions (e.g., camera angle changes, location changes, or character focus shifts)
        * Contain complete comedic moments without cutting mid-scene
    - Rank the top {num} segments in terms of humor and provide **both start and end timestamps for each segment**. Additionally, save **only the start timestamps** in a separate list.
    - For each segment, create a short and catchy YouTube Shorts title capturing the essence of the humor or key moment in that clip.

    ***Examples of YouTube Shorts Titles for Humorous Moments:***
    - "Kevin Outsmarts the Burglars Yet Again!"
    - "Epic Slip on the Ice Stairs!"
    - "Kevin's Cleverest Trap Unleashed!"
    - "The Burglars Get a Shocking Surprise!"
    - "Kevin Takes Pranks to a New Level!"

    ***Input Data Format:***
    The data is provided as a list of segments with the following fields:
    {{
        "timestamp": Timestamp of the video segment,
        "image_caption": Visual description of the scene,
        "audio_caption": Description of the audio content
    }}

    ***Video Segments:***
    {}

    ***Output Format (JSON Structure):***
    Please respond **strictly** in the following JSON format:
    
    {{
        "funniest_timestamps_full": [
            ["00:02:30", "00:03:30", "Title for Clip #1"],
            ["00:05:15", "00:06:15", "Title for Clip #2"],
            ... ({num} pairs in total)
        ],
        "funniest_start_timestamps_only": [
            "00:02:30",
            "00:05:15",
            ... ({num} timestamps in total)
        ]
    }}

    **Important:** Ensure the response strictly follows the JSON structure above, with only start and end timestamps and titles for each clip in "funniest_timestamps_full" and only start timestamps in "funniest_start_timestamps_only".
    """.format(
        video_title,
        objective,
        "".join([f"Timestamp: {segment['time']}\nVideo Caption: {segment['video_caption']}\nAudio Caption: {segment['audio_caption']}\n\n" for segment in video_data]),
        num=num_clips
    )

    return prompt_template

def humorous_timestamps_prompt_home_alone(video_data, video_title, num_clips = 10, objective="Identify the funniest segments with start and end timestamps for YouTube Shorts, plus separate start timestamps within a 1-minute limit"):
    """
    재미있는 순간을 찾는 프롬프트 생성
    
    Args:
        video_data: 비디오 세그먼트 데이터
        video_title: 비디오 제목
        num_clips: 추출할 클립 개수 (기본값: 10)
        objective: 사용자 정의 목적 (기본값: None)
    """
    if objective is None:
        objective = f"Identify the {num_clips} funniest segments with start and end timestamps for YouTube Shorts"
    
    prompt_template = """
    You are an expert in comedy analysis. Your task is to review the video segments of the movie, "Home Alone", titled "{}".
    The objective is to select the {num} funniest or most entertaining moments based on the provided timestamps, audio captions, and video captions.

    **Movie Description:**
    "Home Alone" is a classic comedy film about a young boy, Kevin, who is left alone at home during Christmas and hilariously defends his house from two burglars using clever tricks and pranks. The film is known for slapstick humor and exaggerated reactions. Key comedic elements include:

    - **Kevin’s Clever Pranks:** Ingenious tricks Kevin uses to thwart the burglars.
    - **Slapstick Humor:** Physical comedy from the burglars encountering various traps.
    - **Exaggerated Reactions:** Over-the-top reactions from characters like the burglars facing Kevin's traps.
    
    ***Characteristics of Humorous Scenes:***
    1. **Physical Comedy**: Scenes where characters, especially the burglars, face physical humor due to Kevin’s traps.
    2. **Exaggerated Reactions**: Characters react in extreme ways to otherwise simple or mundane events, often shown by the burglars or Kevin himself.
    3. **Unexpected Situations**: Moments when Kevin or the burglars end up in unexpected or unusual predicaments.
    4. **Cleverness and Ingenuity**: Kevin’s quick thinking and clever setups to thwart the burglars.

    ***Objective:*** {}

    ***Instructions:***
    - Review each segment's timestamp, image caption, and audio caption.
    - The data is presented in chronological order; consider the context from previous and following segments to understand the full humor potential of each moment.
    - Choose segments spread out across the movie to capture humor from various points in time, ensuring the funniest moments aren’t clustered too closely together.
    - Select segments that:
        * Have a maximum duration of 1 minute 30 seconds
        * Start and end at natural scene transitions (e.g., camera angle changes, location changes, or character focus shifts)
        * Contain complete comedic moments without cutting mid-scene
    - Rank the top {num} segments in terms of humor and provide **both start and end timestamps for each segment**. Additionally, save **only the start timestamps** in a separate list.
    - When selecting start and end points, prioritize:
        * Scene transitions as natural cut points
        * Complete comedic sequences
    - Do not include descriptions or explanations, only the timestamps.

    ***Input Data Format:***
    The data is provided as a list of segments with the following fields:
    {{
        "timestamp": Timestamp of the video segment,
        "image_caption": Visual description of the scene,
        "audio_caption": Description of the audio content
    }}

    ***Video Segments:***
    {}

    ***Output Format:***
    List both the start and end timestamps of the top {num} funniest segments, as well as a separate list of only the start timestamps:
    
    {{
        "funniest_timestamps_full": [
            ["00:02:30", "00:03:30"],
            ["00:05:15", "00:06:15"],
            ... ({num} pairs in total)
        ],
        "funniest_start_timestamps_only": [
            "00:02:30",
            "00:05:15",
            ... ({num} timestamps in total)
        ]
    }}
    """.format(
        video_title,
        objective,
        "".join([f"Timestamp: {segment['time']}\nVideo Caption: {segment['video_caption']}\nAudio Caption: {segment['audio_caption']}\n\n" for segment in video_data]),
        num = num_clips
    )

    return prompt_template

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # JSON 데이터를 파이썬 딕셔너리로 로드
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not a valid JSON file.")

async def get_funny_timestamps(merged_output_json_path, video_title, api_key, output_json_path, num_clips = 10):
    """
    재미있는 순간을 찾아 타임스탬프 생성
    
    Args:
        merged_output_json_path: 병합된 출력 JSON 파일 경로
        video_title: 비디오 제목
        api_key: OpenAI API 키
        output_json_path: 출력 JSON 파일 경로
        num_clips: 추출할 클립 개수 (기본값: 10)
    """
    merged_data = load_json(merged_output_json_path)
    if not merged_data:
        return None

    prompt = humorous_timestamps_prompt_home_alone(merged_data, video_title, num_clips = num_clips)
    openai.api_key = api_key
    # OpenAI Chat Completion API call for GPT-4
    response = await openai.ChatCompletion.acreate(
        model="gpt-4-turbo",  # Using the GPT-4 model
        messages=[
            {"role": "system", "content": "You are an expert in comedy analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    # Extract the text response
    output = response['choices'][0]['message']['content'].strip()
    print(output)
    
    try:
        output_data = json.loads(output)
        
        # 시작 타임스탬프만 초 단위로 변환하여 저장
        start_timestamps_seconds = [
            str(convert_timestamp_to_seconds(ts))
            for ts in output_data["funniest_start_timestamps_only"]
        ]
        
        # 기존 형식과 호환되는 형태로 저장
        compatible_output = {
            "funniest_timestamps": start_timestamps_seconds,
            # 원본 데이터도 보존
            "full_timestamps": output_data["funniest_timestamps_full"]
        }
        
        # JSON 파일로 저장
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(compatible_output, f, ensure_ascii=False, indent=4)

        return compatible_output

    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the output.")
        return None
