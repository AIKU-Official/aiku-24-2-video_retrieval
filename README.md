# Video retrieval

>> 📢 2024년 2학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

## 소개

숏폼 콘텐츠가 현재 핵심 트렌드로 자리 잡았지만, 기존의 콘텐츠 편집 과정은 많은 시간과 비용을 요구합니다. 특히, 긴 영상에서 흥미로운 순간을 찾고 이를 편집하여 쇼츠로 제작하는 과정은 매우 비효율적입니다. 저희 프로젝트는 모델을 활용해 이 과정을 자동화하고자 했습니다. whsiper, blip을 활용해 영상에서 text를 추출할 수 있도록 하고 LLM을 활용해 이 text에서 영상의 특정 부분을 선택할 수 있도록 했습니다. 해당 파이프라인을 이용해 '나홀로집에' 영화에서 웃긴 영상을 뽑았을 때, 팀원들이 모두 인정할만한 재밌는 영상을 찾을 수 있었습니다.

## 방법론
1. audio to text (whisper), image to text (blip) 를 통해 원본 영상에서 text 추출
2. text를 바탕으로 LLM을 활용해 사용자의 요구를 반영한 특정 타임스탬프 출력
3. 추출된 쇼츠에서 행동 및 감정을 분석하는 알고리즘을 이용하여 제목 생성
86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-12-25_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.05.54.png)

## 결과

https://github.com/user-attachments/assets/dcae2979-9757-443e-b589-70cfc8fe2709


## 팀원

- [서연우](https://github.com/readygetset): 문제 정의, 전체 모델 파이프라인 구성, 데이터셋 수집
- [김지연](https://github.com/delaykimm):코드 작성, 논문 리서치, 프롬포트 작성, video caption 추출
- [성유진](https://github.com/dinyudin203): 코드 작성, 눈문 리서치, STT 추출
- [송현지](https://github.com/kelly062001): 코드 작성, 논문 리서치, STT 추출
