# 선박 안전 운항 보조시스템
###  디지털 스마트 부산 2팀

- 한창훈 : 영상 처리, 서버 구축
- 조상은 : 영상처리, FLASK, HTML
- 박태언 : 라벨링, 모델 학습
- 문지민 : 라벨링, 모델 학습

## Features
- 선두의 부유물과 배를 탐지
- 선교의 사람 탐지
- 카메라에 탐지된 선박,부유물 Tracking -> 방향
- 사람의 고개 각도 계산
- 캡쳐 이벤트 발생 시 캡쳐된 이미지 인코딩하여 DB에 저장 이 후 디코딩 되어 HTML로 전송되어 웹에서 확인

> 선교에 사람이 없다면 알람 (선원법, 해사안전법)
> 전방에 배 또는 부유물이 있지만 선교에 선원의 고개 각도가 아래를 보고 이
있다면 알람을 울림 (전방 견시 부주의 방지)

## Tech

- [Yolov5] - Deep learning Object Detection
- [OpenCV] - Computer Vision
- [MediaPipe] - Open-source framework for building pipelines to perform computer vision inference over arbitrary sensory data such as video or audio
- [MySQL] - Client/server system that consists of a multithreaded SQL server
- [Flask] -  A web framework
- [HTML5] -  A markup language used for structuring and presenting content on the World Wide Web

## Practice

콘다 콘솔 가상환경에서 실행.
```sh
conda activate smart_port
cd 파이썬 파일 경로

python track.py --source 0
```
새 콘솔 창을 하나 더 띄어서 실행
```sh
conda activate smart_port
cd 파이썬 파일 경로

python person.py
```

## AWS
AWS - 5000번 포트 개방 후 사용

FLASK 프레임워크를 올려 두 개의 파이썬 모듈간의 통신을 제공하고 웹 페이지를 배포

## 구현결과
<img width="80%" src="https://user-images.githubusercontent.com/73980198/204086773-19aee76f-089b-4ce7-be6f-45793a5bf20d.png"/>