# Data-Analysis-Project_Preventive-Measure-against-Child-Abuse

**2주간의 짧은 프로젝트 기간 중 적은 인원으로 데이터 분석 및 영상 분석 작업을 동시에 진행하기가 원활하지 않아 표정 분석은 다른 사이트 등을 참고하였음**
- [CCTV 영상 분석 및 챗봇 프로젝트](https://github.com/yoonth95/Data-Analysis-Project_Preventive-Measure-against-Child-Abuse/blob/master/CCTV%20%EC%98%81%EC%83%81%20%EB%B6%84%EC%84%9D%20%EB%B0%8F%20%EC%B1%97%EB%B4%87%20%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8.pdf)

## 1. Overview
- Organization : 한경닷컴IT교육센터
- Project Title : 어린이집 학대 탐지 AI CCTV & 챗봇 서비스
- Project Description : 어린이집 아동학대 데이터 분석 및 AI CCTV 구현 & 챗봇 서비스 제안
- Date : 2021.07.12 ~ 2021.07.29
<br>

## 2. Dataset
- [경기도_어린이집_현황.csv](https://data.gg.go.kr/portal/data/service/selectServicePage.do?infId=0L9Q27735HPYCGJWAALS12611803&infSeq=1)
- [서울_어린이집_현황.csv](http://data.seocho.go.kr/openinf/sheetview.jsp?infId=OA-20322)
- [빅카인즈](https://www.bigkinds.or.kr/)_cctv_아동.csv
- 빅카인즈_아동학대_어린이집.csv
- SentiWord.csv
<br>

## 3. Summary
- 연일 사회적 이슈인 '아동학대' : 빈도수 및 검색량을 보면 매년 상승 추세, 그 중 유치원 보단 어린이집 아동학대에 대한 이슈가 꾸준함
- 어린이집 긍부정어 분석 및 LDA 토픽분석 : 유치원보다 학대 관련 부정적 반응이 많음
- 육아에 대한 부모의 생각 : 영유아기를 가장 힘든 시기로 뽑은 것으로 보아 양육부담을 줄이기 위해 어린이집으로 보내는 것을 확인
- CCTV 연관분석 : 어린이집, 아동학대와 CCTV가 연관도가 높은 것으로 보임
- CCTV가 있어도 꾸준히 발생 : 폭행, 정서학대 상황이 발견되기가 힘들고 사건 담당인력이 부족, CCTV 설치만으로 아동학대 예방 안 됨
- 아동학대 연관분석 및 맘카페 CCTV 빈도수 확인 : 학대의 중요한 단서가 '표정' 이라는 것<br><br>

AI CCTV로 학대 자동 탐지 (폭행 및 표정)<br>
              ↓<br>
MariaDB 데이터베이스에 학대 의심 상황 (이미지, 시간대) 저장<br>
              ↓<br>
챗봇을 통해 저장된 이미지 또는 시간대를 보육 정책 담당자 및 어린이집 원장이 당일 및 전날의 일어난 정황 전송 후 빠른 조취<br>
<br>

**※ 챗봇을 사용한 이유 ※**
- CCTV를 돌려보지 않아도 학대 정황을 빠르게 체크할 수 있음, 신고가 접수 될 경우 경찰은 해당 학대 시간대만 확인<br><br>

**※ [서울+경기_어린이집_cctv.ipynb](https://github.com/yoonth95/Data-Analysis-Project_Preventive-Measure-against-Child-Abuse/blob/master/%EC%84%9C%EC%9A%B8%2B%EA%B2%BD%EA%B8%B0_%EC%96%B4%EB%A6%B0%EC%9D%B4%EC%A7%91_cctv.ipynb)에서 CCTV의 갯수가 현원에 영향을 미치는지 중요도 확인 ※**
<img src="https://user-images.githubusercontent.com/78673090/133652201-ffdf04cd-cefa-459c-a10b-9cb4fb4d07d5.png" width="600" height="400">

- XG부스트로 CCTV와 어린이집 정원의 관계를 봤을 때 연관성이 높게 나오지만 이는 CCTV가 있는 곳으로 아이들을 보낸다기 보단 어린이집의 규모가 클수록 아이가 많은 것이고 그에 따른 CCTV 수가 많은 것으로 보임<br><br>

## 4. 개선점
- 상황적 한계 : 개인정보 사각지대, 데이터를 모으기 어려움
- 왜 이때까지 해결이 안 됐을까라는 의문점 : 실현 가능성 힘듬, 민간에서는 안 하겠지만 국공립에서는 정책차원에서 할 가능성이 있음
- 정확도가 높은 기능을 추려서 볼 수 있는 기능 추가한다면 훨씬 적용 가능성이 높아질 것임 => 효율 시간대를 추릴 수도 있음
<br>

## 5. 영상 분석
### 5.1 Yolov5_폭행 영상 분석
#### 5.1.1. 동영상 데이터 이미지로 변환 후 라벨링 작업
1. 영상 데이터 : [AI Hub 이상행동 CCTV 영상](https://aihub.or.kr/aidata/139)
2. [영상 프레임단위 이미지 저장.ipynb]() 실행하여 다운받은 폭행 영상 프레임 단위로 이미지 저장
3. labelImg-master 폴더 안에 있는 [labelImg.py]() 파일 실행하여 이미지 라벨링 작업 (폭행 이미지 위주로 라벨링 작업)
> - cmd 실행 후
> - labelImg-master 경로로 이동
> - python labelImg.py 입력 후 파이썬 파일 실행
> - train, value 폴더 생성 후 라벨링 된 이미지 나눠 저장
4. [rotateAll.py]() 파일 실행하여 라벨링 된 이미지를 회전시켜 이미지 데이터 수 증가
&nbsp;
#### 5.1.2. 라벨링한 데이터를 Colab에서 진행 (컴퓨터의 GPU, 속도 때문에)
1. yaml 파일 생성 (train, value 경로 설정 중요)
2. [yolo_detect.ipynb]() 파일을 Colab에서 실행 후 학습
3. 라벨위치 값 파일과 라벨링된 이미지 파일 등을 담은 train, value 폴더를 Colab 지정해 놓은 경로에 넣음
4. 정확도는 낮지만 속도가 빠른 yolov5s 파일로 진행 (yolo detect.ipynb에서 git clone으로 다운받은 yolov5로 폴더에 models 폴더안에 있음)
5. 학습시켜 저장된 "name.pt" 파일과 "test.mp4" 넣어서 진행
> - yolob5-master 폴더 안에 있는 detect.py 파일 실행 (yolo detect.ipynb에서 git clone으로 다운받은 yolov5로 폴더에 있음)
> - python detect.py --source "test.mp4" --weights "name.pt"
&nbsp;
&nbsp;
<br>

### 5.2 Kaggle 표정 데이터셋으로 모델링 작업 후 표정 분석
#### 5.2.1 Kaggle 표정 데이터.csv 파일 다운 후 모델링 작업 (Colab에서 진행)
1. Dataset : [표정 데이터셋](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
2. 모델링 참고 사이트 : https://www.kaggle.com/drcapa/facial-expression-eda-cnn?cellIds=1&kernelSessionId=74537191
- csv파일로 된 이미지를 변환하는 과정만 참고, 이후 모델링 작업을 직접 수정
- 참고 사이트랑 정확도는 비슷하나 Loss값은 줄일 수 있었음
- 'Facial_Expression_Recognition_model.hdf5' 파일로 저장

3. 실시간 표정 분석 참고 사이트 : https://github.com/prabhuiitdhn/Emotion-detection-VGGnet-architecture-fer2013.git
- zip 파일 다운로드 
- checkpoints에서 Facial_Expression_Recognition_model.hdf5 파일로 변경 후 진행
- python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/Facial_Expression_Recognition_model.hdf5 --video "테스트.mp4"
