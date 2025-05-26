학습 실행 


python3 /home/nfsyang/deep-learning-demo/newline/constra_train.py \
  --train_data_root /home/nfsyang/deep-learning-demo/0-waterdeer-vs-wildboar-dataset/train \
  --classes water-deer wild-boar \
  --ratio 1:1 \
  --oversampling
  --save_root /home/nfsyang/newline/saved_runs


----------------이거 쓸거임 ----------------
실험 목적	명령어 예시
✅ baseline (둘 다 꺼짐)	(아무 옵션도 안 씀)


python3 train.py \
  --train_data_root ... \
  --classes water-deer wild-boar \
  --ratio 1:0.01
| ✅ contrastive only | --contrastive 추가 |


python3 train.py \
  --train_data_root ... \
  --classes water-deer wild-boar \
  --ratio 1:0.01 \
  --contrastive
| ✅ oversampling only | --oversampling 추가 |


python3 train.py \
  --train_data_root ... \
  --classes water-deer wild-boar \
  --ratio 1:0.01 \
  --oversampling
| ✅ both (둘 다 사용) | --contrastive --oversampling 둘 다 추가 |


python3 train.py \
  --train_data_root ... \
  --classes water-deer wild-boar \
  --ratio 1:0.01 \
  --contrastive \
  --oversampling
  ----------------------------------------------



검증 
model_root /home/nfsyang/newline/saved_runs/baseline을 돌리면 하위 비율별 폴더의 run을 싹 읽고 베스트 pth (max) 값을 뽑아서 테스트함. 


python3 /home/nfsyang/deep-learning-demo/newline/validation.py \
  --model_root /home/nfsyang/newline/saved_runs/baseline \
  --test_data_root /home/nfsyang/deep-learning-demo/0-waterdeer-vs-wildboar-dataset/valid \
  --classes water-deer wild-boar



  전체 프로젝트 디렉토리 구조

project_root/
│
|── test.py                  # 테스트 코드
├── train.py                 # 학습 코드
├── validation.py            # validation 코드 (best run 자동 선택)
├── utils.py                 # 로깅, 스케줄러 등 유틸
├── saved_runs/              # 학습 결과 자동 저장
│   └── baseline/
│       └── 1:0.01(wil)/
│           └── run_1/
│               ├── best_model.pth
│               └── train.log
|   └── constrastive/
|
|   └── oversampling/
|
|   └── constrastive_oversampling/
|
|
|
├── dataset/
│   ├── train/
│   │   ├── water-deer/
│   │   └── wild-boar/
│   └── valid/
│       ├── water-deer/
│       └── wild-boar/
| dataset에 빈 폴더 해놨으니까 여기 데이서 셋 넣으면 된다.
----------------------------------gpt 설명 코드 (참고하라고)-------------------------
Train.py
✅ 기본 명령어
bash
복사
편집
python3 train.py \
  --train_data_root ./dataset/train \
  --classes water-deer wild-boar \
  --ratio 1:0.01 \
  --contrastive \
  --oversampling \
  --save_root ./saved_runs
✅ 인자 설명
인자	설명
--train_data_root	학습 데이터가 들어있는 상위 폴더 (ImageFolder 구조)
--classes	학습에 사용할 클래스 이름 2개 (서브폴더 이름 기준)
--ratio	클래스 간 비율 설정 (예: 1:0.01)
--contrastive	컨트라스티브 손실 포함 여부 (--contrastive 옵션만 넣으면 적용됨)
--oversampling	클래스 불균형을 오버샘플링으로 보정
--save_root	실험 결과 저장 루트 폴더 (자동으로 run_1, run_2... 생성됨)

🗂️ 학습 결과는 자동으로 저장됨
bash
복사
편집
saved_runs/baseline/1:0.01(wil)/run_1/
├── best_model.pth
└── train.log
폴더명 규칙:

실험 유형: baseline, contrastive, oversampling, contrastive_oversampling

비율+클래스: 1:0.01(wil) 형식으로 폴더 생성됨



✅ 2. 검증: validation.py 실행 방법
✅ 기본 명령어
bash
복사
편집
python3 validation.py \
  --model_root ./saved_runs/baseline \
  --test_data_root ./dataset/valid \
  --classes water-deer wild-boar
✅ 인자 설명
인자	설명
--model_root	학습 결과 상위 경로 (각 run_x 폴더 포함)
--test_data_root	검증 데이터 루트 폴더 (ImageFolder 구조)
--classes	평가할 클래스 이름 2개 (학습 시 사용한 것과 동일해야 함)

✅ 기능 요약
model_root 하위 폴더(ex: 1:0.01(wil))에서 모든 run_x 중 train.log의 Train Acc가 가장 높은 run을 선택

해당 run의 best_model.pth로 검증

클래스별 acc, 전체 acc, CE Loss 출

------------------
실험 비율 여러 개 돌리고 saved_runs/에 쌓으면, validation.py 하나로 자동 평가 가능
--------------------


------------------test----------
 python3 validation.py \
  --model_path '/home/nfsyang/deep-learning-demo/imbalance-test/saved/train/1:1(roe)/run_3/best_model.pth' \
  --test_data_root '/home/nfsyang/deep-learning-demo/aihub-data-cropped/valid/'



 python3 validation.py \
  --model_path '/home/nfsyang/deep-learning-demo/imbalance-test/saved/train/1:1(roe)' \
  --test_data_root '/home/nfsyang/deep-learning-demo/aihub-data-cropped/valid/'




이게 진짜 실행코드 (test.py)

python3 /home/nfsyang/deep-learning-demo/newline/test.py \
  --model_root /home/nfsyang/deep-learning-demo/newline/trained_model/ \
  --test_data_root /home/nfsyang/deep-learning-demo/1-waterdeer-vs-roedeer-dataset/valid 