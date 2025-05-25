# 2025-AI-캡스톤
## Overview
ResNet18 모델 사용한 고라니와 노루 fine-grained and data imbalance 문제 해결
## Process
### 0. Installation
`
python3나 필요한 것들 다운하는 명령어
`
#### 0-1. AIhub dataset
<details>
<summary><strong>[Click to expand]</strong></summary>
This project uses a wildlife image dataset provided by AIHub, a public data platform operated by the Korean government.  
Due to license and privacy restrictions, the dataset is **not included in this repository** and must be downloaded manually by the user.

- Source: [AIHub - Wildlife Image Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=53)
- Description: Images of 11 wild animal species (e.g., boar, roe deer) captured by infrared and normal cameras
- Purpose: For training and evaluating image classification models
##### 1) How to download
1. Go to the [AIHub dataset page](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=53)
2. Sign up and log in
3. Agree to the terms of use and request access
4. Download the provided dataset file (usually in .zip format)

##### 2) Directory structure (example)
After extracting the dataset, please organize it as follows:
<pre><code>``` 
project_root/ ├── data/ │ ├── train/ │ │ ├── class_01_boar/ │ │ ├── class_02_roe_deer/ │ │ └── ... │ ├── valid/ │ └── test/ 
``` </code></pre>
※ You may need to manually split the data into train/validation/test sets and rename folders accordingly.  
※ If a preprocessing script (e.g., `prepare_dataset.py`) is provided, you can automate this step.
##### 3) License Notice
This dataset is provided **for non-commercial, research purposes only**.  
Please make sure to review and comply with the AIHub [Terms of Use](https://www.aihub.or.kr/guide/terms) before using the data in your project or publication.
</details>

### 1. git clone
```
git clone https://github.com/lxxsxoh/Test-using-ResNet18-for-fine-grained-problem.git
```
### 2. 폴더 확인
```
ls Test-using-ResNet18-for-fine-grained-problem
```
### 3. Training (with AIhub data, trian 과정이 필요 없다면 바로 4번으로)
#### 3-1. 폴더 접속
```
cd Test-using-ResNet18-for-fine-grained-problem
```
#### 3-2. train
```
python3 [파일명(예: train].py [OPTIONS]
```
|Argument|Type|Default|Description|
|-----|-----|-----|-----|
|--train_data_root|path|Default|train data 가져올 경로|
|--num_classes|int|2|분류할 class 수|
|--batch_size|int|64|배치사이즈|
|--num_workers|int|4|사용할 core 수|
|--비율|int|1|고라니 대비 노루 데이터 비율(또는 수?)|
### 4. Test Using pretrained model (별도의 train 없이 pretrianed model 사용하여 test만)
```
python3 [파일명(예: test)].py [OPTIONS]
```
|Argument|Type|Default|Description|
|-----|-----|-----|-----|
|--model_path|path|Default|pretrained weight 가져올 경로|
|--test_data_root|path|Default|train data 가져올 경로|
|--num_classes|int|2|분류할 class 수|
|--batch_size|int|64|배치사이즈|
|--num_workers|int|4|사용할 core 수|
|--비율|int|1|고라니 대비 노루 데이터 비율(또는 수?)|
### 5. Result Example
`
코드 파일 만들어지면 돌려보고 accuracy 측정되는 거 캡쳐해서 올리기?
`
