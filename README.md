# 전기차 충전 수요 예측 프로젝트 (Spatio-Temporal Graph Neural Network)

## 1. 프로젝트 개요 (Project Overview)
본 프로젝트는 성남시 전기차 충전소의 과거 이력 데이터를 바탕으로 미래의 충전 수요를 예측하는 것을 목표로 합니다. STGCN(Spatio-Temporal Graph Convolutional Network)을 기반 모델로 사용합니다. GAT모듈을 추가로 사용할 수 있습니다. (WP 코드는 학교 저작권 문제로 포함하지 않았습니다.)

## 2. 디렉토리 구조 (Directory Structure)
```
.
├── data/                           # 데이터 파일 저장 위치
│   └── ours/           
│       ├── adj.npz
│       ├── train.npz
│       ├── val.npz
│       └── test.npz
├── stgcn/    
│   ├── script/
│   │   └── dataloader.py           # 모델 아키텍처 스크립트
│   ├── train.py
│   └── visualize.py
├── Graph-WaveNet/                  # 유틸리티 스크립트
│   ├── train.py
├── make_history_station_csv.py     # 1. history.csv 데이터 생성 코드
├── generate_data.py                # 2. history.csv와 station.csv 정보로 수요 집계 데이터와 인접 행렬 생성 코드
├── generate_training_data.py       # 3. 모델 훈련/검증/테스트 데이터 생성 코드
├── requirements.txt                # 파이썬 라이브러리 의존성 파일
├── README.md                       # 프로젝트 안내 문서
```

## 3. 설치 및 환경 설정 (Setup)

### 3.1. Python 가상환경 생성
```bash
# Python 3.8 이상 권장
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
```

또는 conda 사용
```bash
conda create -n EmpactLabs  
conda activate EmpactLabs 
```

### 3.2. 필요 라이브러리 설치
프로젝트에 필요한 모든 라이브러리는 requirements.txt 파일에 명시되어 있습니다. 아래 명령어로 한 번에 설치할 수 있습니다.

```bash
pip install -r requirements.txt
```

## 4. 실행 방법 (How to Run)

### 4.1. 데이터 준비
`data/ours_doc_bfs/` 경로에 `adj.npz`, `train.npz`, `val.npz`, `test.npz` 파일이 있는지 확인합니다. 데이터 생성이 필요한 경우, 아래 스크립트를 먼저 실행해야 합니다. 

데이터 생성을 위해서 이미 전처리된 파일(`vel.csv`: 수요를 한 시간 단위로 집계한 파일, `adj.npz`: 충전소의 인접행렬)이 있어야 합니다.

**점유율 기반 수요 합산과 인접행렬 파일 생성**
```bash
python generate_data.py \
    --history_path history.csv \
    --station_path station.csv \
    --output_dir ./output # 임의 설정
```

**모델을 위한 데이터 생성**
```bash
python generate_training_data.py 
```

### 4.2. 모델 학습 (Training)

#### 4.2.1. STGCN 모델 (stgcn폴더)
`train.py` 스크립트를 사용하여 모델을 학습합니다. `--graph_conv_type` 인자를 통해 Base, GAT 모델을 선택할 수 있습니다. 데이터 변경 시 (이름이나, 충전소 개수) `stgcn/script/dataloader.py`에서 수정 필요한 부분에 수정을 해야 합니다. (주석으로 표시함)

**예시: 피쳐 4개로 396개의 충전소의 48시간 동안의 수요 예측**
```bash
python stgcn/train.py --dataset ours --n_his 168 --n_pred 48 
```

**예시: 피쳐 4개로 396개의 충전소의 48시간 동안의 수요 예측 + GAT 모듈 사용 (성능 올리기 위해)**
```bash
python stgcn/train.py --dataset ours --n_his 168 --n_pred 48 --cheb_graph_conv gat --num_heads 2 --gat_dropout 0.1
```

GAT의 `num_heads`는 `[2, 4]`, `gat_dropout`은 `[0.1, 0.3]` 정도로 모델 성능을 비교한 뒤 최고 성능인 것을 선택하면 됩니다.

학습이 완료되면 가장 성능이 좋았던 모델의 가중치가 `stgcn/checkpoint` 폴더의 `STGCN_ours_doc.pt`와 같은 이름으로 저장됩니다.

#### 4.2.2. GraphWaveNet 모델 (Graph-WaveNet 폴더)
```bash
python Graph-WaveNet/train.py --data ./data/ours --seq_length 48 --in_dim 7 --num_nodes 392
# gat 사용 시 --gat 추가, 과거 시퀀스 길이가 너무 길면 메모리 부족 문제 발생함
```

### 4.3. 결과 시각화 및 평가 (Visualization & Evaluation)
`visualize.py` 스크립트를 사용하여 학습된 모델의 성능을 평가하고 예측 결과를 시각화합니다.

**예시: stgcn의 2개 모델 비교 시각화**
시각화 결과는 `stgcn/visuals`에 저장됨. 예시로 폴더에 시각화 결과가 있으니 참고.

```bash
python stgcn/visualize.py \
    --dataset ours \
    --base_ckpt "path/to/STGCN_BASE.pt"
    # --gat_ckpt "path/to/STGCN_GAT_BEST.pt" # 지금 코드는 생략 
```

참고용으로 `STGCN_BASE.pt`와 `STGCN_GAT_BEST.pt` 파일을 포함하여 시각할 수 있도록 하였습니다.

- `--*_ckpt` 인자에 각 모델의 저장된 가중치 파일 경로를 정확히 입력해야 합니다.
- 이 스크립트는 학습 때 사용했던 최적 하이퍼파라미터를 불러와야 정확한 결과가 재현됩니다.
