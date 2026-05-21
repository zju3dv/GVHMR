# Hand Temporal Model - 학습 가이드

## 요구 사항

학습을 실행하기 전에 아래 파일과 데이터셋을 준비해야 합니다.

---

## 1. MANO 모델 준비

`mano/` 폴더 안에 아래 두 파일을 넣어주세요. (이미 있음)

```
mano/
├── MANO_LEFT.pkl
└── MANO_RIGHT.pkl
```

MANO 모델은 [MANO 공식 사이트](https://mano.is.tue.mpg.de/)에서 다운로드할 수 있습니다.

---

## 2. HO3D 데이터셋 준비

### 2-1. 데이터셋 다운로드
[HO3D 공식 git](https://github.com/shreyashampali/ho3d)에서 v3 데이터셋을 다운로드합니다.

### 2-2. 전처리 실행
```bash
python prepare_ho3d.py
```

### 2-3. 결과물 확인
전처리가 완료되면 아래 파일이 생성됩니다.
```
ho3d_train.npz
```

---

## 3. InterHand2.6M 데이터셋 준비

### 3-1. 데이터셋 다운로드
[InterHand2.6M 공식 사이트](https://mks0601.github.io/InterHand2.6M/) -> [annotation](https://drive.google.com/drive/folders/1MJ4ztwhMJ1RyFdsxzAprExvzr1cV6otS)에서 아래 세 가지를 다운로드합니다.

```
images/
annotations/
    ├── train/
    └── val/
```

### 3-2. 데이터 배치
다운로드한 파일을 프로젝트 폴더에 넣어주세요.

### 3-3. 전처리 실행
```bash
python prepare_interhand26.py
```

### 3-4. 결과물 확인
전처리가 완료되면 아래 두 파일이 생성됩니다.
```
interhand_train.npz
interhand_val.npz
```

---

## 4. 최종 폴더 구조 확인

학습 전 아래 파일들이 모두 있는지 확인해주세요.

```
프로젝트 폴더/
├── mano/
│   ├── MANO_LEFT.pkl
│   └── MANO_RIGHT.pkl
├── ho3d_train.npz
├── interhand_train.npz
├── interhand_val.npz
└── train.py
```

---

## 5. 학습 실행

```bash
python train.py
```

학습이 완료되면 `hand_temporal_model.pth` 파일이 생성됩니다.