# GVHMR + HaMeR SMPL-X Visualization 

이 프로젝트는 **GVHMR**의 전신 모션 데이터와 **HaMeR**의 정교한 손 동작 데이터를 결합하여, **SMPL-X** 모델로 시각화하는 도구입니다. 추가적으로 캐릭터의 손목 위치에 맞춰 테니스 라켓 등의 아이템을 부착하는 기능을 포함하고 있습니다.

## 🛠 설치 방법 (Installation)

이 코드는 **Python 3.10** 환경에서 최적화되어 있습니다.

### 1. Conda 환경 생성
```bash
conda create -n gvhmr_vis python=3.10 -y
conda activate gvhmr_vis
```

### 2. PyTorch 설치 (CUDA 11.8 예시)
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```

### 3. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

## 📂 데이터 준비 (Data Preparation)

실행 전 아래 파일들이 올바른 경로에 있어야 합니다.

1.  **SMPL-X 모델**: [SMPL-X 공식 홈페이지](https://smpl-x.is.tue.mpg.de/)에서 모델 파일을 다운로드하여 `models/` 폴더에 넣어주세요.
2.  **GVHMR 결과**: `hmr4d_results.pt` 파일이 루트 폴더에 필요합니다.
3.  **HaMeR 프레임**: HaMeR에서 추출된 `.npz` 파일들이 지정된 경로(기본값: `data/hamer_out`)에 있어야 합니다.

## 🚀 실행 방법 (Usage)

경로 설정을 확인한 후 다음 명령어를 실행합니다.

```bash
python view_motion.py
```
- `Ctrl + C`: 터미널에서 강제 종료

## 🏗 프로젝트 구조
```text
gvhmr_smplx_visualizing/
├── data/               # HaMeR .npz 데이터 저장소
├── models/             # SMPL-X 모델 파일 (.npz, .pkl 등)
├── main_vis.py         # 메인 시각화 실행 스크립트
├── requirements.txt    # 의존성 패키지 목록
└── README.md           # 프로젝트 문서
```

## ⚠️ 주의 사항
- 코드 내의 `HAMER_PATH` 등 절대 경로(`C:\Users\...`)는 자신의 환경에 맞게 **상대 경로**로 수정하여 사용하세요.
- HaMeR 데이터 로드 시 프레임 번호 자릿수(예: `0001_`)가 데이터와 일치하는지 확인해야 합니다.
