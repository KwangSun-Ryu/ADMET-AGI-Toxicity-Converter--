# Toxicity Converter 설명서

## 프로젝트 개요

- **목적**: ADMET 독성 예측 과정에서 요구되는 주요 분자 특성 식별 및 일반화 가능한 중요도 추출을 위한 특성 중요도 산출 프레임워크를 구축한다.
- **구현 내용**: 1차년도 1단계 목표로 Toxicity Converter 개발을 수행한다. 본 프레임워크는 Tox21 기반 독성 예측 데이터를 이용해 LightGBM / GBDT / CatBoost 3종 모델을 100회 반복 학습하면서 ROC-AUC 안정성과 특성 중요도 패턴을 뽑아낸다.

## 모델/런타임

- **사용 모델**

  - LightGBM (`lightgbm.LGBMClassifier`)
  - GBDT (`sklearn.ensemble.GradientBoostingClassifier`)
  - CatBoost (`catboost.CatBoostClassifier`)
- **공통 설정**

  - **반복 횟수**: 100회
  - **평가 지표**: ROC-AUC
  - : `SelectFromModel(threshold="median")`

## 디렉터리 트리 & 파일 설명

```
Toxicity Converters/
  ├─ tox_prediction.py       # Toxicity Converter 메인 스크립트
  ├─ models/                 # (실행 후) 모델 관련 아티팩트 저장 폴더
  └─ results/                # (실행 후) AUC/Feature Importance/요약 통계 저장 폴더
```

- `tox_prediction.py`: Tox21 CSV를 읽어 3개 모델(LightGBM/GBDT/CatBoost)을 100회 반복 학습·평가하고, AUC/특성 중요도를 CSV로 저장하는 스크립트
- `models/`: 추후 모델 객체 저장 확장용 폴더
- `results/`
  - `auc_by_run.csv`: run × model 단위 AUC 기록
  - `auc_summary.csv`: 모델별 AUC 평균/표준편차/최소/중앙값/최대 요약 통계
  - `feature_importance_{model}_run{run}.csv`: 각 run/model에 대해 선택된 특성과 중요도

## 실행 및 평가 흐름

1) **실행 전 준비**
   - `DATA_PATH` 를 CSV 파일 위치에 맞게 설정
   - 가상환경 패키지 설치 예시:
     ```bash
     pip install pandas numpy scikit-learn lightgbm catboost matplotlib seaborn
     ```
2) **실행 명령**
   ```bash
   cd "공개 SW/Toxicity Converters"
   python tox_prediction.py
   ```

## 내부 처리 흐름

1. **데이터 로드 및 전처리**
   - CSV 로드 → `ASSAY_NAME`, `LABEL`, `SMILES`, `Can_SMILES` 제거
   - 모든 피처 수치형 변환, inf/NaN 처리, 전부 NaN 컬럼 삭제
   - 레이블 `LABEL` 결측 행 제거
   - 극단값 mask 후 1~99% 분위수 clip으로 수치 안정화
2. **반복 학습 & 특성 선택 (100회 루프)**
   - Stratified train/test split (`test_size=0.3`, `random_state=run`)
   - `SimpleImputer(strategy="median")`로 결측값 대체
   - 각 모델에 대해: 1차 모델로 전체 특성 학습 후 중요도 추출 → `SelectFromModel(threshold="median")`로 상위 특성 선택 → 선택 피처가 0개면 중요도/표준편차 기반 최소 1개 강제 선택 → 선택 특성으로 최종 모델 학습 → 테스트 ROC-AUC 계산 및 `auc_records`에 기록 → 특성 중요도 CSV(`feature_importance_{model}_run{run}.csv`) 저장
3. **요약 통계 & 결과 출력**
   - `auc_by_run.csv`: 모든 run/model 조합의 AUC 기록
   - `auc_summary.csv`: 모델별 AUC 평균/표준편차/최소/중앙값/최대 계산 후 저장, 콘솔에도 출력
4. **(옵션) Top-10 피처 빈도 히트맵**
   - `results/feature_importance_*.csv` 집계 → run/model별 상위 10개 피처 등장 빈도(%) 계산
   - `seaborn.heatmap`으로 시각화 (화면 표시).

## 환경/운영 참고

- **데이터 경로**: `DATA_PATH`를 CSV 경로에 맞게 수정해야 한다.
- **반복 횟수 조정**: 기본 `range(100)`이지만, 실행 시간/리소스에 따라 반복 횟수는 변경 가능
