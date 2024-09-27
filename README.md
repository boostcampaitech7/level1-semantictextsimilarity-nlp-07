# NLP Mini Project

![1726648039867](./docs/image/README/1726648039867.png)

## 📕프로젝트 개요

* 부스트캠프 AI Tech `NLP`분야에서 개체된 NLP 기초 대회
* `문맥적 유사도(STS)`를 측정하는 Task
  * 의미 유사도 판별(Semantic Text Similarity, STS)이란 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 자연어처리 Task
* 학습 데이터셋은 9,324개, 검증 데이터는 550개, 테스트 데이터는 1,100개로 테스트 데이터 중 50%만 Public으로 반영 대회 종료 후 Private 점수가 공개됨.
* `피어슨 상관계수`를 통한 평가.

## 📆세부일정
* 프로젝트 기간(2주) : 09.10(화) ~ 09.26(목)

## 😁팀소개

| 강감찬 | 이채호 | 오승범 | 이서현 | 유채은 | 서재덕 |
| :-: | :-: | :-: | :-: | :-: | :-: |
||![꼬부기2](./docs/image/README/꼬부기2.png)|![꼬부기3](./docs/image/README/꼬부기3.png)||||
|[@감찬](https://github.com/gsgh3016)|[@채호](https://github.com/chell9999)|[@승범](https://github.com/Sbeom12)|[@서현](https://github.com/seohyeon0677)|[@채은](https://github.com/canolayoo78)|[@재덕](https://github.com/jduck301)|
|고독한 음악인|혼자있는 지방러|게임이 하고 싶은 승범|서현 막내|야구가 싫은 채은|재덕=? 오리|

## 프로젝트 수행 절차 및 방법
🔄 프로젝트 순환 프로세스: 가설 ➡️ 실험 ➡️ 검증  
1. 🔍 데이터 EDA (탐색적 데이터 분석)  
   * 데이터 분포 확인  
   * 이상치 및 결측치 탐지  
   * 특성 간 상관관계 분석
       
2. 🔬 데이터 증강
   * 텍스트 변형 기법 적용(RTT)
   * 동의어/유의어 치환
   * KoEDA
     
4. 🤖 모델 
   * SOTA 모델 비교 분석
   * 태스크 특화 모델 탐색
   * 앙상블 기법 고려
     
5. ⚙️ 하이퍼파리마터 튜닝
   * Optuna 실험
   * learning rate 조절
     

## 프로젝트 아키텍쳐

## 프로젝트 폴더 구조
📦level1-semantictextsimilarity-nlp-07  
 ┣ 📂docs  
 ┃ ┗ 📂image  
 ┃ ┃ ┗ 📂README  
 ┃ ┃ ┃ ┗ 📜1726648039867.png  
 ┣ 📂logger  
 ┃ ┗ 📜__init__.py  
 ┣ 📂saved  
 ┃ ┗ 📜log  
 ┣ 📂src  
 ┃ ┣ 📂callback  
 ┃ ┃ ┣ 📜early_stopping.py  
 ┃ ┃ ┣ 📜epoch_print_callback.py  
 ┃ ┃ ┣ 📜learning_rate_monitor.py  
 ┃ ┃ ┣ 📜model_checkpoint.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂config  
 ┃ ┃ ┣ 📜data_loader_config.py  
 ┃ ┃ ┗ 📜path_config.py  
 ┃ ┣ 📂data_loader  
 ┃ ┃ ┣ 📜dataset.py  
 ┃ ┃ ┗ 📜loader.py  
 ┃ ┣ 📂model  
 ┃ ┃ ┗ 📜model.py  
 ┃ ┣ 📂preprocessing  
 ┃ ┃ ┗ 📜preprocessor.py  
 ┃ ┣ 📂tokenizing  
 ┃ ┃ ┗ 📜tokenizing.py  
 ┃ ┗ 📂trainer  
 ┃ ┃ ┗ 📜predict.py  
 ┣ 📜main.py  
 ┣ 📜README.md  
 ┣ 📜requirements.txt  
 ┣ 📜test.py  
 ┗ 📜train.py  

## Appendix

### 협업방식

* Notion
* Git  

### 
