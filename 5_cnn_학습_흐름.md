## 5. CNN 학습 흐름

CNN은 단순히 이미지를 넣고 결과를 뽑는 '검은 상자'가 아닙니다. 그 내부에서는 **데이터가 여러 층을 거치며 점점 똑똑한 정보로 바뀌고**, 이 과정이 반복적으로 **학습(Training)** 됩니다.

---

### 🧭 CNN 학습 과정 한눈에 보기

1. 이미지 입력
2. Convolution + ReLU → 특징 추출
3. Pooling → 정보 요약
4. 여러 층 반복 (Convolution → Pooling)
5. Flatten → 1D 벡터로 변환
6. Fully Connected Layer → 통합 판단
7. Output Layer → 결과 출력 (Softmax)
8. 손실 계산 (Loss)
9. 역전파 (Backpropagation)로 파라미터 업데이트

---

### 🔁 학습은 이렇게 반복됩니다!


입력 → 추론 → 예측 결과 → 정답과 비교 → 오차 계산 → 오차를 바탕으로 다시 파라미터 수정

이 흐름이 **수천~수만 번(Epochs)** 반복되며, CNN은 점점 더 정답에 가까운 예측을 하게 됩니다.

## 🧪 단계별 설명
### 🖼️ 1. 입력 (Input)
입력은 일반적으로 28x28 (흑백) 또는 224x224x3 (컬러 RGB) 이미지

CNN의 첫 번째 계층에 전달됨

### 🔍 2. 합성곱 + ReLU
필터(Kernel)를 이용해 이미지의 지역적 특징 추출

ReLU로 음수 제거 → 비선형성 부여

### 📉 3. 풀링
Max Pooling 등을 통해 크기 축소

연산량 감소 + 과적합 방지 + 특징 요약

### 🔁 4. 여러 층 반복
Convolution → ReLU → Pooling 과정을 2~5회 반복

깊은 층일수록 더 복잡한 특징(눈, 귀, 전체 모양 등) 인식 가능

### 📐 5. Flatten
여러 차원의 Feature Map을 1차원 벡터로 펼침

예: (7, 7, 64) → 3136차원 벡터

### 🔗 6. Fully Connected Layer
Flatten된 벡터를 Dense Layer에 입력

전체 특징들을 종합적으로 판단

### 🎯 7. Output Layer (Softmax)
각 클래스에 대해 예측 확률 출력

예: [0.02, 0.96, 0.01, 0.01] → "클래스 1"로 판단

### 🎯 손실 함수 (Loss Function)
모델이 얼마나 틀렸는지 측정하는 지표

대표 함수: Cross Entropy Loss

정답과의 차이를 계산해 학습에 활용

정답: [0, 1, 0] / 예측: [0.1, 0.8, 0.1] → 손실(Loss) ↓
정답: [0, 1, 0] / 예측: [0.2, 0.4, 0.4] → 손실(Loss) ↑

### 🔁 역전파 (Backpropagation)
손실을 줄이기 위해 가중치(Weight)를 어떻게 조정할지를 계산

CNN은 자동으로 필터(커널)의 값을 업데이트하며 학습

### 🏋️ 반복 학습
**Epoch**
전체 데이터셋을 한 바퀴 학습한 횟수

10~50 Epoch 정도 반복하며 모델이 발전

**Batch**
학습 데이터를 나눠서 학습

예: 총 데이터 10,000개 / Batch Size = 100 → 한 Epoch에 100번 학습

## 🔄 요약 흐름 다이어그램
[Input Image]  
     ↓  
[Conv → ReLU → Pooling] × N  
     ↓  
[Flatten]  
     ↓  
[Fully Connected Layers]  
     ↓  
[Softmax]  
     ↓  
[Loss Calculation]  
     ↓  
[Backpropagation]  
     ↓  
[Weight Update]  
     ↓  
Repeat for many Epochs  

## 🧠 CNN이 '학습'하는 방식이란?
CNN은 사람이 직접 특징을 지정하지 않아도, 스스로 '좋은 특징'을 찾아냅니다.

합성곱 필터는 초기엔 무작위지만, 학습을 반복하면서 점점 유의미한 패턴(선, 곡선, 윤곽 등)에 반응하도록 조정됩니다.

### 📝 Tip: CNN 학습은 마치 눈앞의 이미지를 수백 번씩 보며 점점 더 잘 알아보게 되는 사람의 학습과 비슷합니다!
