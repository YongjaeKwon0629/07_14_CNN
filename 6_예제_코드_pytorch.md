## 6. 예제 코드 (PyTorch)

이 섹션에서는 PyTorch를 사용한 **간단한 CNN 모델 구현 예제**를 소개합니다.  
MNIST(손글씨 숫자 이미지)처럼 크기가 작은 28x28 이미지에 적합한 구조로 설계되어 있습니다.

---

### 🧱 모델 구조

- 입력 이미지 크기: 28x28, 채널 1개 (흑백)
- Convolution → ReLU → Max Pooling
- Convolution → ReLU → Max Pooling
- Flatten → Fully Connected → 출력 (10 클래스)

---

### 🧪 코드 구현

```
import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN 클래스 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 첫 번째 합성곱 층: 입력 채널 1개, 출력 채널 32개, 필터 크기 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # 두 번째 합성곱 층: 입력 32개, 출력 64개
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 풀링 층 (Max Pooling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 완전 연결층 (Flatten 후 연결): 7x7x64 → 10 클래스
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 숫자 0~9 분류

    def forward(self, x):
        # 첫 번째 Conv → ReLU → Pooling
        x = self.pool(F.relu(self.conv1(x)))  # 출력: (32, 14, 14)

        # 두 번째 Conv → ReLU → Pooling
        x = self.pool(F.relu(self.conv2(x)))  # 출력: (64, 7, 7)

        # Flatten (벡터로 펴기)
        x = x.view(-1, 64 * 7 * 7)

        # Fully Connected → ReLU → Fully Connected → Softmax
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```          
