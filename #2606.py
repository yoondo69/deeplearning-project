#2606

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 재현성을 위한 시드 설정
torch.manual_seed(42)
np.random.seed(42)

# =========================================================
# 1. 가상 데이터 생성 (고양이와 개를 헷갈리게 만들기)
# =========================================================
n_samples = 2000
n_features = 20
n_classes = 10

# 10개 클래스의 중심점 생성
centers = np.random.uniform(-10, 10, (n_classes, n_features))

# [핵심] 고양이(3)와 개(5)의 중심점을 매우 가깝게 설정 (오분류 유도)
centers[5] = centers[3] + np.random.uniform(-1.5, 1.5, n_features) 

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, 
                  cluster_std=2.5, random_state=42) # std를 키워 분포를 겹치게 함

# 데이터 분할 및 텐서 변환
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# =========================================================
# 2. 모델 및 학습 함수 정의
# =========================================================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.fc(x)

def train_and_eval(weights=None, title="Model"):
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # [전략적 지시 핵심] 가중치 적용 여부
    if weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
    else:
        criterion = nn.CrossEntropyLoss()
        
    # 학습 (Epoch 100)
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
    # 평가
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        
    return confusion_matrix(y_test.numpy(), preds.numpy())

# =========================================================
# 3. 비교 실험 실행
# =========================================================

# Case A: 일반적인 학습 (Baseline)
cm_baseline = train_and_eval(weights=None, title="Baseline")

# Case B: 전략적 학습 (Strategic - Weighted Loss)
# 고양이(3)를 틀리면 벌점을 3배로 부여! ("이건 꼭 맞춰!")
class_weights = [1.0] * 10
class_weights[3] = 3.0  # 고양이(Class 3)에 가중치 3배
# class_weights[5] = 1.5 # (선택) 개(Class 5)도 조금 신경 쓰게 하려면 추가 가능

cm_strategic = train_and_eval(weights=class_weights, title="Strategic")

# =========================================================
# 4. 결과 시각화
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
labels = [f'{i}' for i in range(10)]
labels[3] = 'Cat(3)'
labels[5] = 'Dog(5)'

# Baseline Heatmap
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=labels, yticklabels=labels)
axes[0].set_title("1. Baseline (Standard)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

# Strategic Heatmap
sns.heatmap(cm_strategic, annot=True, fmt='d', cmap='Reds', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title("2. Strategic (Weighted Loss on Cat)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

# 하이라이트: 고양이(3) 행의 변화 확인
plt.show()

# 수치 비교 출력
def get_cat_accuracy(cm):
    cat_idx = 3
    correct = cm[cat_idx, cat_idx]
    total = cm[cat_idx].sum()
    return correct, total

base_corr, base_tot = get_cat_accuracy(cm_baseline)
strat_corr, strat_tot = get_cat_accuracy(cm_strategic)

print(f"\n[고양이(Cat) 분류 성적표]")
print(f"1. 일반 모델: {base_tot}마리 중 {base_corr}마리 정답 (정확도: {base_corr/base_tot*100:.1f}%)")
print(f"2. 전략 모델: {strat_tot}마리 중 {strat_corr}마리 정답 (정확도: {strat_corr/strat_tot*100:.1f}%)")
print(f"-> 전략 적용 후 고양이 정답률 변화 확인 필요")
#[출처] [DeepLearning] 시각화 그래프의 활용 - cifar10|작성자 바뿌사