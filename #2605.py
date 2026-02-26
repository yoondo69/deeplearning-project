#2605
import pandas as pd            # 데이터 분석용 (pd)
import numpy as np             # 수치 계산용 (np)
import matplotlib.pyplot as plt # 그래프 그리기용 (plt)
import seaborn as sns          # 예쁜 그래프 그리기용 (sns)
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

# [상황 설정]
# 모델의 마지막 레이어(FC Layer) 직전에서 나온 64차원 특징 벡터라고 가정
# 10개 클래스가 뭉쳐 있긴 하지만, 일부는 겹쳐 있는 상황
X_features, y_labels = make_blobs(n_samples=1000, centers=10, n_features=64, 
                                  cluster_std=2.0, random_state=42)

# 1. t-SNE로 2차원 축소 (시간이 좀 걸릴 수 있음)
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_embedded = tsne.fit_transform(X_features)

# 2. 산점도 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_labels, cmap='jet', alpha=0.6)
plt.colorbar(scatter, label='Class Label')
plt.title('t-SNE Visualization of Feature Space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
#[출처] [DeepLearning] 시각화 그래프의 활용 - cifar10|작성자 바뿌사

