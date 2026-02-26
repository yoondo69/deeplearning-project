import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import numpy as np

# 1. CIFAR-10 데이터 로드 (Train set)
train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# 2. 라벨(정답) 추출
labels = train_ds.targets
class_names = train_ds.classes

# 3. 카운트플롯 시각화
plt.figure(figsize=(12, 5))
sns.countplot(x=labels, palette='viridis')
plt.title("CIFAR-10 Class Distribution")
plt.xlabel("Class Index")
plt.xticks(ticks=range(10), labels=class_names)
plt.ylabel("Number of Images")
plt.show()
#[출처] [DeepLearning] 시각화 그래프의 활용 - cifar10|작성자 바뿌사