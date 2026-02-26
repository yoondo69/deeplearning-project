#2607

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
data_counts = [500, 520, 480, 45, 510, 110, 495, 505, 490, 515]

data = []
for name, count in zip(class_names, data_counts):
    data.extend([name] * count)

# 데이터프레임 생성 (실제로는 pd.read_csv('my_cifar_subset.csv')를 사용하겠죠?)
df = pd.DataFrame(data, columns=['Label'])

# 2. seaborn.countplot을 활용한 시각화
plt.figure(figsize=(12, 6))
# countplot: 범주형 변수의 빈도수를 막대그래프로 그려줌
ax = sns.countplot(x='Label', data=df, palette='viridis', hue='Label', legend=False)

# 그래프 가독성 높이기
plt.title('CIFAR-10 Subset Class Distribution (Checking for Imbalance)', fontsize=15)
plt.xlabel('Class Name', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.xticks(rotation=45) # 글자가 겹치지 않게 회전
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 막대 위에 실제 숫자 표시하기 (옵션)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()