#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import tree
X = [[0,0],[1,1]]
Y = [0,1]


# In[3]:


clf = tree.DecisionTreeClassifier()
clf


# In[4]:


# 학습
clf = clf.fit(X,Y)
clf


# In[6]:


clf.predict([[2.,2.]])


# In[7]:


clf.predict_proba([[2.,2.]])
# 비율로 출력이 된다.


# In[ ]:


#load_iris
# 복구 실행


# In[8]:


from sklearn.datasets import load_iris
from sklearn import tree
X, y = load_iris(return_X_y = True)
X, y


# In[9]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


# In[10]:


# tree.plot_tree(clf.fit(iris.data, iris.target))
tree.plot_tree(clf.fit(X, y))


# In[11]:


# 색이 없어서 색을 넣어주자
tree.plot_tree(clf.fit(X, y), filled=True)


# In[12]:


# 트리 작아 확대하자


import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
tree.plot_tree(clf.fit(X, y), filled=True)


# In[ ]:


# 그래프 이쁘게 그릴려면 graphviz사용 - 따로 설치
# pip install graphviz
# plt는 안이쁘대


# In[ ]:





# 데이터 구성
# 
# Pregnancies : 임신 횟수
# 
# Glucose : 2시간 동안의 경구 포도당 내성 검사에서 혈장 포도당 농도
# 
# BloodPressure : 이완기 혈압 (mm Hg)
# 
# SkinThickness : 삼두근 피부 주름 두께 (mm), 체지방을 추정하는데 사용되는 값
# 
# Insulin : 2시간 혈청 인슐린 (mu U / ml)
# 
# BMI : 체질량 지수 (체중kg / 키(m)^2)
# 
# DiabetesPedigreeFunction : 당뇨병 혈통 기능
# 
# Age : 나이
# 
# Outcome : 768개 중에 268개의 결과 클래스 변수(0 또는 1)는 1이고 나머지는 0입니다.

# # 2 필요한 라이브러리 로드
# 

# In[13]:


# 데이터 분석을 위한 pandas
# 수치계산을 위한 numpy
# 시각화 seaborn, matplotlib.pyplot
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # 3 데이터셋 로드

# In[15]:


df = pd.read_csv('data/diabetes.csv')
df.shape


# In[16]:


df.head()


# # 4 학습, 예측 데이터셋 나누기

# In[18]:


# 8:2 의 비율로 구하기 위해 전체 데이터의 행에서 80% 위치에 해당되는 값을 구해서
# split_count라는 변수에 담는다
# df.shape[0] # 행의 값만 들고와

split_count = int(df.shape[0]*0.8)
split_count


# In[20]:


# train, test로 슬라이싱을 통해 데이터로 나눈다
# df로 전체를 부른다
train = df[:split_count].copy()
train
# split_count 는 614이기에 0~613번호까지 불러옴


# In[21]:


train.shape


# In[24]:


test = df[split_count:].copy()
test.shape


# # 5학습,예측에 사용할 컬럼

# In[30]:


# feature_names 라는 변수에 학습과 예측에 사용할 컬럼명 가져옴
feature_names=train.columns[:-1].tolist()
feature_names
# 맨 마지막것만 빼고 가져오겠다


# # 6정답값이자 예측해야 될 값

# In[31]:


# label_name 이라는 변수에 예측할 컬럼의 이름을 담는다

label_name = train.columns[-1]
label_name
# string 형태로 하나만 불러옴


# # 7학습, 예측 데이터셋 만들기

# In[32]:


# 학습 세트 만들기
# 예) 시험의 기출문제
X_train = train[feature_names]
print(X_train.shape)
X_train.head()


# In[34]:


# 정답 값을 만들어준다
# 예) 기출문제의 정답
y_train = train[label_name]
print(y_train.shape)
y_train.head()


# In[35]:


# 예측에 사용할 데이터세트를 만든다
# 예) 실전 시험 문제
X_test = test[feature_names]
print(X_test.shape)
X_test.head()


# In[36]:


# 예측의 정답값
# 예) 실전 시험 문제의 정답
y_test=test[label_name]
print(y_test.shape)
y_test.head()
# 실제로는 실제 시험 정답 모르지만 지금은 아니까


# # 8머신러닝 아로리즘 가져오기
# ### 이제 정답을 모르는 실제 시험 문제를 풀어보자

# In[37]:


# decision tree를 사용하자
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model


# # 9학습(훈련)

# In[ ]:


# 시험을 볼 때 기출문제X_trian 정담y_train을보고 공부하는 과정과 유사


# In[41]:


model.fit(X_train, y_train)


# # 10예측

# In[ ]:


# 실전 시험문제라고 보면되낟
# 정답을 직접 예측


# In[44]:


y_predict = model.predict(X_test)
y_predict[:5]


# # 11트리 알고리즘 분석하기

# In[ ]:


# 의사결정나무를 시각화


# In[48]:


from sklearn.tree import plot_tree
plt.figure(figsize=(20,20))
tree = plot_tree(model, feature_names=feature_names, filled=True, 
                 fontsize=10)


# In[49]:


# 피처의 중요도 추출

model.feature_importances_


# In[51]:


# 피처의 중요도 시각화

sns.barplot(x=model.feature_importances_, y=feature_names)


# # 12정확도 측정하기

# In[52]:


# 실제값 - 예측값을 빼주면 같은 값은 0
# 여기에서 절대값을 싀운 값이 1인 값이 다르게 예측한 값

abs(y_test - y_predict).sum() / len(y_test)


# In[56]:


diff_count = abs(y_test - y_predict).sum()
diff_count

# 매번 다르값이 나오기 때문에 갑싱 달라진다


# In[57]:


# 예측의 정확도를 구한다
# 100점 만점 중 몇점을 맞았는지 구한다고 보면 된다ㅏ

(len(y_test)-diff_count) / len(y_test)*100


# In[58]:


# 위에서 처럼 직접 구할 수 도 있지만 
# 미리 구현된 알고리즘을 가져와 사용

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)*100


# In[60]:


# model의 score로 점수를 걔산

model.score(X_test, y_test)*100

