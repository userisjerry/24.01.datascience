#!/usr/bin/env python
# coding: utf-8

# # 1. Exploratoty data analysis

# # 2. 데이터셋

# In[1]:


# 3. 라이브러리 로드
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# # 4. 데이터 로드
# 

# In[2]:


df = pd.read_csv('data/diabetes.csv')
df.shape


# In[3]:


# 위에서 5개 미리보기
df.head()


# In[4]:


# info 로 데이터 타입, 결측치, 메모리 사용량  정보
df.info()


# In[5]:


# 결측치

df_null=df.isnull()
df_null.head()


# In[6]:


df_null.sum()


# In[7]:


# 수치데이터 요약

df.describe() 
 #df.describe(include='number') 하면 오류가 난다. 오브젝트가 없어서
    


# In[8]:


# 가장 마지막에 있는 outcome은 label 값이기 때문에 제외
# 학습과 예측에 사용할 컬럼을 만들어 준다
# feature_colimns 라는 변수에 담아준다


feature_columns = df.columns[:-1].tolist()
feature_columns


# # 5. 결측치 시각화
# * 값을 요약해 보면 최솟값이 0으로 나오는 값들이 있다
# * 0이 나올 수 있는 값도 있지만 인슐린이나 혈압 등의 값은 0값이 결측치라고 볼 수 있다
# * 따라서 0인 값을 결측치로 처리하고 시각화

# In[9]:


cols = feature_columns[1:]
cols


# In[10]:


# 결측치 여부를 나타내는 데이터프레임 만든다
# 0 값을 결측치라 가정하고 정답(label, target)값을 제외한 컬럼에 대해
# 결측치 여부를 구해서 df_null 이라는 데이터 프레임에 담는다.

df_null = df[cols].replace(0, np.nan)
df_null = df_null.isnull()
df_null.sum()


# In[11]:


df_null.mean()*100  # *100 -> 퍼센트


# In[79]:


# 결측치의 갯수를 구해 막대 그래프로 시각화

df_null.sum().plot.bar()


# In[13]:


# 결측치를 heatmap 시각화
# 넓게 그려보자
plt.figure(figsize=(15,4))
sns.heatmap(df_null , cmap='Greys_r')
# 결측치만 흰색으로->Greys_r


# # 6. 정답값
# * target, label이라고 부르기도 한다.

# In[14]:


# 정답값인 outcome 의 갯수를 본다

df['Outcome'].value_counts()


# In[15]:


# 정답값인 outcome의 비율을 본다

df['Outcome'].value_counts(normalize=True)


# In[16]:


# 다른 변수와 함께 본다
# 임신 횟수와 정답값을 비교해 본다
# Pregnancies를 groupby 로 그룹화 해서 outcome에 대한 비율을 구한다
# 결과를 df_po라는 변수에 저장

# df.groupby(['Pregnancies'])['Outcome'].mean()

# 빈도수와 같이 보자
# df.groupby(['Pregnancies'])['Outcome'].agg(['mean','count'])

# column 값으로 배열하자
df_po = df.groupby(['Pregnancies'])['Outcome'].agg(['mean','count']).reset_index()
df_po


# In[17]:


# 임신횟수에 따른 당뇨병 발병 비율

# df_po.plot()
# 비율만 볼래
df_po['mean'].plot.bar(rot=0)
# rot=0 글자세우는거


# # 7. countplot

# In[18]:


# 위에서 구했던 당뇨병 발병 비율을 구한다
# 당뇨병 발병 빈도수 비교

sns.countplot(data=df, x='Outcome')


# In[19]:


# 임신횟수에 따른 당뇨병 발병 빈도수 비교
sns.countplot(data=df, x='Pregnancies', hue='Outcome')


# In[20]:


# 임신횟수의 많고 적음에 따라 Pregnancies_high 변수를 만든다

df['Pregnancies_high'] = df['Pregnancies']>6
df[['Pregnancies','Pregnancies_high']].head()


# In[21]:


# Pregnancies_high 변수의 빈도수ㅡㄹ countplor 으로 그리고
# Outcome 값에 따라 다른 색상으로 표현

sns.countplot(data=df, x='Pregnancies_high', hue='Outcome')


# # 8. barplot
# * 기본 설정으로 시각화하면 y축에는 평균을 추정해서 그리게 됩니다

# In[22]:


# 당뇨병 발병에 따른 bmi 수치를 비교
sns.barplot(data=df, x='Outcome', y='BMI')


# In[23]:


# 당노병 발병에 따른 Glucose 수치를 비교
sns.barplot(data=df, x='Outcome', y='Glucose')


# In[24]:


# Insulin 수치가 0 이상인 관측치에 대해서 당뇨병 발병을 비교
sns.barplot(data=df, x='Outcome', y='Insulin')


# In[25]:


# 임신 횟수에 대해 당뇨병 발병 비율을 비교
sns.barplot(data=df, x='Pregnancies', y='Outcome')


# In[26]:


# Pregnancies에 따른 Glucose 수치를 당뇨병 Outcome에 따라 시각화

sns.barplot(data=df, x='Pregnancies', y='Glucose', hue='Outcome')


# In[27]:


# Pregnancies에 따른 BMI 수치를 당뇨병 Outcome에 따라 시각화
sns.barplot(data=df, x='Pregnancies', y='BMI', hue='Outcome')


# # 9. boxplot
# 

# In[29]:


# Pregnancies에 따른 Insulin를 Outcome에 따라 시각화
# 인슐린 수치에는 결측치가 많기 때문에 0보다 큰 ㅏㄱㅂ셍 대해서만 그린다

sns.boxplot(data=df[df['Insulin']>0], x='Pregnancies', y='Insulin', hue='Outcome')


# In[30]:


sns.barplot(data=df[df['Insulin']>0], x='Pregnancies', y='Insulin', hue='Outcome')


# In[31]:


# 10. violinplot


# In[32]:


plt.figure(figsize=(15,4))
sns.violinplot(data=df[df['Insulin']>0], x='Pregnancies', y='Insulin', hue='Outcome', split=True)


# # 11. swarmplot

# In[34]:


plt.figure(figsize=(15,4))
sns.swarmplot(data=df[df['Insulin']>0], x='Pregnancies', y='Insulin', hue='Outcome')


# # 12. distplot

# In[35]:


df_0 = df[df['Outcome']==0]
df_1 = df[df['Outcome']==1]
df_0.shape, df_1.shape


# In[36]:


# 임신횟수에 따른 당뇨병 발병 여부 시각화

sns.distplot(df_0['Pregnancies'])
sns.distplot(df_1['Pregnancies'])
# a -> pandas series
# rug -> 표시랄거냐 말거냐


# In[37]:


# 나이에 따른 당뇨병 발병 여부 시각화

sns.distplot(df_0['Age'], hist = False, rug=True, label=0)
sns.distplot(df_1['Age'], hist = False, rug=True, label=1)


# # 13. sublot
# 

# # 13.1 pandas 통한 histplot그리기
# * pandas를 사용하면 모든 변수에 대한 서브플롯을 한번에 그려준다

# In[38]:


df['Pregnancies_high'] = df['Pregnancies_high'].astype(int)
df
# pregnancies -> 0,1
df.hist(figsize=(15,15), bins=20)


# In[39]:


# 13.2 반복문을 통한 서브플롯 그릭ㅣ


# In[40]:


# 13.2.1 distplot


# In[41]:


# 컬럼의 수 만큼 for문을 만들어서 서브 플롯으로 시각화 

#col_num = df.columns.shape
#col_num

cols = df.columns[:-1].tolist()
cols 
 


# In[42]:


# distplot 으로 서브플롯

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))
# 빈 ㄴsubplot
for i, col_name in enumerate(cols) :
    row = i//3
    col = i % 3
    sns.distplot(df[col_name], ax=axes[row][col])
    
    
    
# sns.distplot(df['Outcome'], ax=axes[1][1])


# In[43]:


df[df['Outcome']==0]


# In[44]:


cols


# In[45]:


# 모든 변수에 대한 distplot 을 그려본다

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,15))
# 빈 ㄴsubplot
for i, col_name in enumerate(cols[:-1]) :
    row = i//2
    col = i % 2
    sns.distplot(df_0[col_name], ax=axes[row][col])
    sns.distplot(df_1[col_name], ax=axes[row][col])
    


# In[46]:


# 13.2.2 violinplot


# In[47]:


# violinplot 으로 서브플롯을 그려본다

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15,15))
# 빈 ㄴsubplot
for i, col_name in enumerate(cols[:-1]) :
    row = i//2
    col = i % 2
    sns.violinplot(data=df, x='Outcome', y=col_name, ax=axes[row][col])


# # 13.2.3 implot
# * 상관계수가 높은 두 변수에 대해 시각화
# 

# In[48]:


# Glucose와 Insulin 을 Outcome으로 구분
sns.lmplot(data=df, x='Glucose', y='Insulin', hue='Outcome')


# In[49]:


# Insulin 수치가 0 이상인 데이터로만 그려본다

sns.lmplot(data=df[df['Insulin']>0], x='Glucose', y='Insulin', hue='Outcome')


# # 13.2.4 pairplot

# In[50]:


# PairGrid를 통해 모든 변수에 대해 Outcome에  따른 scatterplot을 그려본다

g = sns.pairplot(df, hue='Outcome')
g.map(plt.scatter)


# # 14. 상관 분석

# In[56]:


df_matrix = df.iloc[:,:-2].replace(0,np.nan)
df_matrix['Otcome'] = df["Outcome"]
df_matrix


# In[67]:


# 정답 값인 Outcome을 제외하고 feature로 사용할 컬럼들에 대해 0을 결측치로 만들어 준다
# 상관계수를 구한다

df_corr = df_matrix.corr()
df_corr.style.background_gradient()


# In[69]:


# 위에서 수한 상관계수를 heatmap으로 시각화
plt.figure(figsize=(15,8))
sns.heatmap(df_corr, annot=True, vmax=1, vmin=-1, cmap='coolwarm')


# In[78]:


# Outcome 수치에 대한 상관계수만 모아서 본다

df_corr['Otcome']


# # 14.1 상관계수가 높은 변수끼리 보기

# In[72]:


sns.regplot(data=df, x='Insulin', y='Glucose')


# In[73]:


# Insulin 과 Glucose 로 replot그리기

sns.regplot(data=df, x='Insulin', y='Glucose')


# In[74]:


# df_0 으로 결측치 처리한 데이터프레임으로
# Insulin 과 Glucose로 regplot 그리기
sns.regplot(data=df_matrix, x='Insulin', y='Glucose')


# In[75]:


# Age, Pregnancies - regplot
sns.regplot(data=df, x='Age', y='Pregnancies')


# In[77]:


# Age & Pregnancies -> lmplot
# Outcome에 따라 다른 색상

sns.lmplot(data=df, x='Age', y='Pregnancies', hue='Outcome', col='Outcome')

