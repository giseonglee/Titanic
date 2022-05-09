import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 모델학습 부분
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

### 머신러닝 정의 : 데이터에서 패턴을 학습하여, 정보를 얻는 것

### 머신러닝 종류 : 지도학습(분류, 회귀)
#                비지도학습(군집, 차원축소)
#                강화학습

### 머신러닝 순서 : 문제정의 -> 데이터수집 -> 전처리 -> EDA -> 모델선택 -> 학습 -> 평가


### 문제정의 : 타이타닉 정보를 통해 죽은 사람과 산 사람을 예측해보자.


### 데이터 수집
train = pd.read_csv('./1강 예제1 Titanic 데이터/train.csv', index_col='PassengerId')
test = pd.read_csv('./1강 예제1 Titanic 데이터/test.csv', index_col='PassengerId')


### 전처리 (특성 ->Survived, Pclass, Name, Sex, Age, Sibsp, Parch, Ticket, Fare, Cabin, Embarked)
#        Name, Sibsp, Parch, Ticket, Fare, Cabin 은 필요없는 특성이라 판단하고 지운다.


train = train.drop(['Name'], axis=1)
test = test.drop(['Name'], axis=1)

train = train.drop(['SibSp'], axis=1)
test = test.drop(['SibSp'], axis=1)

train = train.drop(['Parch'], axis=1)
test = test.drop(['Parch'], axis=1)

train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)

train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)

print("train : ")
print(train.isnull().sum())
print()
print("test : ")
print(test.isnull().sum())

# 결측치인 Age, Embarked는 채우도록 한다.(2가지 방식으로 해봄.) 

train['Age'] = train.groupby(['Pclass','Sex'])['Age'].transform('median')
test['Age'] = test['Age'].fillna(test.groupby(['Pclass','Sex'])['Age'].transform('median'))

# Embarked는 도메인 널리지를 기반으로 채움.

train['Embarked'] = train['Embarked'].fillna('S')


print("train : ")
print(train.isnull().sum())
print()
print("test : ")
print(test.isnull().sum())

# 현재 train, test는 Survived 특성을 제외하고 똑같다. 


# train의 'Survived' 특성을 따로 빼기

feature = ['Pclass','Sex','Age','Embarked']

# 원핫인코딩 (get_dummies)
X_train = pd.get_dummies(train[feature])
y_train = train['Survived']
X_test = pd.get_dummies(test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

### 모델선택


model = RandomForestClassifier(random_state=0)

param_grid = {'max_depth': [5,6,7,8,9], # 트리의 노드 깊이
             "n_estimators": [100], # 트리의 개수
             "max_features":[0.2,0.4,0.6,0.8,0.1], # 하나의 노드에서 고려될 특성의 범위(0.2 = 전체 특성 중 20%만 선택)
             'min_samples_leaf': [6,7,8,9,10] # 리프 노드에서 최소 존재해야할 샘플의 개수 (과대적합을 보완해줌)
             }

gs = GridSearchCV(model, param_grid, cv=5)

gs.fit(X_train, y_train)

print("최상의 크로스 밸리데이션 점수: {:.6f}".format(gs.best_score_))
print("최적의 매개변수: ", gs.best_params_)

score = pd.DataFrame(gs.cv_results_['params'])
score["mean_test_score"]=gs.cv_results_['mean_test_score']
score.sort_values(by='mean_test_score', ascending=False)



