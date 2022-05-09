import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 머신러닝 정의 : 데이터에서 패턴을 학습하여, 정보를 얻는 것

# 머신러닝 종류 : 지도학습(분류, 회귀)
#                비지도학습(군집, 차원축소)
#                강화학습

# 머신러닝 순서 : 문제정의 -> 데이터수집 -> 전처리 -> EDA -> 모델선택 -> 학습 -> 평가


# 문제정의 : 타이타닉 정보를 통해 죽은 사람과 산 사람을 예측해보자.


# 데이터 수집
train = pd.read_csv('./1강 예제1 Titanic 데이터/train.csv', index_col='PassengerId')
test = pd.read_csv('./1강 예제1 Titanic 데이터/test.csv', index_col='PassengerId')


# 전처리 (특성 ->Survived, Pclass, Name, Sex, Age, Sibsp, Parch, Ticket, Fare, Cabin, Embarked)
#        Name, Sibsp, Parch, Ticket, Fare, Cabin 은 필요없는 특성이라 판단하고 지운다.


# print("train : ")
# print(train.isnull().sum())
# print()
# print("test : ")
# print(test.isnull().sum())

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
