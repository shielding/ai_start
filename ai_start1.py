#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
from sklearn.datasets import load_iris # Iris花的一个数据集
from sklearn import tree
iris = load_iris()
# print(iris.feature_names) 
## ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# print(iris.target_names)
## ['setosa' 'versicolor' 'virginica']
# print(iris.data[0])
## [5.1 3.5 1.4 0.2]
# print(iris.target[0])
## 0
test_id = [0,50,100] # 选择3个data和target用作测试

train_data = np.delete(iris.data, test_id, axis=0)
train_target = np.delete(iris.target, test_id)

test_data = iris.data[test_id]
test_target = iris.target[test_id]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))