# -*- encoding:utf-8 -*-
from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression as LinearSVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy
class Classifier:
    def __init__(self,**kwargs):
        self.model = LinearSVC(**kwargs)
        self.paras = kwargs
    def train(self,features, labels):
        self.model.fit(features, labels)
    def predict(self,features):
        return self.model.predict(features)
    def score(self,features,labels):
        return self.model.score(features,labels)
    def micro_f1(self, features, labels):
        labels_p = self.predict(features)
        f1 = metrics.f1_score(labels, labels_p, average='micro')
        return f1
    def macro_f1(self, features, labels):
        labels_p = self.predict(features)
        f1 = metrics.f1_score(labels, labels_p, average='macro')
        return f1

class MultiLabelClassifier:
    def __init__(self, **kwargs):
        self.model = OneVsRestClassifier(LinearSVC(**kwargs),n_jobs=1)
        self.paras = kwargs
        self.mlb = MultiLabelBinarizer()
    def train(self, features, labels):
        label_bin = self.mlb.fit_transform(labels)
        self.model.fit(features, label_bin)
    def predict(self, features):
        return self.model.predict(features)
    def score(self, features, labels):
        label_bin = self.mlb.transform(labels)
        return self.model.score(features, label_bin)
    def micro_f1(self, features, labels):
        label_rea = self.mlb.transform(labels)
        label_pre = self.predict(features)
        n,m = label_pre.shape
        label_rea_all = label_rea.reshape((n*m,1))
        label_pre_all = label_pre.reshape((n*m,1))
        return metrics.f1_score(label_rea_all, label_pre_all, average='binary')

    def macro_f1(self, features, labels):
        label_rea = self.mlb.transform(labels)
        label_pre = self.predict(features)
        n,m = label_pre.shape
        f1 = []
        for i in range(m):
            f1.append(
                metrics.f1_score(label_rea[:,i], label_pre[:,i], average='binary')
            )
        return numpy.average(f1)
        
        
        
        

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.datasets import make_multilabel_classification
    # train_data = datasets.fetch_20newsgroups_vectorized(subset="train")
    # test_data = datasets.fetch_20newsgroups_vectorized(subset="test")
    # features = train_data.data
    # labels = train_data.target
    # model = Classifier()
    # model.train(features, labels)
    # print(model.model.score(train_data.data,train_data.target))    
    # print(model.model.score(test_data.data,test_data.target))

    n_samples = 100
    n_features = 10
    n_classes = 4
    x, y = make_multilabel_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, random_state=100)
    yy = []
    for i in range(y.shape[0]):
        d = set()
        for j in range(y.shape[1]):
            if y[i,j] == 1:
                d.add(j)
        yy.append(d)
    model = MultiLabelClassifier()
    model.train(x, yy)
    print(model.score(x, yy))
    print(model.micro_f1(x, yy))
    print(model.macro_f1(x, yy))
    


