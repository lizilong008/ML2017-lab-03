import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import time
import math
                           

class AdaBoostClassifier(object):
    '''A simple AdaBoost Classifier.'''
    def __init__(self,n_weakers_limit):
        '''Initialize AdaBoostClassifier
        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.classifier_list=[]
        for i in range(n_weakers_limit):
            temp_classifier=DecisionTreeClassifier(splitter="random",max_depth=1,random_state=i)
            self.classifier_list.append(temp_classifier)
        self.alpha=[0.0]*n_weakers_limit
        pass

    def is_good_enough(self):
        '''Optional'''
        pass
    def cal_reg_cof(self,D,alhpa,y,pre_y):
        total=0
        for i in range(len(D)):
            total+=D[i]*math.exp(-alhpa*y[i]*pre_y[i])
        return total
    def fit(self,x_train,y_train):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        
        D=[1.0/len(x_train)]*len(x_train)
        #i th loop
        for i in range(len(self.classifier_list)):         
            self.classifier_list[i].fit(x_train, y_train,D)
            error_rate=1-self.classifier_list[i].score(x_train, y_train)
            #print ("error_rate"+str(error_rate))
            self.alpha[i]=0.5*math.log((1-error_rate)/(error_rate))

            pre_y=self.classifier_list[i].predict(x_train)
            Z=self.cal_reg_cof(D,self.alpha[i],y_train,pre_y)
            for j in range(len(D)):
                D[j]=D[j]*math.exp(-self.alpha[i]*y_train[i]*pre_y[i])/Z
            
        pass


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        total=0
        for i in range(len(self.alpha)):
            total+=self.alpha[i]*self.classifier_list[i].predict(X)
        return total

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.
        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        temp=[X]
        total_score=self.predict_scores(temp)
        if(total_score>threshold):
            return 1
        else :
            return -1
    def predict_list(self, X_list, threshold=0):
        out_list=[]
        for i in range(len(X_list)):
            out_list.append(self.predict(X_list[i],threshold))
        return out_list

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
