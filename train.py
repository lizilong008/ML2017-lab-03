from skimage import io,transform,color,img_as_ubyte
from sklearn import datasets as ds,model_selection as ms
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from feature import NPDFeature
from ensemble import AdaBoostClassifier
import numpy as np
import pickle
import time
def extract_fea(img_dirs,img_labels,store_name):
    '''
    预处理阶段，处理为24*24，灰度图
    '''
    fea_list=[]
    for i in range(len(img_dirs)):
        temp_img=io.imread(img_dirs[i])
        temp_gray_img=color.rgb2gray(temp_img)
        temp_resized_img=transform.resize(temp_gray_img, (24, 24))
        temp_resized_img=img_as_ubyte(temp_resized_img)
        #提取特征
        npd_fea=NPDFeature(temp_resized_img)
        temp_fea=npd_fea.extract()
        temp_label=img_labels[i]
        fea_list.append((temp_fea,temp_label))
    o_file = open(store_name, 'wb')
    pickle.dump(fea_list, o_file,-1)
    o_file.close()
if __name__=="__main__": 
    #提取特征 
    T_IMG_DIR="datasets/original/face/"
    T_imgs=os.listdir(T_IMG_DIR)
    for i in range(len(T_imgs)):
        T_imgs[i]=T_IMG_DIR+T_imgs[i]
    T_labels=[1]*len(T_imgs)

    F_IMG_DIR="datasets/original/nonface/"
    F_imgs=os.listdir(F_IMG_DIR)
    for i in range(len(F_imgs)):
        F_imgs[i]=F_IMG_DIR+F_imgs[i]
    F_labels=[-1]*len(F_imgs)


    X=T_imgs+F_imgs
    y=T_labels+F_labels
    X_train,X_val,y_train,y_val=ms.train_test_split(X,y,test_size=0.33)

    extract_fea(X_train,y_train,"train.fea")
    extract_fea(X_val,y_val,"val.fea")
    print ("extract sucess")
    
    



    #从文件中读取特征
    train_file = open('train.fea', 'rb')
    train_list = pickle.load(train_file)
    train_file.close()
    X_train=[]
    y_train=[]
    for i in range(len(train_list)):
        X_train.append(train_list[i][0])
        y_train.append(train_list[i][1])

    val_file = open('val.fea', 'rb')
    val_list = pickle.load(val_file)
    val_file.close()
    X_val=[]
    y_val=[]
    for i in range(len(val_list)):
        X_val.append(val_list[i][0])
        y_val.append(val_list[i][1])
    print ("load fea success")

    
    
    # #训练
    # print ("train")
    # temp_cls=AdaBoostClassifier(100)
    # begin_time =time.time()#
    # temp_cls.fit(X_train,y_train)
    # end_time =time.time()#
    # total_time=(end_time-begin_time)
    # AdaBoostClassifier.save(temp_cls,"fd.model")
    # print ("total train time"+str(total_time))
    # print ("save trained model success")


    #加载训练好的模型
    temp_cls=AdaBoostClassifier.load("fd.model")
    print (len(temp_cls.classifier_list))
    print (len(temp_cls.alpha))
    #测试
    acc=0.0
    predict_y=temp_cls.predict_list(X_val)
    report=classification_report(y_val, predict_y, labels=None, target_names=None, sample_weight=None, digits=2)
    report_file = open('report.txt', 'w')
    report_list = report_file.write(report)
    report_file.close()
    print ("report:"+str(report))

    for i in range(len(y_val)):
        if(predict_y[i]==y_val[i]):
            acc+=1
    acc/=len(y_val)
    print ("acc:"+str(acc))
    



    









