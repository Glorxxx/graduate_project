import random

import numpy as np
import scipy.io
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import BaggingClassifier
import xlsxwriter

import Softmax
import multi_class

def zeroMean(dataMat):                  #去均值化
    meanVal = np.mean(dataMat, axis=0)  # 压缩行，对各列求均值，返回 1* n 矩阵
    newData = dataMat - meanVal
    return newData, meanVal


def PCA(dataMat):
    # 算协方差矩阵
    newData, meanVal = zeroMean(dataMat)
    covMat = np.cov(newData, rowvar=0)          #对去均值的数据求协方差矩阵
    # 求矩阵的特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 保留主要成分
    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(81):-1]  # 取特征值最大的80个特征向量作为
    n_eigVect = eigVects[:, n_eigValIndice]
    return n_eigVect

#以下是添加噪声需要的代码
def noisyCount(sensitivity,epsilon):
    beta=sensitivity/epsilon
    u1=np.random.random()
    u2=np.random.random()
    if u1<=0.5:
        n_value=-beta*np.log(1.-u2)
    else:
        n_value=beta*np.log(u2)
    return n_value
def laplace_mech(data,sensitivity,epsilon):
    for i in range(len(data)):
        data[i]=data[i]+noisyCount(sensitivity,epsilon)
    return data
def PCA_noisy(datamat):
    epsilon=5
    newData,meanVal=zeroMean(datamat)
    covMat=np.cov(newData,rowvar=0)
    #计算敏感度并加入噪声
    sensitivity=2*datamat.shape[1]/datamat.shape[0]
    covMat_noisy=laplace_mech(covMat,sensitivity,epsilon)
    eigVals,eigVects=np.linalg.eig(np.mat(covMat_noisy))
    #保留主成分
    eigValIndice=np.argsort(eigVals)
    n_eigValIndice=eigValIndice[-1:-(81):-1] # 最大的n个特征值的下标
    n_eigVect=eigVects[:,n_eigValIndice]
    return n_eigVect
def Para_perturb(datamat):
    epsilon=0.5
    sensitivity=2/datamat.shape[0]
    param_noisy=laplace_mech(datamat,sensitivity,epsilon)
    return param_noisy


#正常算SA的过程
def subAlign(S, T, Xs, Xt):
    S = np.mat(S)
    M = Xs.T * Xt
    M = np.array(M)
    Target_Aligned_Source_Data = S * (Xs * Xs.T * Xt)
    Target_Projected_Data = T * Xt
    return Target_Aligned_Source_Data, Target_Projected_Data
#自助法采样,这里的m就是每个子空间保留的特征的个数
#调用一次生成一组数据集
def rand_train(S,m,T):
    train_data = np.zeros((S.shape[0],m))
    test_data = np.zeros((T.shape[0], m))
    for i in range(m):
        index=random.randint(0,S.shape[1]-1) #在S和T的所有特征中随机挑选
        train_data[:,i]=S[:,index]
        test_data[:,i]=T[:,index]
    return train_data,test_data

#将源域和目标域对其
def align(S,T):
    Xs=PCA(S).real
    Xt_noisy=PCA_noisy(T).real
    TAS,TPD=subAlign(S,T,Xs,Xt_noisy)
    return TAS,TPD
#用multi_class
def begging_by_tree3(S,T,Ys,n,m,Yt):
    predict_list=[]
    for i in range(n):
        train_data,test_data=rand_train(S,m,T)
        TAS,TPD=align(train_data,test_data)
        #10是标签有多少类，1是正则化系数，0.05是epsilon，一共是10个模型，2.5/10
        all_theta=multi_class.one_vs_all(TAS,Ys,10,1,0.05)
        y_predict=multi_class.predict_all(TPD,all_theta)
        y_predict=np.array(y_predict.ravel())
        y_predict=y_predict.reshape(-1,)
        predict_list.append(y_predict)
    return predict_list

#汇总结果计算准确率
def calc_error(predict_list):
    predict_label=np.zeros((1,len(predict_list[0])))
    for i in range(len(predict_list[0])):
        dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,10:0}
        for j in range(len(predict_list)):
            c=predict_list[j][i]
            dict[c]=int(dict.get(c)+1)
        predict_label[0][i]=int(max(dict,key=dict.get))
    return predict_label

if __name__ == '__main__':
    workbook = xlsxwriter.Workbook('epsilon5.xlsx')
    worksheet = workbook.add_worksheet()
    #D to A
    src, tar = 'data/dslr_SURF_L10.mat', 'data/webcam_SURF_L10.mat'
    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    S, Ys = src_domain['fts'], src_domain['labels']
    T, Yt = tar_domain['fts'], tar_domain['labels']

    # 数据预处理
    S = np.mat(S)  # 读进来是数组形式，要将他转成矩阵形式
    S = S / np.tile(S.sum(axis=1), np.shape(S)[1])  # 将每一列的数除以每一列的和
    S = preprocessing.scale(S)  # 归一化

    T = np.mat(T)
    T = T / np.tile(T.sum(axis=1), np.shape(T)[1])
    T = preprocessing.scale(T)

    T = np.array(T)
    S = np.array(S)

    # 得到特征子空间
    Xs = PCA(S).real
    Xt = PCA(T).real

    # Target_Aligned_Source_Data, Target_Projected_Data = subAlign(S, T, Xs, Xt)
    #
    #
    #
    # #以上是不加噪声的部分
    # for i in range(0,50):
    #     # 以下是添加噪声的部分
    #     Xt_noisy = PCA_noisy(T).real
    #     sa, tt = subAlign(S, T, Xs, Xt_noisy)
    #
    #     # softmax部分,epsilon是0.5
    #     all_theta = multi_class.one_vs_all(sa, Ys, 10, 1, 0.5)
    #
    #     # 预测
    #     y_pred = multi_class.predict_all(tt, all_theta)
    #
    #     # 准确率
    #     acc7 = sklearn.metrics.accuracy_score(y_pred, Yt)
    #
    #     worksheet.write(i, 1, acc7)

    for i in range(0, 20):
        # 以下是添加噪声的部分
        predict_list = Softmax.begging_by_tree3(S, T, Ys, 10, 600, Yt)
        predict_label = calc_error(predict_list)
        predict_label = tuple(predict_label)
        acc8 = sklearn.metrics.accuracy_score(Yt, predict_label[0])
        worksheet.write(i, 1, acc8)




    #
    # #第二组
    # # D to C
    # src, tar = 'data/webcam_SURF_L10', 'data/Caltech10_SURF_L10'
    # src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    # S, Ys = src_domain['fts'], src_domain['labels']
    # T, Yt = tar_domain['fts'], tar_domain['labels']
    #
    # # 数据预处理
    # S = np.mat(S)  # 读进来是数组形式，要将他转成矩阵形式
    # S = S / np.tile(S.sum(axis=1), np.shape(S)[1])  # 将每一列的数除以每一列的和
    # S = preprocessing.scale(S)  # 归一化
    #
    # T = np.mat(T)
    # T = T / np.tile(T.sum(axis=1), np.shape(T)[1])
    # T = preprocessing.scale(T)
    #
    # T = np.array(T)
    # S = np.array(S)
    #
    # # 得到特征子空间
    # Xs = PCA(S).real
    # Xt = PCA(T).real
    #
    # for i in range(0, 50):
    #     # 以下是添加噪声的部分
    #     predict_list = begging_by_tree3(S, T, Ys, 10, 600, Yt)
    #     predict_label = calc_error(predict_list)
    #     predict_label = tuple(predict_label)
    #     acc8 = sklearn.metrics.accuracy_score(Yt, predict_label[0])
    #     worksheet.write(i, 2, acc8)


    # # 第三组
    # # W to D
    # src, tar ='data/webcam_SURF_L10','data/dslr_SURF_L10'
    # src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    # S, Ys = src_domain['fts'], src_domain['labels']
    # T, Yt = tar_domain['fts'], tar_domain['labels']
    #
    # # 数据预处理
    # S = np.mat(S)  # 读进来是数组形式，要将他转成矩阵形式
    # S = S / np.tile(S.sum(axis=1), np.shape(S)[1])  # 将每一列的数除以每一列的和
    # S = preprocessing.scale(S)  # 归一化
    #
    # T = np.mat(T)
    # T = T / np.tile(T.sum(axis=1), np.shape(T)[1])
    # T = preprocessing.scale(T)
    #
    # T = np.array(T)
    # S = np.array(S)
    #
    # # 得到特征子空间
    # Xs = PCA(S).real
    # Xt = PCA(T).real
    #
    # for i in range(0, 50):
    #     # 以下是添加噪声的部分
    #     predict_list = begging_by_tree3(S, T, Ys, 10, 600, Yt)
    #     predict_label = calc_error(predict_list)
    #     predict_label = tuple(predict_label)
    #     acc8 = sklearn.metrics.accuracy_score(Yt, predict_label[0])
    #     worksheet.write(i, 3, acc8)


    workbook.close()