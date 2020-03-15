import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn import preprocessing
import random
import multi_class
import xgbost
#import xlsxwriter

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
def zeroMean(dataMat):                  #去均值化
    meanVal = np.mean(dataMat, axis=0)  # 压缩行，对各列求均值，返回 1* n 矩阵
    newData = dataMat - meanVal
    return newData, meanVal
def noisyCount(sensitivity,epsilon):
    lamda=sensitivity/epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 < 0.5:
        n_value = lamda * np.log(2*u2)
    else:
        n_value = -lamda*np.log(2*(1-u2))
    return n_value
def laplace_mech(data,sensitivity,epsilon):
    for i in range(len(data)):
        data[i]=data[i]+noisyCount(sensitivity,epsilon)
    return data
def PCA_noisy(datamat):
    #隐私预算的分割为0.5/n
    epsilon=0.05
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

#正常算SA的过程
def subAlign(S, T, Xs, Xt):
    S = np.mat(S)
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

#接下来这部分是生成n个模型
#这个没加参数扰动
# def begging_by_tree(S,T,Ys,n,m):
#     predict_list=[]
#     k=np.zeros((m,1))       #存放合起来的Sa
#     l=np.zeros((m,1))       #存放合起来的Xt
#     for i in range(n):
#         train_data,test_data=rand_train(S,m,T)
#         Xs=PCA(train_data).real
#         Xt_noisy=PCA_noisy(test_data).real
#         Sa=subAlign1(Xs,Xt_noisy)
#         k=np.hstack((k,Sa))
#         l=np.hstack((l,Xt_noisy))
#     return k,l

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

# def Para_perturb(datamat):
#     #n取20的情况
#     epsilon=0.5/10
#     sensitivity=2/datamat.shape[0]
#     param_noisy=laplace_mech(datamat,sensitivity,epsilon)
#     return param_noisy

#参数扰动和子空间噪声都加上了
# def begging_by_tree2(S,T,Ys,n,m,Yt):
#     predict_list=[]
#     for i in range(n):
#         train_data,test_data=rand_train(S,m,T)
#         TAS,TPD=align(train_data,test_data)
#         clf=linear_model.LogisticRegression()
#         a=clf.fit(TAS,Ys)
#         coef=a.coef_
#         a.coef_=Para_perturb(coef)
#         y_predicted=a.predict(TPD)
#         acc7 = sklearn.metrics.accuracy_score(Yt, y_predicted)
#         #print(acc7)
#         predict_list.append(y_predicted)
#     return predict_list

#用multi_class
def begging_by_tree3(S,T,Ys,n,m,Yt):
    predict_list=[]
    for i in range(n):
        train_data,test_data=rand_train(S,m,T)
        TAS,TPD=align(train_data,test_data)
        #10是标签有多少类，1是正则化系数，0.05是epsilon，一共是10个模型，0.5/10
        all_theta=multi_class.one_vs_all(TAS,Ys,10,1,0.05)
        y_predict=multi_class.predict_all(TPD,all_theta)
        y_predict=np.array(y_predict.ravel())
        y_predict=y_predict.reshape(-1,)
        predict_list.append(y_predict)
    return predict_list

def begging_by_tree4(S,T,Ys,n,m,epsilon,lamda):
    epsilon2=0
    predict_list=[]
    train_list = []
    test_list = []
    q_list = []
    #算特征的重要程度
    for i in range(n):
        train_data, test_data, q = xgbost.rand_train(S, m, T,Ys)
        train_list.append(train_data)
        test_list.append(test_data)
        q_list.append(q)
    #算epsilon'
    for i in range(n):
        q_list[i] = q_list[i] / sum(q_list)
        epsilon_1 = np.log(1 + np.power(q_list[i], 2) / (2 * len(S) * lamda) + np.power(q_list[i], 4) / (
                    16 * np.power(len(S), 2) * np.power(lamda, 2)))
        epsilon2 = epsilon2 + epsilon_1
    epsilon_1 = epsilon - epsilon2
    for i in range(n):
        train_data,test_data=train_list[i],test_list[i]
        TAS,TPD=align(train_data,test_data)
        #10是标签有多少类，1是正则化系数，0.05是epsilon，一共是10个模型，0.5/10
        all_theta=multi_class.one_vs_all_q(TAS,Ys,10,lamda,epsilon_1,q_list[i])
        y_predict=multi_class.predict_all(TPD,all_theta)
        y_predict=np.array(y_predict.ravel())
        y_predict=y_predict.reshape(-1,)
        predict_list.append(y_predict)
    return predict_list
if __name__ == '__main__':
    src, tar = 'data/amazon_SURF_L10', 'data/webcam_SURF_L10'
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





    #后续随机取样
    #train_data,test_data=rand_train(S,700,T)#m应该是维数,S是源域数据集，T是目标域数据集
    #predict_list=begging_by_tree2(S,T,Ys,10,700,Yt)   #这里是n取10，m取700的情况,n是生成多少个模型，m是每个模型有多少个特征
    #print(len(predict_list[0]))
    #predict_label=calc_error(predict_list)
    #predict_label=tuple(predict_label)
    #acc7=sklearn.metrics.accuracy_score(Yt,predict_label[0])
    #print(acc7)

    # workbook = xlsxwriter.Workbook('epsilon1.xlsx')
    # worksheet = workbook.add_worksheet()
    # 引用multi_class5
    predict_list = begging_by_tree3(S, T, Ys, 10, 600, Yt)
    predict_label = calc_error(predict_list)
    predict_label = tuple(predict_label)
    acc8 = sklearn.metrics.accuracy_score(Yt, predict_label[0])
    print(acc8)

    predict_list = begging_by_tree4(S, T, Ys, 10, 600, 0.05,1)
    predict_label = calc_error(predict_list)
    predict_label = tuple(predict_label)
    acc9 = sklearn.metrics.accuracy_score(Yt, predict_label[0])
    print(acc9)

    # for i in range(0, 50):
    #     # 以下是添加噪声的部分
    #     predict_list = begging_by_tree3(S, T, Ys, 10, 600, Yt)
    #     predict_label = calc_error(predict_list)
    #     predict_label = tuple(predict_label)
    #     acc8 = sklearn.metrics.accuracy_score(Yt, predict_label[0])
    #
    #     worksheet.write(i, 1, acc8)
    #
    # workbook.close()









