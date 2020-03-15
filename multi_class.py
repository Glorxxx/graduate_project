import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from scipy.optimize import minimize
import scipy
from sklearn import preprocessing
import sklearn
from sklearn.metrics import classification_report

def noisy_function(beta):
    lamda = beta
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 < 0.5:
        n_value = lamda * np.log(2 * u2)
    else:
        n_value = -lamda * np.log(2 * (1 - u2))
    return n_value

def noisy_count(theta,beta):
    b=np.zeros((theta.shape[0],1))
    for i in range(theta.shape[0]):
        b[i]=noisy_function(beta)
    return b

def sigmoid(z):
    return 1/(1+np.exp(-z))

#损失函数
#lamda是正则化系数
def cost(theta,X,y,lamda,b,delta):
    theta=np.mat(theta)
    X=np.mat(X)
    y=np.mat(y)
    first=np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second=np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    reg=(lamda/(2*len(X)))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
    noise=np.sum((b.T*theta.T)/len(X))+delta*np.sum(np.power(theta[:,1:theta.shape[1]],2))/2

    return np.sum(first-second)/len(X)+reg+noise

#梯度
def gradient(theta,X,y,lamda,b,delta):
    theta=np.mat(theta)
    X=np.mat(X)
    y=np.mat(y)

    parameters=int(theta.ravel().shape[1])
    error=sigmoid(X*theta.T)-y

    grad=((X.T*error)/len(X)).T+((lamda/len(X))*theta)+b.T/len(X)+delta

    return np.array(grad).ravel()

#训练一对多分类器
def one_vs_all(X,y,num_labels,lamda,epsilon):
    rows=X.shape[0]
    params=X.shape[1]

    # k*(n+1) array for the parameters of each of the k classifiers
    all_theta=np.zeros((num_labels,params+1))

    # insert a column of ones at the begining for the intercept term
    X=np.insert(X,0,values=np.ones(rows),axis=1)

    # 需要计算b和delta
    epsilon_1 = epsilon - np.log(1 + 1 / (2 * len(X) * lamda) + 1 / (16 * np.power(len(X), 2) * np.power(lamda, 2)))
    if epsilon_1 > 0:
        delta = 0
    else:
        delta = 1 / (4 * len(X) * (np.exp(epsilon / 4) - 1)) - lamda
        epsilon_1 = epsilon / 2
    beta =  2/epsilon_1

    #labels are 1-indexed instead of 0-indexed
    for i in range(1,num_labels+1):
        theta=np.zeros(params+1)
        #将每个对应类标记为1其他类标记为0
        y_i=np.array([1 if label==i else 0 for label in y])
        y_i=np.reshape(y_i,(rows,1))
        b = noisy_count(theta, beta)
        #minimize the objective function
        fmin=minimize(fun=cost,x0=theta,args=(X,y_i,lamda,b,delta),method='TNC',jac=gradient)
        all_theta[i-1,:]=fmin.x
    return all_theta

def one_vs_all_q(X,y,num_labels,lamda,epsilon,q):
    rows=X.shape[0]
    params=X.shape[1]

    # k*(n+1) array for the parameters of each of the k classifiers
    all_theta=np.zeros((num_labels,params+1))

    # insert a column of ones at the begining for the intercept term
    X=np.insert(X,0,values=np.ones(rows),axis=1)

    if epsilon > 0:
        delta = 0
        epsilon_k = epsilon
    else:
        delta = np.power(q, 2) / (4 * len(S) * (np.exp(epsilon * q / 4) - 1)) - lamda
        epsilon_k = epsilon / 2
    beta = 2 * q / (epsilon_k)
    #labels are 1-indexed instead of 0-indexed
    for i in range(1,num_labels+1):
        theta=np.zeros(params+1)
        #将每个对应类标记为1其他类标记为0
        y_i=np.array([1 if label==i else 0 for label in y])
        y_i=np.reshape(y_i,(rows,1))
        b = noisy_count(theta, beta)
        #minimize the objective function
        fmin=minimize(fun=cost,x0=theta,args=(X,y_i,lamda,b,delta),method='TNC',jac=gradient)
        all_theta[i-1,:]=fmin.x
    return all_theta
#一对多预测
def predict_all(X,all_theta):
    rows=X.shape[0]
    params=X.shape[1]
    num_labels=all_theta.shape[0]

    # same as before, insert ones to match the shape
    X=np.insert(X,0,values=np.ones(rows),axis=1)

    # convert to matrices
    X=np.mat(X)
    all_theta=np.mat(all_theta)

    # compute the class probability for each class on each training instance
    h=sigmoid(X*all_theta.T)

    # create array of the index with the maximum probability
    h_argmax=np.argmax(h,axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax=h_argmax+1

    return h_argmax

#主函数部分
if __name__ == '__main__':
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

    #训练一对多模型
    #一共有10类，正则化系数是1
    #epsilon也是1
    all_theta=one_vs_all(S,Ys,10,1,0.05)

    #一对多预测
    y_pred=predict_all(T,all_theta)
    acc7 = sklearn.metrics.accuracy_score(y_pred,Yt)

    #输出预测结果
    print(acc7)




