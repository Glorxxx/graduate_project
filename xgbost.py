# plot feature importance using built-in function
from numpy import loadtxt
from sklearn import preprocessing
import sklearn.metrics
import numpy as np
import random
import scipy.io
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from scipy.optimize import minimize
import Softmax
import multi_class
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
#调用一次生成一组数据集并计算
def rand_train(S,m,T,Ys):
    # fit model no training data
    model = XGBClassifier()
    model.fit(S, Ys)
    # plot feature importance
    importance = model.feature_importances_
    q=0
    train_data = np.zeros((S.shape[0],m))
    test_data = np.zeros((T.shape[0], m))
    for i in range(m):
        index=random.randint(0,S.shape[1]-1) #在S和T的所有特征中随机挑选
        train_data[:,i]=S[:,index]
        test_data[:,i]=T[:,index]
        q=q+importance[index]
    return train_data,test_data,q
def sigmoid(z):
    return 1/(1+np.exp(-z))
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
    # theta0 不需要regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()
def one_vs_all(X,y,num_labels,lamda,delta,beta):
    rows = X.shape[0]
    params = X.shape[1]

    # k*(n+1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the begining for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # labels are 1-indexed instead of 0-indexed
    for i in range(1,num_labels+1):
        theta = np.zeros(params + 1)
        # 将每个对应类标记为1其他类标记为0
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        b = noisy_count(theta, beta)
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, lamda, b, delta), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x
    return all_theta
# # load data
# src, tar = 'data/dslr_SURF_L10.mat', 'data/webcam_SURF_L10.mat'
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
# T / np.tile(T.sum(axis=1), np.shape(T)[1])
# T = preprocessing.scale(T)
#
# T = np.array(T)
# S = np.array(S)
#
# # fit model no training data
# model = XGBClassifier()
# model.fit(S, Ys)
# # plot feature importance
# importance=model.feature_importances_
#
#
# # plot
# # pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
# # pyplot.show()
#
# train_list=[]
# test_list=[]
# q_list=[]
# n=10
# m=600
# epsilon=0.5
# epsilon2=0
# lamda=1
#
#
#
#
# # 计算每个数据集的特征重要程度
# for i in range(n):
#     train_data, test_data, q = rand_train(S, m, T)
#     train_list.append(train_data)
#     test_list.append(test_data)
#     q_list.append(q)
# # 将重要程度归一化
# for i in range(n):
#     q_list[i] = q_list[i] / sum(q_list)
#     epsilon_1 = np.log(1 + np.power(q_list[i], 2) / (2 * len(S) * lamda) + np.power(q_list[i], 4) / (16 * np.power(len(S), 2) * np.power(lamda, 2)))
#     epsilon2 = epsilon2 + epsilon_1
# epsilon_1 = epsilon - epsilon2
#
#
#
#
# #训练模型
# predict_list = []
# for i in range(n):
#     # 使所有traindata中的值小于qk
#     train_data= q_list[i] * train_list[i] / np.amax(train_list[i])
#
#     test_data=test_list[i]
#     TAS,TPD=Softmax.align(train_data,test_data)
#     # 计算b和delta
#     if epsilon_1 > 0:
#         delta = 0
#         epsilon_k = epsilon_1
#     else:
#         delta = np.power(q_list[i], 2) / (4 * len(S) * (np.exp(epsilon * q_list[i] / 4) - 1)) - lamda
#         epsilon_k = epsilon / 2
#     beta = 2*q_list[i]/(epsilon_k)
#     # 10是标签有多少类，1是正则化系数，0.05是epsilon，一共是10个模型，0.5/10
#     all_theta=one_vs_all(TAS,Ys,10,1,delta,beta)
#     y_predict = multi_class.predict_all(TPD, all_theta)
#     y_predict = np.array(y_predict.ravel())
#     y_predict = y_predict.reshape(-1, )
#     predict_list.append(y_predict)
#
# predict_label=Softmax.calc_error(predict_list)
# predict_label = tuple(predict_label)
# acc8 = sklearn.metrics.accuracy_score(Yt, predict_label[0])
# print(acc8)




