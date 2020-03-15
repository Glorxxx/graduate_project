import numpy as np
import scipy.io
import xlsxwriter
from sklearn import preprocessing
from sklearn.decomposition import PCA


def noisyCount(sensitivity,epsilon):
    beta=sensitivity/epsilon
    u1=np.random.random()
    u2=np.random.random()
    if u1<=0.5:
        n_value=-beta*np.log(1.-u2)
    else:
        n_value=beta*np.log(u2)
    return n_value
def noisyCount1(sensitivity,epsilon):
    lamda=sensitivity/epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 < 0.5:
        n_value = lamda * np.log(2*u2)
    else:
        n_value = -lamda*np.log(2*(1-u2))
    return n_value


def laplace_mech(data, epsilon):
    for i in range(len(data)):
        data[i] += noisyCount(epsilon)
    return data


src, tar = 'data/PIE05.mat', 'data/PIE07.mat'
src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
S, Ys = src_domain['fea'], src_domain['gnd']
T, Yt = tar_domain['fea'], tar_domain['gnd']

# 数据预处理
S = np.mat(S)  # 读进来是数组形式，要将他转成矩阵形式
S = S / np.tile(S.sum(axis=1), np.shape(S)[1])  # 将每一列的数除以每一列的和
S = preprocessing.scale(S)  # 归一化

T = np.mat(T)
T = T / np.tile(T.sum(axis=1), np.shape(T)[1])
T = preprocessing.scale(T)

T = np.array(T)
S = np.array(S)

pca = PCA(n_components='mle')
pca.fit(S)
print(pca.n_components_)
