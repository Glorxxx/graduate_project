import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

# def laplace_function(x, epsilon):
#     beta=epsilon/2
#     result = (1 / (2 * beta)) * np.exp(-1 * (np.abs(x) / beta))
#     return result
#
#
# # 在-5到5之间等间隔的取10000个数
# x = np.linspace(-5, 5, 10000)
# y1 = [laplace_function(x_, 1) for x_ in x]
# y2 = [laplace_function(x_, 5) for x_ in x]
# y3 = [laplace_function(x_, 10) for x_ in x]
#
# plt.plot(x, y1, color='r', label='epsilon:1')
# plt.plot(x, y2, color='g', label='epsilon:5')
# plt.plot(x, y3, color='b', label='epsilon:10')
# plt.title("Laplace distribution")
# plt.legend()
# plt.show()
def noisyCount(sensitivity,epsilon):
    beta=sensitivity/epsilon
    u1=np.random.random()
    u2=np.random.random()
    if u1<=0.5:
        n_value=-beta*np.log(1.-u2)
    else:
        n_value=beta*np.log(u2)
    return n_value
workbook = xlsxwriter.Workbook('laplace.xlsx')
worksheet = workbook.add_worksheet()


a=noisyCount(100,10)
# sensitivity=2
# epsilon=1
# for i in range(20):
#     s = noisyCount(sensitivity,epsilon)
#     worksheet.write(i, 1, s)
#
# epsilon=10.
#
# for i in range(20):
#     s = noisyCount(sensitivity, epsilon)
#     worksheet.write(i, 2, s)
#
# workbook.close()