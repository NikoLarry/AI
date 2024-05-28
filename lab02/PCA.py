import numpy as np
import random
from math import pi
import matplotlib.pyplot as plt


def te_us_from_data(test_num_list, data_arr):  # 生成测试样本列表，训练样本列表
    test_arr = np.zeros((36, 14))
    used_arr = np.zeros((142, 14))
    te_num = 0  # test_arr里面已赋值数量
    us_num = 0  # used_arr里面已赋值数量
    for te_us_num in range(178):
        if te_num != 36 and (te_us_num == test_num_list[te_num]):
            test_arr[te_num] = data_arr[te_us_num]
            te_num += 1
        else:
            used_arr[us_num] = data_arr[te_us_num]
            us_num += 1
    return test_arr, used_arr



def main():
    # 读取数据
    f = open('wine.data', 'r')
    data_big_list = f.readlines()  # 每个元素是以\n结尾的一行字符串
    height = len(data_big_list)  # 元素个数
    width = 14  # 每个元素长度
    data_arr = np.zeros((height, width))  # 存放所有样本类别及13个属性数据，每行存一个样本
    for create_i in range(height):
        data_arr[create_i] = (data_big_list[create_i].rstrip('\n')).split(',')

    random.seed(66)
    test_num_list = random.sample(range(178), 36)  # 测试样本序号列表
    test_num_list.sort()  # 列表元素从小到大排序
    test_arr, used_arr = te_us_from_data(test_num_list, data_arr)  # 生成测试样本列表，训练样本列表
                                                                   # 形状为 样本个数*14
    # 以下三个是同学们可以用到的三组数据：训练集的特征，训练集的标签，测试集的特征
    train_features = used_arr[:,1:]     # 训练集共142个数据
    train_labels = used_arr[:,0]
    test_features = test_arr[:,1:]      # 测试集共36个数据

    ############## 同学完成 ###############
    # task：同学最终需要得到的结果是根据特征test_features，得到test集的预测红酒种类test_result_arr（36*1的向量，每个元素代表test集中每个数据的种类
    result = []
    d = 6
    train_size = train_features.shape
    train_mean = train_features.mean(axis=0)
    train_features = train_features - train_mean          # 对个样本属性数据中心化
                                                    # 降到指定维数
    X = np.cov(np.array(train_features.T))                       # 计算特征值与特征向量
    w, v = np.linalg.eig(X)
    idx = w.argsort()[::-1]         # 按降序排列
    eigenvectors = v[:, idx]
    dimension_array = eigenvectors[:, :d]

    Y =np.dot(train_features, dimension_array)                                # 投影

    # 贝叶斯分类


    N1,N2,N3 = 0,0,0                    # 类别先验概率
    wine1 = []
    wine2 = []
    wine3 = []
    for i in range(train_size[0]):
        if(train_labels[i] == 1):
            N1 = N1 + 1
            wine1.append(Y[i,:])
        if(train_labels[i] == 2):
            N2 = N2 +1
            wine2.append(Y[i, :])
        if(train_labels[i] == 3):
            N3 = N3 +1
            wine3.append(Y[i, :])

    # 先验概率
    P1 = N1/train_size[0]
    P2 = N2/train_size[0]
    P3 = N3/train_size[0]


    wine1 = np.array(wine1)
    wine2 = np.array(wine2)
    wine3 = np.array(wine3)
    avg = [[[] for m in range(d)] for n in range(3)]
    sigma = [[[] for m in range(d)] for n in range(3)]
    for i in range(d):                                             # 计算第i类在第c个属性上的均值方差
        avg[0][i].append(np.mean(wine1[:, i],axis=0))             # 降维后第 c 个属性样本均值
        sigma[0][i].append(np.var(wine1[:, i]))                        # 降维后第C个属性样本方差
        avg[1][i].append(np.mean(wine2[:, i],axis=0))
        sigma[1][i].append(np.var(wine2[:, i]))
        avg[2][i].append(np.mean(wine3[:, i],axis=0))
        sigma[2][i].append(np.var(wine3[:, i]))

    Test = test_features - train_mean
    y = np.dot(Test, dimension_array)
    test_result_arr = []
    # print(y)
    for j in range(len(y[:, 0])):
        p1, p2, p3 = 1, 1, 1
        for i in range(d):
            p1 = p1 * 1 / (np.sqrt(2 * pi * sigma[0][i][0])) * np.exp(-(y[j, i] - avg[0][i][0]) ** 2 / (2 * sigma[0][i][0]))
            p2 = p2 * 1 / (np.sqrt(2 * pi * sigma[1][i][0])) * np.exp(-(y[j, i] - avg[1][i][0]) ** 2 / (2 * sigma[1][i][0]))
            p3 = p3 * 1 / (np.sqrt(2 * pi * sigma[2][i][0])) * np.exp(-(y[j, i] - avg[2][i][0]) ** 2 / (2 * sigma[2][i][0]))
        A = [p1*P1,p2*P2,p3*P3]
        test_result_arr.append(A.index(max(A))+1)
    test_result_arr = np.array(test_result_arr)




    ###### 以上由同学完成 #############

    test_result_arr=test_result_arr.squeeze()
    contract_arr = test_result_arr - test_arr[:, 0]
    goodones = len(np.argwhere(contract_arr == 0)) # 这个正确率可以作为评价指标
    print(goodones)
    result.append(goodones)
    # d1 = [i for i in range(13)]
    #
    # plt.plot(d1,result)
    # plt.title("判断正确数随维度变化")
    # plt.xlabel("PCA维度")
    # plt.ylabel("正确数量/个")
    # plt.show()


if __name__ =='__main__':
    main()