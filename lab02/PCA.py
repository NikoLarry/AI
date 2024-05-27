import numpy as np
import random
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
    train_features = used_arr[:,1:]# 训练集共142个数据
    train_labels = used_arr[:,0]
    test_features = test_arr[:,1:]# 测试集共36个数据

    ############## 同学完成 ###############
    # task：同学最终需要得到的结果是根据特征test_features，得到test集的预测红酒种类test_result_arr（36*1的向量，每个元素代表test集中每个数据的种类
    train_size = train_features.shape
    sum_feature = np.zeros(train_size[1])
    for i in range(train_size[0]):
        sum_feature = sum_feature + train_features[i, :]
    mean_feature = sum_feature/train_labels[0]
    for i in range(train_size[0]):
        train_features[i, :] = train_features[i, :] - mean_feature        # 对个样本属性数据中心化

    # 降到指定维数
    d = 2
    X = np.mat(np.cov(np.array(train_size.T)))
    w, v = np.linalg.eig(X)
    dimension_array = np.mat(v[:d, :].T)
    Y = X * dimension_array         # 投影


    # 贝叶斯分类
    P1,P2,P3 = 0                    # 类别先验概率
    for i in range(train_size[0]):
        if(train_labels[i] == 1):
            P1 = P1 + 1
        if(train_labels[i] == 2):
            P2 = P2 +1
        if(train_labels[i] == 3):
            P3 = P3 +1
    P1 = P1/train_size[0]
    P2 = P2/train_size[0]
    P3 = P3/train_size[0]





























    ###### 以上由同学完成 #############

    test_result_arr=test_result_arr.squeeze()
    contract_arr = test_result_arr - test_arr[:, 0]
    goodones = len(np.argwhere(contract_arr == 0)) # 这个正确率可以作为评价指标
    print(goodones)

if __name__ =='__main__':
    main()