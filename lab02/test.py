import numpy as np
import random
from math import sqrt, pi, exp
import matplotlib.pyplot as plt

def te_us_from_data(test_num_list, data_arr):
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
    with open('wine.data', 'r') as f:
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

    train_features = used_arr[:, 1:]  # 训练集共142个数据
    train_labels = used_arr[:, 0]
    test_features = test_arr[:, 1:]  # 测试集共36个数据

    # PCA降维
    train_features_mean = train_features.mean(axis=0)
    train_features_centered = train_features - train_features_mean
    cov_matrix = np.cov(train_features_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = eigenvalues.argsort()[::-1]  # 按降序排列
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors[:, :2]  # 取前两个主成分
    train_pca = np.dot(train_features_centered, eigenvectors)
    # print(eigenvectors)
    # 贝叶斯分类
    class_means = []
    class_vars = []
    priors = []

    for label in np.unique(train_labels):
        features = train_pca[train_labels == label]
        class_means.append(features.mean(axis=0))
        class_vars.append(features.var(axis=0))
        priors.append(len(features) / len(train_labels))
    # print("fangcha",class_vars,"avg",class_means)
    # 对测试集进行PCA
    test_features_centered = test_features - train_features_mean
    test_pca = np.dot(test_features_centered, eigenvectors)
    test_result_arr = []
    # print(test_pca)
    for sample in test_pca:
        posteriors = []
        for mean, var, prior in zip(class_means, class_vars, priors):
            likelihood = np.prod(1 / np.sqrt(2 * pi * var) * np.exp(- (sample - mean) ** 2 / (2 * var)))
            print(var)

            posterior = likelihood * prior
            posteriors.append(posterior)

        test_result_arr.append(np.argmax(posteriors) + 1)

    test_result_arr = np.array(test_result_arr)

    test_result_arr = test_result_arr.squeeze()
    contract_arr = test_result_arr - test_arr[:, 0]
    goodones = len(np.argwhere(contract_arr == 0))  # 这个正确率可以作为评价指标
    print(f'Number of correctly classified samples: {goodones}')
    print(f'Accuracy: {goodones / len(test_result_arr):.2f}')

if __name__ == '__main__':
    main()
