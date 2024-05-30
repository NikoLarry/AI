
import plotly.graph_objs as go
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def main():
    # 读取数据
    f = open('wine.data', 'r')
    data_big_list = f.readlines()  # 每个元素是以\n结尾的一行字符串
    all_data_arr = np.ones((178, 14))
    for create_i in range(178):
        all_data_arr[create_i,] = (data_big_list[create_i].rstrip('\n')).split(',')
    np.random.shuffle(all_data_arr)  # 洗牌（行之间进行洗牌）
    data_arr = all_data_arr[:, 1:]
    ########################## 同学完成 #####################
    def distance(a, b):
        d = 0
        for i in range(len(a)):
            d = d + (a[i] - b[i]) ** 2
        return d
    dim = 8
    data_mean = data_arr.mean(axis=0)
    data_arr1 = data_arr - data_mean                              # 对个样本属性数据中心化
                                                                # 降到指定维数
    X = np.cov(np.array(data_arr1.T))                             # 计算特征值与特征向量
    w, v = np.linalg.eig(X)
    idx = w.argsort()[::-1]         # 按降序排列
    eigenvectors = v[:, idx]
    dimension_array = eigenvectors[:, :dim]

    data_arr1 = np.dot(data_arr1, dimension_array)                                # 投影

    sigma = 0.1 # 方差缩放比例
    data_arr_normalize = data_arr1 - data_arr1.mean(axis=0)           # 数据中心化

    for i in range(len(data_arr1[0, :])):                             # 方差比例缩放,归一化[-1,1]
        data_arr_normalize[:, i] = data_arr_normalize[:, i] * np.var(data_arr_normalize[:, i]) * sigma
        data_arr_normalize[:, i] = data_arr_normalize[:, i] / max(np.max(data_arr_normalize[:, i]), -np.min(data_arr_normalize[:, i]))
    k1 = []
    k2 = []
    k3 = []

    K = 3                      # 聚类类别数

    point_index = np.random.choice([i for i in range(len(data_arr_normalize[:, 0]))], K, replace=False)
    point = np.array([data_arr_normalize[point_index[i], :] for i in range(K)])           # 初始化point
    D_min = np.zeros((len(data_arr_normalize[:, 0])))

    for m in range(1000):
        D = []
        for p in point:
            d = []
            for i in range(len(data_arr_normalize[:,0])):
                a = data_arr_normalize[i, :]
                dis = distance(a, p)
                d.append(dis)
            D.append(d)
        D_min = np.argmin(D,axis=0)
        for j in range(K):
            id = np.argwhere(D_min == j).T
            l = len(id[0])
            if l == 0:
                continue
            else:
                point[j] = np.mean(data_arr_normalize[id, :][0], axis=0)
    D_min = D_min + 1
    result_class = D_min.astype(float)
    # task：输出一个result_class
    ########### 同学完成 ######################
    true_class = all_data_arr[:, 0]
    final_art_value = metrics.adjusted_rand_score(true_class, result_class)  # 兰德系数
    print(final_art_value)



    ######################## 降维可视化 ####################
    d = 3
    data_mean = data_arr.mean(axis=0)
    data_arr = data_arr - data_mean  # 对个样本属性数据中心化
    # 降到指定维数
    X = np.cov(np.array(data_arr.T))  # 计算特征值与特征向量
    w, v = np.linalg.eig(X)
    idx = w.argsort()[::-1]  # 按降序排列
    eigenvectors = v[:, idx]
    dimension_array = eigenvectors[:, :d]
    data_arr = np.dot(data_arr, dimension_array)  # 投影
    for i in range(len(D_min)):
        if D_min[i] == 1:
            k1.append(data_arr[i])
        if D_min[i] == 2:
            k2.append(data_arr[i])
        if D_min[i] == 3:
            k3.append(data_arr[i])
    k1 = np.array(k1)
    k2 = np.array(k2)
    k3 = np.array(k3)


    # 创建散点图数据
    scatter1 = go.Scatter3d(
        x=k1[:,0],
        y=k1[:,1],
        z=k1[:,2],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
        )
    )
    scatter2 = go.Scatter3d(
        x=k2[:,0],
        y=k2[:,1],
        z=k2[:,2],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
        )
    )
    scatter3 = go.Scatter3d(
        x=k3[:,0],
        y=k3[:,1],
        z=k3[:,2],
        mode='markers',
        marker=dict(
            size=5,
            color='green',
        )
    )
    # 创建布局
    layout = go.Layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        )
    )
    # 创建Figure并绘制
    fig = go.Figure(data=[scatter1,scatter2,scatter3], layout=layout)
    fig.show()



if __name__ == '__main__':
    main()