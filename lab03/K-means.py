import numpy as np
from sklearn import metrics


def main():
    # 读取数据
    f = open('/data/bigfiles/3a37376f-3bc6-4f6d-a49a-99789991aa7b.data', 'r')
    data_big_list = f.readlines()  # 每个元素是以\n结尾的一行字符串
    all_data_arr = np.ones((178, 14))
    for create_i in range(178):
        all_data_arr[create_i,] = (data_big_list[create_i].rstrip('\n')).split(',')
    np.random.shuffle(all_data_arr)  # 洗牌（行之间进行洗牌）
    data_arr = all_data_arr[:, 1:]
    ########################## 同学完成 #####################

    # task：输出一个result_class
    ########### 同学完成 ######################
    true_class = all_data_arr[:, 0]
    final_art_value = metrics.adjusted_rand_score(true_class, result_class)  # 兰德系数
    print(final_art_value)


if __name__ == '__main__':
    main()