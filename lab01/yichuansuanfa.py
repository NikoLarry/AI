import numpy as np
import time
import sys
import random
import matplotlib.pyplot as plt


class Logger(object):
    """
    输出控制台
    """

    def __init__(self, fileN="result1.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def read(data_file, i):
    """
    读取txt文件
    :param data_file:输入测试样例文件路径
    :param i:样例i
    :return: W：背包最大容量(int)；N：物品数量(int)；w：每件物品的重量(list);v：每件物品的价值(list)
    """
    W = 0
    N = 0
    w = []
    v = []
    with open(data_file, 'r') as f:
        string = f.readlines()
        for j in range(len(string)):
            if string[j] == '(' + str(i) + ')' + ' \n':
                W = int(string[j + 2].split(' ')[0])
                N = int(string[j + 2].split(' ')[1])
                for k in range(1, N + 1):
                    w.append(int(string[j + k + 2].split(' ')[0]))
                    v.append(int(string[j + k + 2].split(' ')[1]))
    return W, N, w, v


def GA(W, N, w, v):
    """
    遗传算法解决0-1背包问题主函数
    :param W: 背包最大承重
    :param N: 物品总数
    :param w: 每件物品的重量
    :param v: 每件物品的价值
    :param save_fig_path: 样例i的收敛曲线存储路径
    :return: max_value:求解的放入背包的物品最大价值(int)；best_solu：放入背包的物品序号(list)
    背包最大载重bag_max_we，物品件数N，种群数量zq_num，精英池数量good_num（基因池前goodnum个都是精英池），
总迭代次数max_iteration,已迭代次数iteration_num，运行进程process=iteration_num/max_iteration
（zq_num*（N+3））大基因库矩阵all_arr（第0列存重量，1列存价值，2列存适应度），（zq_num*N）基因库矩阵gen_arr
（N*1）物品重量矩阵we_arr，（N*1）物品价值矩阵va_arr，
（zq_num*1）种群重量矩阵zq_we_arr，（zq_num*1）种群价值矩阵zq_va_arr，（zq_num*1）种群适应度矩阵zq_fi_arr
    item_max_we = max(w)  # 最大物品重量
    """
    # -----------------请同学们完成遗传算法-----------
    VbiW = {i : v[i]/w[i] for i in range(0, N)}  # 性价比字典用于解修正
    pm = 0.05       #变异概率
    #生成初始种群
    zq_num = 100
    zq = []
    for i in range(zq_num):
        ind = []
        for j in range(N):
            ind.append(str(random.randint(0, 1)))  # 为每一种选择方案编码成01序列，0代表未放入，1代表放入
        ind = ''.join(ind)
        zq.append(ind)
    #自适应度计算，使用解修正方法
    zq_syd = {}

    ##
    for i in zq:
        ind_val = 0
        ind_wei = 0
        loc = []            #记录放入的物品的序号
        for j in range(N):
            if i[j] == '1':
                ind_val += v[j]
                ind_wei += w[j]
                loc.append(j)
        while ind_wei >= W:
            vbw = []
            for l in loc:
                vbw.append(VbiW[l])
            vbw,loc = zip(*sorted(zip(vbw,loc)))
            rem_index = loc[0]                  #移除性价比最小的物品
            ind = list(i)
            ind[rem_index] = '0'
            i = ''.join(ind)
            ind_wei = ind_wei - w[rem_index]
            ind_val = ind_val - v[rem_index]
            loc = loc[1:]
        zq_syd.update({i:ind_val})

    max_iteration = 1000
    iteration_num = 0

    elite_ind = []
    elite_syd = []
    while iteration_num < max_iteration:        #迭代
        # 将精英解池中的个体放入种群
        for i,s in zip(elite_ind,elite_syd):
            zq_syd.update({i:s})
        #建立轮盘
        zq_syd = sorted(zq_syd.items(), key = lambda x:x[1])        #将种群按照个体适应度从小到大排序
        zq_syd = {i[0] : i[1] for i in zq_syd}
        zq_ind = list(zq_syd.keys())
        syd_list = list(zq_syd.values())

        # 将种群中最优个体放入精英解池中
        elite_ind.append(zq_ind[-1])
        elite_syd.append(syd_list[-1])

        iteration_num += 1
        if iteration_num == max_iteration:
            break
        sum_syd = sum(syd_list)
        proba = [i/sum_syd for i in syd_list]                   #计算每个个体的轮盘概率
        for i in range(1,len(proba)):
            proba[i] = proba[i-1]+proba[i]
        max_off= zq_num             #生产后代个数
        off_num = 0
        new_zq = {}
        #选择出父代个体
        while off_num < max_off:
            pointer1 = random.random()
            pointer2 = random.random()
            father1 = ''
            father2 = ''
            for i in range(len(proba)):
                if pointer1 < proba[i]:
                    father1 += zq_ind[i]
                    break
            for i in range(len(proba)):
                if pointer2 < proba[i]:
                    father2 += zq_ind[i]
                    break
        # 使用单点交叉
            Crossover_point = random.randint(0,N)
            offspring = father1[0:Crossover_point]+father2[Crossover_point:N]

            mutation_point = random.randint(0,N-1)            #随机选一点进行变异判定
            if random.random() < pm:
                muta_va = (int(offspring[mutation_point])+1)%2
                offspring = offspring[:mutation_point]+str(muta_va)+offspring[mutation_point+1:]

            # 计算后代适应度后将后代加入种群
            ind_val = 0
            ind_wei = 0
            loc = []
            for j in range(N):
                if offspring[j] == '1':
                    ind_val += v[j]
                    ind_wei += w[j]
                    loc.append(j)
            while ind_wei >= W:
                vbw = []
                for l in loc:
                    vbw.append(VbiW[l])
                vbw, loc = zip(*sorted(zip(vbw, loc)))
                rem_index = loc[0]  # 移除性价比最小的物品
                ind = list(offspring)
                ind[rem_index] = '0'
                offspring = ''.join(ind)
                ind_wei = ind_wei - w[rem_index]
                ind_val = ind_val - v[rem_index]
                loc = loc[1:]
            new_zq.update({offspring: ind_val})
            off_num += 1
            zq_syd = new_zq         #新种群替代旧种群
    y = []
    for i in elite_ind:
        solu = list(i)
        solu = np.array(list(map(int,solu)))
        y.append(solu.dot(v))
    x = range(len(y))
    plt.plot(x,y)
    plt.xlabel('Generations')
    plt.ylabel('The best total value')
    plt.grid()
    max_value=y[-1]
    best_solu = solu

    # -----------------以上由同学完成---------------
    return max_value, best_solu


if __name__ == '__main__':
    data_file = "sackdata.txt"
    sys.stdout = Logger("result1.txt")
    qinkong = open('result1.txt', 'w')  # 把之前数据清空，以免多次运行后数据太多
    qinkong.close()
    W, N, w, v = read(data_file, 5)
    start = time.time()

    max_value, best_solu = GA(W, N, w, v)
    assert len(best_solu) == N, '物品的件数为%d，给出的方案长度应当与之匹配' % N
    assert best_solu.dot(v) == max_value, '最大价值与给出的方案不匹配'
    assert best_solu.dot(w) < W, '给出的方案超重'
    print(max_value)
    plt.show()
