import numpy as np
import time
import sys
import random
import matplotlib.pyplot as plt

class Logger(object):
    """
    输出控制台
    """

    def __init__(self, fileN="sackdata.txt"):
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
    herd = []
    n = 100       # 初始化种群数量
    M = 2
    P_cross = 0.5
    P_mutation = 0.2
    init_herd(n, N, herd)
    elite_size = 5
    iters = 100
    iteration = [i for i in range(iters)]
    iter_value = []


    for i in range(iters):
        elite = select_elite(W, w, v, M, elite_size, herd)
        offspring = creat_offspring(W,w,n,herd,M,P_cross,P_mutation)
        herd = offspring + elite

        bestfit1 = fitness(W, w, v, herd, M)
        id1 = np.array(bestfit1).argmax()
        iter_value.append(np.dot(herd[id1],v))
    bestfit = fitness(W,w,v,herd,M)
    id = np.array(bestfit).argmax()
    max_value = np.dot(herd[id],v)
    best_solu = herd[id]
    plt.plot(iteration,iter_value)
    plt.show()
    # -----------------以上由同学完成---------------

    return max_value, best_solu



def creat_offspring(W,w,size,herd,M,P_cross,P_mutation):
    offspring1 = []
    while len(offspring1) <= size:
        parent = select_parent(W, w, 20, M,herd)
        offspring = cross(P_cross,parent)
        offspring = mutation(P_mutation,offspring)
        offspring1.extend(offspring)
    return offspring1





def init_herd(n,N,herd):
    while len(herd) < n:
        gene = [random.randint(0, 1) for i in range(N)]
        herd.append(gene)
    return

def fitness(W,w,v,herd,M):          #  适应度计算
    herd_fitness = []
    for individual in herd:
        weight = sum(np.multiply(individual,w))
        value = sum(np.multiply(individual,v))
        if weight > W:                          # 增加罚函数
            herd_fitness.append(0)
        else:
            herd_fitness.append(value)
    min_fitness = min(herd_fitness)
    # total = sum(herd_fitness)
    fit = [(i - min_fitness + 1) for i in herd_fitness]
    return fit



# 选择操作
def select_elite(W,w,v,M,elite_size,herd):  # N为每次选择个体数
    fit = fitness(W,w,v,herd,M)
    if fit == 0:
        while np.dot(herd[0],w)>1000:
            herd[0] = [random.randint(0, 1) for i in range(N)]
    # print(fit)
    elite = [herd[i] for i in np.array(fit).argsort()[::-1][-elite_size:]]  # 选出种群精英解
    return elite

def select_parent(W,w,N,M,herd):
    fit = fitness(W, w, v, herd, M)
    total = float(sum(fit))
    wheel_select = [fit[i]/total for i in range(len(fit))]
    # print(wheel_select)
    herd_index = np.random.choice(range(len(fit)), size = N, p = wheel_select)
    new_herd = [herd[i] for i in herd_index]
    # P = 0
    # for i in range(len(fit)):  # 计算每个个体价值概率
    #     P = P+fit[i]/total
    #     wheel_select.append(P)
    # print(len(wheel_select))
    # for j in range(N):
    #     p = random.uniform(0,1)
    #     for k in range(len(wheel_select)-1):
    #         if p >= wheel_select[k] and p < wheel_select[k+1]:
    #             new_herd.append(herd[k])
    return new_herd


def cross(P,herd):
    new_herd = []
    for i in range(len(herd)):
        p = random.uniform(0,1)
        x1 = herd[i]
        if p <= P:
            x2 = herd[random.randint(0,len(herd)-1)]
            cross_position = len(herd[0])//2
            cut = x2[cross_position:]
            x2[cross_position:] = x1[cross_position:]
            x1[cross_position:] = cut
            new_herd.append(x1)
            new_herd.append(x2)
    return new_herd

def mutation(P,herd):
    gene_len = len(herd[0])
    for mutation_gene in herd:
        for i in range(5):
            if random.uniform(0,1) <= P:
                k = random.randint(0,gene_len-1)
                if mutation_gene[k] == 1:
                    mutation_gene[k] = 0   # 一次发生10个点位突变
                else:
                    mutation_gene[k] = 1
    return herd





if __name__ == '__main__':
    data_file = "sackdata.txt"
    sys.stdout = Logger("result1.txt")
    qinkong = open('result1.txt', 'w')  # 把之前数据清空，以免多次运行后数据太多
    qinkong.close()
    W, N, w , v = read(data_file, 5)
    start = time.time()
    max_value, best_solu = GA(W, N, w, v)
    weight = np.dot(best_solu,w)
    print("value:",max_value,"weight:",weight ,"\n", best_solu)
    # assert len(best_solu) == N, '物品的件数为%d，给出的方案长度应当与之匹配' % N
    # assert best_solu.dot(v) == max_value, '最大价值与给出的方案不匹配'
    # assert best_solu.dot(w) < W, '给出的方案超重'
    # print(max_value)

