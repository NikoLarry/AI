import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random



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
    item_max_we = max(w)  # 最大物品重
    """
    # -----------------请同学们完成遗传算法-----------
    # 参数
    population_size = 100             # 初始种群数量
    max_iterations = 1000             # 迭代次数
    crossover_rate = 0.8
    mutation_rate = 0.05
    elite_size = 10                   # 精英解保持

    # 初始化种群
    def initialize_population():
        return np.random.randint(2, size=(population_size, N))

    # 计算适应度
    def fitness(individual):
        total_weight = np.sum(individual * w)
        total_value = np.sum(individual * v)
        if total_weight > W:
            return  -2*total_value            # 罚函数
        return total_value

    # 选择
    def selection(population, fitnesses):
        selected_indices = np.random.choice(range(population_size), size=population_size,p=fitnesses / np.sum(fitnesses))
        return population[selected_indices]

    # 交叉
    def crossover(parent1, parent2):
        if np.random.rand() < crossover_rate:
            point = np.random.randint(1, N - 1)
            return np.concatenate((parent1[:point], parent2[point:]))
        return parent1 if np.random.rand() < 0.5 else parent2

    # 突变
    def mutate(individual):
        for i in range(N):
            if np.random.rand() < mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    # 进化
    def create_new_population(population, fitnesses):
        minfitness = min(fitnesses)
        fitnesses = [(i - minfitness + 1) for i in fitnesses]
        new_population = []
        elite_indices = np.argsort(fitnesses)[-elite_size:]
        elites = population[elite_indices]
        new_population.extend(elites)
        while len(new_population) < population_size:
            parents = selection(population, fitnesses)
            offspring = crossover(parents[0], parents[1])
            offspring = mutate(offspring)
            new_population.append(offspring)
        return np.array(new_population)


    population = initialize_population()

    iterate_value = []
    for iteration in range(max_iterations):
        fitnesses = np.array([fitness(individual) for individual in population])
        population = create_new_population(population, fitnesses)
        iterate_index = np.argmax(fitnesses)
        iterate_value.append(fitnesses[iterate_index])

    # Get the best solution
    fitnesses = np.array([fitness(individual) for individual in population])
    best_index = np.argmax(fitnesses)
    best_individual = population[best_index]
    max_value = fitnesses[best_index]
    best_solu = best_individual
    iterate = np.linspace(0,max_iterations,max_iterations)
    plt.plot(iterate,iterate_value)
    plt.title("iteration of result")
    plt.xlabel("iteration times")
    plt.ylabel("total value")
    plt.show()


    # -----------------以上由同学完成---------------

    return max_value, best_solu


if __name__ == '__main__':
    data_file = "sackdata.txt"
    sys.stdout = Logger("result1.txt")
    qinkong = open('result1.txt', 'w')  # 把之前数据清空，以免多次运行后数据太多
    qinkong.close()
    W, N, w, v = read(data_file, 4)
    start = time.time()

    max_value, best_solu = GA(W, N, w, v)
    print(max_value,np.dot(best_solu,w))
    assert len(best_solu) == N, '物品的件数为%d，给出的方案长度应当与之匹配' % N
    assert best_solu.dot(v) == max_value, '最大价值与给出的方案不匹配'
    assert best_solu.dot(w) <= W, '给出的方案超重'
    print(best_solu)
