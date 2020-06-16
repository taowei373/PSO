import numpy as np
import matplotlib.pyplot as plt
import math

class PSO(object):

    def __init__(self, population_size, max_steps, dim, v_max, target):
        self.w = 0.9
        self.wmin = 0.4
        self.wmax = self.w

        self.c1 = 2
        self.c2 = 2

        self.population_size = population_size  # 粒子群数量
        self.dim = dim  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        self.target = target
        #self.x_bound = x_bound  # 解空间范围 例如[-10, 10]
        self.v_max = np.ones(shape=(self.population_size, self.dim)) * v_max # 最大速度
        
        #self.x = np.random.uniform(low=self.x_bound[0], high=self.x_bound[1], size=(self.population_size, self.dim))  # 初始化粒子群位置
        self.x = np.zeros(shape=(self.population_size, self.dim))
        self.v = np.random.rand(self.population_size, self.dim) # 初始化离子群速度
        fitness = self.calculate_fitness(self.x)
        self.p = self.x  # 个体最佳位置
        self.pg = self.x[np.argmin(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.min(fitness)  # 全局最佳适应度

    # 目标函数
    def calculate_fitness(self, x):
        t = []
        for i in range(self.population_size):
            t.append([self.target[0], self.target[1]])
        c = x - np.array(t)
        result = np.sum(np.square(c), axis=1)
        return result

    def evolve(self):
        evolve_data = []
        for step in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            
            # 确定惯性权重
            self.w = self.wmax - ((self.wmax - self.wmin) / self.max_steps) * step
            if self.w > self.wmax:
                self.wmax = self.w
            if self.w < self.wmin:
                self.wmin = self.w
            
            # 确定学习因子
            #self.c1 = 2 * math.pow(math.sin((math.pi / 2) * (1 - step / self.max_steps)), 2)
            #self.c2 = 2 * math.pow(math.sin((math.pi * step) / (2 * self.max_steps)), 2)

            # 更新速度和权重
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            #更新位置
            self.x = self.v + self.x
            fitness = self.calculate_fitness(self.x)
            
            # 需要更新的个体
            update_id = np.greater(self.individual_best_fitness, fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
            if np.min(fitness) < self.global_best_fitness:
                self.pg = self.x[np.argmin(fitness)]
                self.global_best_fitness = np.min(fitness)
            #print('best fitness: %.5f, mean fitness: %.5f' % (self.global_best_fitness, np.mean(fitness)))  
            # 封装路径数据
            d = dict()
            d['x'] = step
            d['y'] = self.global_best_fitness
            evolve_data.append(d)
        return evolve_data
 


pso = PSO(population_size=100, max_steps=2000, dim=2, v_max=1, target=[400, 300])
data = pso.evolve()
x = []
y = []
for t in data:
    x.append(t['x'])
    y.append(t['y'])
plt.plot(x, y, 'r')

font = {'family' : 'Times New Roman','weight' : 'normal','size' : 23}
plt.xlabel('index', font)
plt.ylabel('gpv', font)
plt.show()

