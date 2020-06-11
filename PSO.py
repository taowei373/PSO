import numpy as np
import matplotlib.pyplot as plt

class PSO(object):

    def __init__(self, population_size, max_steps, dim, learning_factor, v_max):
        self.w = 0.9
        self.c1 = learning_factor[0]
        self.c2 = learning_factor[1]
        self.population_size = population_size  # 粒子群数量
        self.dim = dim  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
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
            t.append([20, 30])
        c = x - np.array(t)
        result = np.sum(np.square(c), axis=1)
        return result

    def evolve(self):
        evolve_data = []
        for step in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)

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

            # 封装数据
            d = dict()
            d['x'] = step
            d['y'] = self.global_best_fitness
            evolve_data.append(d)
        return evolve_data
 
pso = PSO(population_size=10, max_steps=100, dim=2, learning_factor=[3, 3], v_max=1)
data = pso.evolve()
x = []
y = []
for t in data:
    x.append(t['x'])
    y.append(t['y'])
plt.plot(x, y, 'r')
plt.show()
