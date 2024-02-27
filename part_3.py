# ---------------------------------------------------------------
# Часть - 3: Проверка гипотезы о виде распределения
# Используем критерий хи-квадрат
# Наша с.в. принимает положительные значения, поэтому
# разобъём положительную полуось на интервалы
# причём сделаем их равновероятными
# Для этого разобъём значения интегральной функции распределения на интервалы одинаковой длины и возьмём обратную функцию



import random
import math

from matplotlib import patches

import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats


# a - среднее
# D - дисперсия
def generate_theta(a, D):
    # Метод полярных координат Бокса-Мюллера-Марсальи
    # Шаг - 1:
    # Получим две независимые величины,
    # Равномерно распределённые на (0,1)
    S = 2
    while S >= 1:
        U1 = random.random()
        U2 = random.random()
        V1 = 2 * U1 - 1
        V2 = 2 * U2 - 1
        # Шаг - 2:
        S = V1 * V1 + V2 * V2
        # Шаг - 3: Если S >= 1, то возвращаемся
    # Шаг - 4:
    eta = V1 * math.sqrt(-2 * math.log(S) / S)
    return a + eta * math.sqrt(D)


def generate_eta(a, D):
    return math.exp(generate_theta(a, D))


# Теоретическая плотность распределения
# eta - значение с.в.
# a - среднее нормального распределения
# D - дисперсия нормального распределения
def g(eta, a_, D_):
    if eta <= 0:
        return 0
    sigma = math.sqrt(D_)  # Среднеквадратичное отклонение нормального распределения
    t = (math.log(eta) - a_) / sigma
    return math.exp(-t * t / 2) / (sigma * math.sqrt(2 * math.pi) * eta)


# Теоретическое матожидание
# a_ - среднее нормального распределения
# D_ - дисперсия нормального распределения
def E(a_, D_):
    return math.exp(a_ + D_ / 2.0)


# Теоретическая дисперсия
# a_ - среднее нормального распределения
# D_ - дисперсия нормального распределения
def Variance(a_, D_):
    return math.exp(2 * D_ + 2 * a_) - math.exp(2 * a + D)


# Теоретическая интегральная функция распределения (не эффективная реализация)
# a_ - среднее нормального распределения
# D_ - дисперсия нормального распределения
def F(eta, a_, D_):
    if eta <= 0:
        return 0

    dt = 0.001
    S = 0
    x = 0
    while x < eta:
        S += g(x, a_, D_) * dt
        x += dt
    return S


a = 1  # Среднее нормального распределения
D = 0.1  # Дисперсия нормального распределения
n = 1000_000  # Количество экспериментов

right_border = 10  # Правая граница по eta для отрисовки

data = [generate_eta(a, D) for _ in range(n)]  # Данные экспериментов

sorted_data = sorted(data)



# Обратная функция Лапласа
def inverse_laplace(probability):
    inv = stats.norm.ppf(probability, loc=a, scale=np.sqrt(D))
    return np.exp(inv)


k = int(input("Введите число интервалов: "))
# k = 10
# k = 2  # Число равновероятных участков, на которые разбиваем
# Поскольку последней правой границей точно будет бесконечность, то там считать обратную не нужно
probabilities = [j / k for j in range(1, k)]
x_pos = [inverse_laplace(probab) for probab in probabilities]  # Положения границ
# Снизу надо добавить ноль, а сверху бесконечность!
print("Выбраны границы интервалов для равновероятности")
print("Гипотеза H0: при таком выборе интервалов теоретические вероятности q_j = {}".format(1 / k))
alpha = float(input("Введите уровень значимости alpha = "))

# alpha = 0.5

x_pos.append(right_border + 1)

left_border = 0  # Левая граница

elem_index = 0  # Индекс текущего рассматриваемого элемента

R0 = 0.0  # Статистика критерия

histogram = []
density_values = []
fig, ax = plt.subplots()

region_probabs = []
for b in x_pos:
    num = 0  # Число элементов в промежутке

    if elem_index != n:
        while b > sorted_data[elem_index] > left_border:
            elem_index += 1
            num += 1
            if elem_index == n:
                break
    #hist_val = num / (n * (b - left_border))
    R0 = R0 + (num - n / k) ** 2 / (n / k)

    hist_val = num / (n * (b - left_border))
    z = (b + left_border) / 2
    region_probabs.append(num/n)
    density_values.append(g(z, a, D))
    histogram.append(hist_val)
    ax.add_patch(patches.Rectangle((left_border, 0), b - left_border, hist_val, color='b'))

    left_border = b

# Построим гистограмму
plt.xlabel("eta", fontsize=18)
plt.xticks(fontsize=18)
plt.xlim([0, right_border])
plt.ylabel("f_eta", fontsize=18)
plt.yticks(fontsize=18)

r = np.arange(0, right_border, 0.01)

v_g = [g(t, a, D) for t in r]
plt.plot(r, v_g, c='r', label="Теоретическая плотность")
plt.ylim([0, max(v_g) + 0.05])
plt.plot([-1, -2], [-1, -1], 'b', label="Гистограмма")

plt.legend(loc='upper right')
plt.show()

region_probabs = np.array(region_probabs)
print(region_probabs)

neg_F = 1-stats.chi2.cdf(R0,k-1)
print("R0 = {}".format(R0))
print("neg_F(R0) = {}".format(neg_F))
print("Гипотеза H0 {}".format("принята" if neg_F < alpha else "отвержена"))
