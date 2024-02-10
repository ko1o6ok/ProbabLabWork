# Часть - 1
# -----------------------------------------
# Генерирование нормального распределения

import random
import math

from matplotlib import patches
from prettytable import PrettyTable

import matplotlib.pyplot as plt
import numpy as np


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


# print("Содержание задачи")
# print("с.в. theta распределена нормально среднее a, дисперсия sigma^2")
# print("с.в. eta = exp(theta)")
# a = float(input("Введите среднее a:\n"))
# D = float(input("Введите дисперсию D:\n"))
#
# arr = [generate_eta(a,D) for _ in range(3)]
#
# arr_res = sorted(arr)
# for c in arr_res:
#     print(c)

# -----------------------------------------
# Часть - 2

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
n = 100_000  # Количество экспериментов

right_border = 10  # Правая граница по eta для отрисовки

data = [generate_eta(a, D) for _ in range(n)]  # Данные экспериментов

x_mean = sum(data) / n  # Выборочное среднее
S2 = sum([(x - x_mean) ** 2 for x in data]) / n  # Выборочная дисперсия
sorted_data = sorted(data)
R = sorted_data[-1] - sorted_data[0]  # Размах выборки

Expectation = E(a, D)  # Теоретическое матожидание для данных параметров
Var = Variance(a, D)  # Теоретическая дисперсия для данных параметров

# Выборочная медиана
if n % 2 == 0:
    k = n // 2
    Me = (sorted_data[k] + sorted_data[k + 1]) / 2
else:
    k = (n - 1) // 2
    Me = sorted_data[k + 1]

tab = PrettyTable(['E_eta', 'x_mean', '|E_eta-x_mean|', 'D_eta', 'S^2', '|D_eta-S^2|', 'M_e', 'R'])
tab.add_row([Expectation, x_mean, abs(Expectation - x_mean), Var, S2, abs(Var - S2), Me, R])
print(tab)

# Множество значений теор. интегр. функции распределения для 0<= eta <= right_border
r = np.arange(0, right_border, 0.01)  # Диапазон значений
teor_vals = [F(t, a, D) for t in r]

# Множество значений выборочной функции распределения
sample_vals = []

current_index = 0

for eta in r:
    while sorted_data[current_index] < eta:
        if current_index != (n - 1):
            current_index += 1
        else:
            break
    sample_vals.append((current_index + 1) / n)

diff = max([abs(u - v) for u, v in zip(sample_vals, teor_vals)])
print("Максимальное расхождение теоретической и выборочной функций распределения:")
print("Diff = ", diff)
plt.plot(r, teor_vals)
plt.plot(r, sample_vals, c='r')
plt.xlabel("Eta",fontsize = 18)
plt.xticks(fontsize = 18)
plt.ylabel("F(eta)",fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(['Теоретич. функция распределения', 'Выборочная функция распределения'])
plt.show()

print("Границы находятся в диапазоне (0;{}) !".format(right_border))
# k = int(input("Введите число границ промежутков: "))
# borders = [float(input("Введите {}-ю границу: ".format(i + 1))) for i in range(k)]

k = 50

borders = [right_border * t/(k+1) for t in range(1,k+1)]

# print(np.array(borders))

borders.append(right_border)
# Для каждого из участков определим число элементов, попадающих в промежуток
# Идём слева направо и смотрим
left_border = 0  # Левая граница
# b = borders[0] # Правая граница

elem_index = 0  # Индекс текущего рассматриваемого элемента

histogram = []
density_values = []
Z = []

fig, ax = plt.subplots()
for b in borders:
    num = 0  # Число элементов в промежутке

    while b > sorted_data[elem_index] > left_border:
        elem_index += 1
        num += 1
        if elem_index == n-1:
            break
    hist_val = num / (n * (b - left_border))
    histogram.append(hist_val)
    z = (b + left_border) / 2
    Z.append(z)
    density_values.append(g(z, a, D))
    ax.add_patch(patches.Rectangle((left_border,0),b-left_border,hist_val,color='b'))
    left_border = b

tbl2 = PrettyTable()
# tbl2.header = False

# Z.insert(0,"z_j")
# density_values.insert(0,"f_eta(z_j)")
# histogram.insert(0,"hist_j")

tbl2.add_column("z_j",Z)
tbl2.add_column("f_eta(z_j)",density_values)
tbl2.add_column("hist_j",histogram)


print(tbl2)
print("Максимальное расхождение плотности = ", max([abs(u-v) for u,v in zip(density_values,histogram)]))

# Построим гистограмму
plt.xlabel("eta",fontsize=18)
plt.xticks(fontsize=18)
plt.xlim([0,right_border])
plt.ylabel("f_eta",fontsize=18)
plt.yticks(fontsize=18)

v_g = [g(t,a,D) for t in r]
plt.plot(r,v_g,c = 'r',label = "Теоретическая плотность")
plt.ylim([0,max(v_g)+0.05])
plt.plot([-1,-2],[-1,-1],'b',label = "Гистограмма")

plt.legend(loc='upper right')
plt.show()