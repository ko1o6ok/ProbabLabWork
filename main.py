import random
import math


# Часть - 1
# -----------------------------------------
# Генерирование нормального распределения
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

def generate_eta(a,D):
    return math.exp(generate_theta(a, D))
print("Содержание задачи")
print("с.в. theta распределена нормально среднее a, дисперсия sigma^2")
print("с.в. eta = exp(theta)")
a = float(input("Введите среднее a:\n"))
D = float(input("Введите дисперсию D:\n"))

arr = [generate_eta(a,D) for _ in range(3)]

arr_res = sorted(arr)
for c in arr_res:
    print(c)

# -----------------------------------------
# Часть - 2

# Теоретическая плотность распределения
# eta - значение с.в.
# a - среднее нормального распределения
# D - дисперсия нормального распределения
# def g(eta,a,D):
#     sigma = math.sqrt(D) # Среднеквадратичное отклонение нормального распределения
#     t = (math.log(eta)-a)/sigma
#     math.exp(-)