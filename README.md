# BigDataCases
### ***Задача 1***: Вычислить: (2 - 2√3i) / (1 + i√3)

Ответ:
<img src="cases/case6/scatter_plot.png" width="600">

Исходный код:  
```python
import numpy as np
import matplotlib.pyplot as plt

print("=== Задача 1: Действия в алгебраической форме ===")
print("Вычислить: (2 - 2√3i) / (1 + i√3)")

numerator = complex(2, -2 * np.sqrt(3))  # 2 - 2√3i
denominator = complex(1, np.sqrt(3))     # 1 + i√3

result = numerator / denominator

# Вывод результата в алгебраической форме
print(f"\nРезультат деления: {result.real:.3f} {'+' if result.imag >= 0 else '-'} {abs(result.imag):.3f}i")

# === ВИЗУАЛИЗАЦИЯ ===
plt.figure(figsize=(8, 6))
ax = plt.gca()
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Масштабируем оси для лучшего отображения
plt.xlim(-2.5, 3)
plt.ylim(-3.5, 2)

# Рисуем комплексные числа как векторы
vectors = [
    (numerator, 'Числитель (2 - 2√3i)', 'blue', 1.0),
    (denominator, 'Знаменатель (1 + i√3)', 'red', 1.2),
    (result, 'Результат', 'green', 1.5)
]

for z, label, color, scale in vectors:
    plt.quiver(0, 0, z.real, z.imag, 
               color=color, 
               scale=scale, 
               scale_units='xy', 
               angles='xy',
               width=0.004,
               label=label)

# Добавляем подписи к концам векторов
for z, label, color, _ in vectors:
    plt.text(z.real * 1.05, z.imag * 1.05, 
             f'({z.real:.2f}, {z.imag:.2f})',
             color=color,
             fontsize=9)

plt.title('Комплексные числа на комплексной плоскости')
plt.xlabel('Re')
plt.ylabel('Im')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```
### ***Кейс 2***: Обработка результатов анкетирования
Набор данных: [скачать набор данных](cases/case2/Результаты анкетирования.csv)  

В кейсе анализируется результаты актетирования из файла `Результаты анкетирования.csv`. Строятся три типа графиков:
- **Scatter plot**: Диаграмма рассеяния ответов на первый вопрос.
- **Bar plot**: Столбчатая диаграмма среднего результата при ответе на первый вопрос.
- **Box plot**: Коробчатая диаграмма распределения ответов на первый вопрос.

Графики:  

Исходный код:  
```python
# file: task_02_complex_trigonometric_simplified.py
import numpy as np
import matplotlib.pyplot as plt

print("=== Задача 2: Корень шестой степени в тригонометрической форме ===")
print("Вычислить: ∛(1 + √3i) (на самом деле корень 6-й степени)")

# Исходное комплексное число
z = 1 + 1j * np.sqrt(3)

# Вычисляем модуль и аргумент
r = np.abs(z)  # модуль
theta = np.angle(z)  # аргумент в радианах

# Параметры для корня 6-й степени
n = 6
r_root = r ** (1/n)
angles = [(theta + 2 * np.pi * k) / n for k in range(n)]

# Вывод результатов
print(f"\nИсходное число: {z.real:.2f} + {z.imag:.2f}i")
print(f"Модуль: {r:.2f}, Аргумент: {np.degrees(theta):.2f}°")
print(f"\nКорень {n}-й степени:")
print(f"Модуль корня: {r_root:.4f}")

# Визуализация
plt.figure(figsize=(8, 8))
ax = plt.gca()
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

# Рисуем исходное число
plt.quiver(0, 0, z.real, z.imag, 
           color='blue', 
           scale=1.5, 
           scale_units='xy', 
           angles='xy',
           width=0.004,
           label='Исходное число (1 + √3i)')

# Рисуем все корни
for k, angle in enumerate(angles):
    root = r_root * (np.cos(angle) + 1j * np.sin(angle))
    plt.quiver(0, 0, root.real, root.imag, 
               color='red', 
               scale=1.5, 
               scale_units='xy', 
               angles='xy',
               width=0.004,
               label=f'Корень {k+1}' if k == 0 else None)
    
    # Подпись угла
    plt.text(root.real * 1.1, root.imag * 1.1, 
             f'{np.degrees(angle):.1f}°',
             color='red',
             fontsize=8)

# Рисуем окружность радиуса корня
circle = plt.Circle((0, 0), r_root, color='gray', fill=False, linestyle='--', alpha=0.5)
ax.add_patch(circle)

plt.title(f'Корни {n}-й степени из (1 + √3i)')
plt.xlabel('Re')
plt.ylabel('Im')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis('equal')
plt.show()

# Вывод корней в тригонометрической форме
print("\n=== КОРНИ В ТРИГОНОМЕТРИЧЕСКОЙ ФОРМЕ ===")
for k, angle in enumerate(angles):
    deg_angle = np.degrees(angle)
    print(f"z_{k+1} = {r_root:.4f} (cos({deg_angle:.1f}°) + i·sin({deg_angle:.1f}°))")
```

### ***Кейс 3***: Анализ посещаемости занятий
Набор данных: [скачать набор данных](cases/case3/Посещяемость занятий.csv)  

В кейсе анализируется посещаемость студентов из файла `Посещаемость занятий.csv`. Строятся три типа графиков:
- **Scatter plot**: Диаграмма рассеяния посещаемости по группам.
- **Bar plot**: Столбчатая диаграмма средней посещаемоти по группам.
- **Box plot**: Коробчатая диаграмма распределения посещаемости по граппам.

Графики:  


Исходный код:  
```python
# file: task_03_polynomial_factorization_simplified.py
import numpy as np
import matplotlib.pyplot as plt

print("=== Задача 3: Разложение многочлена на неприводимые множители ===")
print("x⁴ - 2x³ + x² - 8x - 12")

# Коэффициенты многочлена (от старшей степени к младшей)
coefficients = [1, -2, 1, -8, -12]

# Нахождение корней
roots = np.roots(coefficients)

# Классификация корней
real_roots = []
complex_pairs = []

for root in roots:
    if np.isclose(root.imag, 0, atol=1e-5):
        real_roots.append(root.real)
    else:
        # Добавляем только один из сопряженных корней
        if all(not np.isclose(root, c, atol=1e-5) for c in complex_pairs):
            complex_pairs.append(root)

# Формирование множителей
factors = []

# Добавляем линейные множители для действительных корней
for r in real_roots:
    sign = '-' if r >= 0 else '+'
    factors.append(f"(x {sign} {abs(r):.2f})")

# Добавляем квадратичные множители для комплексных пар
for z in complex_pairs:
    if z.imag > 0:  # обрабатываем только верхнюю половину
        a = -2 * z.real
        b = abs(z) ** 2
        sign_a = '-' if a >= 0 else '+'
        factors.append(f"(x² {sign_a} {abs(a):.2f}x + {b:.2f})")

# Вывод разложения
print("\n=== РАЗЛОЖЕНИЕ НА НЕПРИВОДИМЫЕ МНОЖИТЕЛИ ===")
print("P(x) = " + " · ".join(factors))

# === ВИЗУАЛИЗАЦИЯ ===
plt.figure(figsize=(10, 6))

# Генерируем данные для графика
x = np.linspace(-2, 4, 500)
y = np.polyval(coefficients, x)

# Строим график
plt.plot(x, y, 'b-', linewidth=2, label='P(x) = x⁴ - 2x³ + x² - 8x - 12')

# Отмечаем действительные корни
for r in real_roots:
    plt.plot(r, 0, 'ro', markersize=8, label=f'Корень x={r:.2f}')

# Настройка графика
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.title('График многочлена и его корни')
plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.xlim(-2.5, 4.5)
plt.ylim(-40, 40)

plt.show()

# Проверка разложения
recovered_coeffs = np.polyfromroots(roots)
print("\n=== ПРОВЕРКА ===")
print(f"Коэффициенты исходного многочлена: {coefficients}")
print(f"Коэффициенты восстановленного: {recovered_coeffs}")
print(f"Совпадают? {np.allclose(coefficients, recovered_coeffs)}")
```

### ***Кейс 4***: Обработка результатов тестирования
Набор данных: [скачать набор данных](cases/case4/Результаты тесирования.csv)  

В кейсе анализируются результаты тестирования студентов из файла `Результаты тесирования.csv`. Строятся три типа графиков:
- **Scatter plot**: Диаграмма рассеяния правильных ответов по студентам.
- **Bar plot**: Столбчатая диаграмма среднего результата правильных ответов по студентам.
- **Box plot**: Коробчатая диаграмма распределения результата правильных ответов по студентам.

Графики:  
<img src="cases/case4/scatter_plot.png" width="600">
<img src="cases/case4/bar_plot.png" width="600">
<img src="cases/case4/box_plot.png" width="600">

Исходный код:  
```python
# file: task_04_determinant_properties_simplified.py
import numpy as np

print("=== Задача 4: Вычисление определителя свойствами ===")

# Исходная матрица
A = np.array([
    [5,  2, -5,  4,  5],
    [9, -3, -7, -5, -5],
    [-2, 4,  2,  8,  3],
    [5,  3, -2,  8,  3],
    [-4,-3,  4, -6, -3]
], dtype=float)

print("\n=== ИСХОДНАЯ МАТРИЦА ===")
for row in A:
    print("  ".join(f"{x:5.0f}" for x in row))

# === Приведение к ступенчатому виду ===
B = A.copy()
n = B.shape[0]
sign = 1  # Для учета перестановок строк

for i in range(n):
    # Ищем ненулевой элемент в столбце
    if B[i, i] == 0:
        for j in range(i + 1, n):
            if B[j, i] != 0:
                B[[i, j]] = B[[j, i]]
                sign *= -1
                break
    
    # Обнуляем элементы под главным
    for j in range(i + 1, n):
        factor = B[j, i] / B[i, i]
        B[j, :] -= factor * B[i, :]

print("\n=== МАТРИЦА В СТУПЕНЧАТОМ ВИДЕ ===")
for row in B:
    print("  ".join(f"{x:5.2f}" for x in row))

# Вычисляем определитель
det = np.prod(np.diag(B)) * sign
print(f"\nОпределитель: {det:.2f}")
```

### ***Кейс 5***: Анализ научной активности
Набор данных: [скачать набор данных](cases/case5/Научная активность.csv)  

В кейсе анализируется научная активность студентов из файла `Научная активность.csv`. Строятся три типа графиков:
- **Scatter plot**: Диаграмма рассеяния количества и типов публикаций по годам.
- **Bar plot**: Столбчатая диаграмма среднего года по типам публикаций.
- **Box plot**: Коробчатая диаграмма распределения годов по типам публикаций.

Графики:  


Исходный код:  
```python
# file: task_05_system_solution_simplified.py
import numpy as np

print("=== Задача 5: Система линейных уравнений ===")
print("""
Система:
  x - 2y +  z = -1
 2x -  y - 4z =  7
  x +  y - 2z =  5
""")

# Матрица системы и вектор свободных членов
A = np.array([
    [1, -2,  1],
    [2, -1, -4],
    [1,  1, -2]
])
b = np.array([-1, 7, 5])

# Проверка совместности (теорема Кронекера-Капелли)
rank_A = np.linalg.matrix_rank(A)
augmented = np.hstack([A, b.reshape(-1, 1)])
rank_aug = np.linalg.matrix_rank(augmented)

print(f"Ранг матрицы A: {rank_A}")
print(f"Ранг расширенной матрицы: {rank_aug}")

if rank_A == rank_aug:
    print("Система совместна и имеет единственное решение")
else:
    print("Система несовместна (решений нет)")

# === а) Метод Гаусса ===
print("\n=== а) МЕТОД ГАУССА ===")
aug = np.hstack([A.astype(float), b.reshape(-1, 1)])
n_rows, n_cols = aug.shape

# Прямой ход
for i in range(n_rows):
    pivot = aug[i, i]
    aug[i, :] /= pivot
    for j in range(i + 1, n_rows):
        factor = aug[j, i]
        aug[j, :] -= factor * aug[i, :]

# Обратный ход
x = np.zeros(n_rows)
for i in range(n_rows - 1, -1, -1):
    x[i] = aug[i, -1] - np.dot(aug[i, i+1:n_cols-1], x[i+1:])

print(f"Решение (Гаусс): x={x[0]:.3f}, y={x[1]:.3f}, z={x[2]:.3f}")

# === б) Метод Крамера ===
print("\n=== б) МЕТОД КРАМЕРА ===")
det_A = np.linalg.det(A)
print(f"Определитель матрицы A: {det_A:.3f}")

solutions = []
for i in range(A.shape[1]):
    A_i = A.copy()
    A_i[:, i] = b
    det_i = np.linalg.det(A_i)
    solutions.append(det_i / det_A)
    var_name = ['x', 'y', 'z'][i]
    print(f"{var_name} = {det_i:.3f} / {det_A:.3f} = {solutions[i]:.3f}")

print(f"Решение (Крамер): x={solutions[0]:.3f}, y={solutions[1]:.3f}, z={solutions[2]:.3f}")

# === в) Матричный метод ===
print("\n=== в) МАТРИЧНЫЙ МЕТОД ===")
A_inv = np.linalg.inv(A)
solution = A_inv @ b
print(f"Решение (матричный): x={solution[0]:.3f}, y={solution[1]:.3f}, z={solution[2]:.3f}")

# === ПРОВЕРКА ===
print("\n=== ПРОВЕРКА ===")
check = A @ solution
print(f"A·X = {check}")
print(f"b   = {b}")
print(f"Совпадает? {np.allclose(check, b)}")
```

### ***Кейс 6***: Обработка данных о трудоустройстве выпускников
Набор данных: [скачать набор данных](cases/case6/Трудоустройство.csv)  

В кейсе анализируется трудоустройство студентов из файла `Трудоустройство.csv`. Строятся три типа графиков:
- **Scatter plot**: Диаграмма рассеяния зарплат по должностям.
- **Bar plot**: Столбчатая диаграмма средней зарплаты по должностям.
- **Box plot**: Коробчатая диаграмма распределения зарплат по должностям.

Графики:  
<img src="cases/case6/scatter_plot.png" width="600">
<img src="cases/case6/bar_plot.png" width="600">
<img src="cases/case6/box_plot.png" width="600">

Исходный код:  
```python
# file: task_08_vector_decomposition_simplified.py
import numpy as np

print("=== Задача 8: Разложение вектора по базису ===")

# Исходные векторы
d = np.array([-2, 11, -2])
a = np.array([1, 2, -3])
b = np.array([3, -3, -2])
c = np.array([-1, 4, 2])

print(f"Вектор d: {d}")
print(f"Базисные векторы:")
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  c = {c}")

# Составляем матрицу базиса (столбцы - векторы a, b, c)
A = np.column_stack((a, b, c))

# Проверяем, является ли набор базисом (определитель != 0)
det_A = np.linalg.det(A)
print(f"\nОпределитель матрицы базиса: {det_A:.2f}")

if abs(det_A) > 1e-10:
    # Решаем систему уравнений A * [x, y, z]^T = d
    coords = np.linalg.solve(A, d)

    print(f"\nКоординаты вектора d в базисе (a, b, c):")
    print(f"x = {coords[0]:.3f}")
    print(f"y = {coords[1]:.3f}")
    print(f"z = {coords[2]:.3f}")

    # Формула разложения
    print(f"\nРазложение: d = ({coords[0]:.3f})a + ({coords[1]:.3f})b + ({coords[2]:.3f})c")

    # Проверка
    check = coords[0] * a + coords[1] * b + coords[2] * c
    print(f"\nПроверка: {check}")
    print(f"Совпадает с исходным d? {np.allclose(check, d)}")
else:
    print("\nВекторы линейно зависимы и не образуют базис. Разложение невозможно.")
```

### ***Кейс 7***: Анализ поведения пользователей в социальных сетях
Набор данных: [скачать набор данных](cases/case7/social_networks.csv)  

В кейсе анализируется интернет активность из файла `social_networks.csv`. Строятся три типа графиков:
- **Scatter plot**: Диаграмма рассеяния затрачиваемого времени с возрастом и полом.
- **Bar plot**: Столбчатая диаграмма среднего потраченного времени и места проживания.
- **Box plot**: Коробчатая диаграмма возраста и операционной системы.

Графики:  
<img src="cases/case7/scatter_plot.png" width="600">
<img src="cases/case7/bar_plot.png" width="600">
<img src="cases/case7/box_plot.png" width="600">

Исходный код:  
```python
# file: task_09_eigen_problem_simplified.py
import numpy as np

print("=== Задача 9: Собственные числа и векторы матрицы A ===")

# Определение матрицы
A = np.array([
    [2,  2, -1],
    [0,  1, -1],
    [0,  3,  5]
])

print("Матрица A:")
print(A)

# Вычисление собственных значений и векторов
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\n=== СОБСТВЕННЫЕ ЗНАЧЕНИЯ ===")
for i, val in enumerate(eigenvalues):
    print(f"λ{i+1} = {val.real:.4f}{' + ' + str(val.imag)+'i' if val.imag != 0 else ''}")

print("\n=== СОБСТВЕННЫЕ ВЕКТОРЫ ===")
for i in range(len(eigenvalues)):
    vec = eigenvectors[:, i]
    print(f"Для λ{i+1} = {eigenvalues[i].real:.4f}:")
    print(f"  [{vec[0].real:.4f}, {vec[1].real:.4f}, {vec[2].real:.4f}]^T")
    
    # Проверка
    Av = A @ vec
    lv = eigenvalues[i] * vec
    print(f"  Проверка (A·v == λ·v): {np.allclose(Av, lv)}")

# Дополнительная информация
print("\n=== ХАРАКТЕРИСТИЧЕСКИЙ МНОГОЧЛЕН ===")
coeffs = np.poly(A)
print(f"λ³ + ({coeffs[1]:.2f})λ² + ({coeffs[2]:.2f})λ + ({coeffs[3]:.2f}) = 0")
```
