# Задачи линейной алгебры
### ***Задача 1***: Вычислить: (2 - 2√3i) / (1 + i√3)

Ответ:
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/abfa45e2-feda-4e76-bf45-464480d9a247" />


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

```
### ***Задача 2***: Вычислить: sqrt((1 + √3i))^6 


Ответ: <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/6956285e-d515-49e4-993b-14e84080d349" />
 

Исходный код:  
```python
import numpy as np
import matplotlib.pyplot as plt

print("=== Задача 2: Корень шестой степени в тригонометрической форме ===")
print("Вычислить: sqrt((1 + √3i))^6")

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

```

### ***Задача 3***: Разложить многочлен x⁴ - 2x³ + x² - 8x - 12 на неприводимые множители



Ответ: <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/65369e63-0651-4dce-a04a-3342bf27d8fa" />



Исходный код:  
```python
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


```

### ***Задача 4***: Вычислить 
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/5cf953bd-7f35-419e-bf7d-260224974fda" />



Ответ:<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/8e1f5eb0-f773-4b50-b2e9-7edcefa2aa35" />


Исходный код:  
```python
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

### ***Задача 5***: Решение системы линейных уравнений
x - 2y +  z = -1
 2x -  y - 4z =  7
  x +  y - 2z =  5


Ответ:  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/571f4a2c-fb15-4edf-910a-90bd05dc63ba" />



Исходный код:  
```python
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
```

### ***Задача 6***: Разложение вектора d по базису a b c
d = (-2, 11, -2)
a = (1, 2, -3)
b = (3, -3, -2)
c = (-1, 4, 2)

Ответ:  <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/0d24ec49-bc4c-49dd-8b1e-bb3913c8a56f" />



Исходный код:  
```python
import numpy as np

print("=== Задача 6: Разложение вектора по базису ===")

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
```

### ***Задача 7***: Найти cобственные числа и векторы матрицы A
<img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/93024dd2-ee66-4bae-853e-22735b072386" />


Ответ: <img width="600" height="600" alt="image" src="https://github.com/user-attachments/assets/be6ea077-56fe-4c59-b172-bb45ca235451" />
  


Исходный код:  
```python
import numpy as np

print("=== Задача 7: Собственные числа и векторы матрицы A ===")

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
    

# Дополнительная информация
print("\n=== ХАРАКТЕРИСТИЧЕСКИЙ МНОГОЧЛЕН ===")
coeffs = np.poly(A)
print(f"λ³ + ({coeffs[1]:.2f})λ² + ({coeffs[2]:.2f})λ + ({coeffs[3]:.2f}) = 0")
```
