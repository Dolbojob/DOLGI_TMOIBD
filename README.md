# BigDataCases
### ***Кейс 1***: Анализ успеваемости студентов
Набор данных: [скачать набор данных](cases/case1/Успеваемость студентов.csv)  

В кейсе анализируется успеваемость студентов из файла `Успеваемость студентов.csv`. Строятся три типа графиков:
- **Scatter plot**: Диаграмма рассеяния оценок по семестрам.
- **Bar plot**: Столбчатая диаграмма средних оценок по группам.
- **Box plot**: Коробчатая диаграмма распределения оценок по дисциплинам.

Графики:  
<img src="cases/case1/scatter_plot.png" width="600">
<img src="cases/case1/bar_plot.png" width="600">
<img src="cases/case1/box_plot.png" width="600">

Исходный код:  
```python
# file: task_01_complex_algebraic_simplified.py
import numpy as np
import matplotlib.pyplot as plt

print("=== Задача 1: Действия в алгебраической форме ===")
print("Вычислить: (2 - 2√3i) / (1 + i√3)")

# Определение комплексных чисел
numerator = complex(2, -2 * np.sqrt(3))  # 2 - 2√3i
denominator = complex(1, np.sqrt(3))     # 1 + i√3

# Прямое деление комплексных чисел
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
<img src="cases/case2/scatter_plot.png" width="600">
<img src="cases/case2/bar_plot.png" width="600">
<img src="cases/case2/box_plot.png" width="600">

Исходный код:  
```python
import pandas as pd
import matplotlib.pyplot as plt

def scatter_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = df['Группа'].unique()
    colors = plt.cm.tab10.colors[:len(groups)]

    for i, group in enumerate(groups):
        subset = df[df['Группа'] == group]
        ax.scatter(
            subset['Группа'],
            subset['Вопрос 1'],
            label=group,
            color=colors[i],
            alpha=0.8
        )

    ax.set_title('Зависимость Вопрос 1 от Группа')
    ax.set_xlabel('Группа')
    ax.set_ylabel('Вопрос 1')
    ax.legend()
    ax.grid(True)
    fig.savefig('scatter_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def bar_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_q1_by_group = df.groupby('Группа')['Вопрос 1'].mean()
    avg_q1_by_group.plot(kind='bar', ax=ax, color='skyblue')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    ax.set_title('Среднее Вопрос 1 по Группа')
    ax.set_xlabel('Группа')
    ax.set_ylabel('Среднее Вопрос 1')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    fig.savefig('bar_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def box_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='Вопрос 1', by='Группа', ax=ax, grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax.set_title('Распределение Вопрос 1 по Группа')
    fig.suptitle('')
    ax.set_xlabel('Группа')
    ax.set_ylabel('Вопрос 1')
    fig.savefig('box_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def main(show_plots = False):
    file_path = "Результаты анкетирования.csv"

    try:
        df = pd.read_csv(
            file_path,
            usecols=["ФИО", "Группа", "Вопрос 1", "Вопрос 2", "Вопрос 3"]
        )
    except ValueError as e:
        raise ValueError(f"Ошибка при чтении CSV: {e}. Проверьте названия столбцов.")

    df["Группа"] = pd.to_numeric(df["Группа"], errors='coerce')
    df["Вопрос 1"] = pd.to_numeric(df["Вопрос 1"], errors='coerce')
    df["Вопрос 2"] = pd.to_numeric(df["Вопрос 2"], errors='coerce')
    df["Вопрос 3"] = pd.to_numeric(df["Вопрос 3"], errors='coerce')
    df = df.dropna()

    print(f"Форма данных: {df.shape}")

    scatter_plot(df, show_plots)
    bar_plot(df, show_plots)
    box_plot(df, show_plots)

if __name__ == "__main__":
    main(show_plots = True)
```

### ***Кейс 3***: Анализ посещаемости занятий
Набор данных: [скачать набор данных](cases/case3/Посещяемость занятий.csv)  

В кейсе анализируется посещаемость студентов из файла `Посещаемость занятий.csv`. Строятся три типа графиков:
- **Scatter plot**: Диаграмма рассеяния посещаемости по группам.
- **Bar plot**: Столбчатая диаграмма средней посещаемоти по группам.
- **Box plot**: Коробчатая диаграмма распределения посещаемости по граппам.

Графики:  
<img src="cases/case3/scatter_plot.png" width="600">
<img src="cases/case3/bar_plot.png" width="600">
<img src="cases/case3/box_plot.png" width="600">

Исходный код:  
```python
import pandas as pd
import matplotlib.pyplot as plt

def scatter_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    groups = df['Группа'].unique()
    colors = plt.cm.tab10.colors[:len(groups)]

    for i, group in enumerate(groups):
        subset = df[df['Группа'] == group]
        ax.scatter(
            subset['Группа'],
            subset['Присутствие'],
            label=group,
            color=colors[i],
            alpha=0.8
        )

    ax.set_title('Зависимость Присутствие от Группа')
    ax.set_xlabel('Группа')
    ax.set_ylabel('Присутствие (0 - Нет, 1 - Да)')
    ax.legend()
    ax.grid(True)
    fig.savefig('scatter_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def bar_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_attendance_by_group = df.groupby('Группа')['Присутствие'].mean()
    avg_attendance_by_group.plot(kind='bar', ax=ax, color='skyblue')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    ax.set_title('Среднее Присутствие по Группа')
    ax.set_xlabel('Группа')
    ax.set_ylabel('Среднее Присутствие')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    fig.savefig('bar_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def box_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='Присутствие', by='Группа', ax=ax, grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax.set_title('Распределение Присутствие по Группа')
    fig.suptitle('')  # Убираем автоматический заголовок от pandas
    ax.set_xlabel('Группа')
    ax.set_ylabel('Присутствие (0 - Нет, 1 - Да)')
    fig.savefig('box_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def main(show_plots = False):
    file_path = "Посещаемость занятий.csv"

    try:
        df = pd.read_csv(
            file_path,
            usecols=["Дата", "Группа", "ФИО", "Присутствие"]
        )
    except ValueError as e:
        raise ValueError(f"Ошибка при чтении CSV: {e}. Проверьте названия столбцов.")

    df["Группа"] = pd.to_numeric(df["Группа"], errors='coerce')
    df["Присутствие"] = df["Присутствие"].map({"Да": 1, "Нет": 0})
    df = df.dropna()

    print(f"Форма данных: {df.shape}")

    scatter_plot(df, show_plots)
    bar_plot(df, show_plots)
    box_plot(df, show_plots)

if __name__ == "__main__":
    main(show_plots = True)
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
import pandas as pd
import matplotlib.pyplot as plt

def scatter_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    ids = df['ID'].unique()

    for i, id_val in enumerate(ids):
        subset = df[df['ID'] == id_val]
        ax.scatter(
            subset['ID'],
            subset['Correct_Count'],
            label=id_val,
            alpha=0.8
        )

    ax.set_title('Зависимость Correct_Count от ID')
    ax.set_xlabel('ID')
    ax.set_ylabel('Correct_Count')
    ax.legend()
    ax.grid(True)
    fig.savefig('scatter_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def bar_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_correct_by_id = df.groupby('ID')['Correct_Count'].mean()
    avg_correct_by_id.plot(kind='bar', ax=ax, color='skyblue')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    ax.set_title('Среднее Correct_Count по ID')
    ax.set_xlabel('ID')
    ax.set_ylabel('Среднее Correct_Count')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    fig.savefig('bar_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def box_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='Correct_Count', by='ID', ax=ax, grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax.set_title('Распределение Correct_Count по ID')
    fig.suptitle('')  # Убираем автоматический заголовок от pandas
    ax.set_xlabel('ID')
    ax.set_ylabel('Correct_Count')
    fig.savefig('box_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def main(show_plots = False):
    file_path = "Результаты тесирования.csv"

    try:
        df = pd.read_csv(
            file_path,
            usecols=["ФИО", "ID", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        )
    except ValueError as e:
        raise ValueError(f"Ошибка при чтении CSV: {e}. Проверьте названия столбцов.")

    # Получаем правильные ответы из последней строки
    correct_answers = df.iloc[-1, 2:]  # Столбцы с 1 по 10
    df = df.iloc[:-1]  # Убираем последнюю строку с правильными ответами

    # Преобразуем ответы в числа
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

    # Считаем количество правильных ответов
    df["Correct_Count"] = (df.iloc[:, 2:] == correct_answers.values).sum(axis=1)

    df["ID"] = pd.to_numeric(df["ID"], errors='coerce')
    df = df.dropna()

    print(f"Форма данных: {df.shape}")

    scatter_plot(df, show_plots)
    bar_plot(df, show_plots)
    box_plot(df, show_plots)

if __name__ == "__main__":
    main(show_plots = True)
```

### ***Кейс 5***: Анализ научной активности
Набор данных: [скачать набор данных](cases/case5/Научная активность.csv)  

В кейсе анализируется научная активность студентов из файла `Научная активность.csv`. Строятся три типа графиков:
- **Scatter plot**: Диаграмма рассеяния количества и типов публикаций по годам.
- **Bar plot**: Столбчатая диаграмма среднего года по типам публикаций.
- **Box plot**: Коробчатая диаграмма распределения годов по типам публикаций.

Графики:  
<img src="cases/case5/scatter_plot.png" width="600">
<img src="cases/case5/bar_plot.png" width="600">
<img src="cases/case5/box_plot.png" width="600">

Исходный код:  
```python
import pandas as pd
import matplotlib.pyplot as plt

def scatter_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    types = df['Тип'].unique()
    colors = plt.cm.tab10.colors[:len(types)]

    for i, typ in enumerate(types):
        subset = df[df['Тип'] == typ]
        ax.scatter(
            subset['Год'],
            subset['Publication_Count'],
            label=typ,
            color=colors[i],
            alpha=0.8
        )

    ax.set_title('Зависимость Publication_Count от Год по Тип')
    ax.set_xlabel('Год')
    ax.set_ylabel('Publication_Count')
    ax.legend()
    ax.grid(True)
    fig.savefig('scatter_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def bar_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_year_by_type = df.groupby('Тип')['Год'].mean()
    avg_year_by_type.plot(kind='bar', ax=ax, color='skyblue')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    ax.set_title('Среднее Год по Тип')
    ax.set_xlabel('Тип')
    ax.set_ylabel('Среднее Год')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    fig.savefig('bar_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def box_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='Год', by='Тип', ax=ax, grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax.set_title('Распределение Год по Тип')
    fig.suptitle('')  # Убираем автоматический заголовок от pandas
    ax.set_xlabel('Тип')
    ax.set_ylabel('Год')
    fig.savefig('box_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def main(show_plots = False):
    file_path = "Научная активность.csv"

    try:
        df = pd.read_csv(
            file_path,
            usecols=["Авторы", "Название", "Тип", "Год"]
        )
    except ValueError as e:
        raise ValueError(f"Ошибка при чтении CSV: {e}. Проверьте названия столбцов.")

    df["Год"] = pd.to_numeric(df["Год"], errors='coerce')
    df["Publication_Count"] = df.groupby('Авторы')['Авторы'].transform('count')
    df = df.dropna()

    print(f"Форма данных: {df.shape}")

    scatter_plot(df, show_plots)
    bar_plot(df, show_plots)
    box_plot(df, show_plots)

if __name__ == "__main__":
    main(show_plots = True)
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
import pandas as pd
import matplotlib.pyplot as plt

def scatter_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = df['Должность'].unique()
    colors = plt.cm.tab10.colors[:len(positions)]

    for i, position in enumerate(positions):
        subset = df[df['Должность'] == position]
        ax.scatter(
            subset['Специальность'],
            subset['Зарплата'],
            label=position,
            color=colors[i],
            alpha=0.8
        )

    ax.set_title('Зависимость Зарплата от Специальность по Должность')
    ax.set_xlabel('Специальность')
    ax.set_ylabel('Зарплата')
    ax.legend()
    ax.grid(True)
    fig.savefig('scatter_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def bar_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_salary_by_position = df.groupby('Должность')['Зарплата'].mean()
    avg_salary_by_position.plot(kind='bar', ax=ax, color='skyblue')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    ax.set_title('Среднее Зарплата по Должность')
    ax.set_xlabel('Должность')
    ax.set_ylabel('Среднее Зарплата')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    fig.savefig('bar_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def box_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='Зарплата', by='Должность', ax=ax, grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax.set_title('Распределение Зарплата по Должность')
    fig.suptitle('')  # Убираем автоматический заголовок от pandas
    ax.set_xlabel('Должность')
    ax.set_ylabel('Зарплата')
    fig.savefig('box_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def main(show_plots = False):
    file_path = "Трудоустройство.csv"

    try:
        df = pd.read_csv(
            file_path,
            usecols=["ФИО", "Специальность", "Место работы", "Должность", "Зарплата"]
        )
    except ValueError as e:
        raise ValueError(f"Ошибка при чтении CSV: {e}. Проверьте названия столбцов.")

    df["Специальность"] = pd.to_numeric(df["Специальность"], errors='coerce')
    df["Зарплата"] = pd.to_numeric(df["Зарплата"], errors='coerce')
    df = df.dropna()

    print(f"Форма данных: {df.shape}")

    scatter_plot(df, show_plots)
    bar_plot(df, show_plots)
    box_plot(df, show_plots)

if __name__ == "__main__":
    main(show_plots = True)
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
import pandas as pd
import matplotlib.pyplot as plt

def scatter_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    genders = df['Gender'].unique()
    colors = plt.cm.tab10.colors[:len(genders)]

    for i, gender in enumerate(genders):
        subset = df[df['Gender'] == gender]
        ax.scatter(
            subset['Age'],
            subset['Total Time Spent'],
            label=gender,
            color=colors[i],
            alpha=0.8
        )

    ax.set_title('Зависимость Total Time Spent от Age по Gender')
    ax.set_xlabel('Age')
    ax.set_ylabel('Total Time Spent')
    ax.legend()
    ax.grid(True)
    fig.savefig('scatter_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def bar_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    avg_time_by_demo = df.groupby('Demographics')['Total Time Spent'].mean()
    avg_time_by_demo.plot(kind='bar', ax=ax, color='skyblue')

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    ax.set_title('Среднее Total Time Spent по Demographics')
    ax.set_xlabel('Demographics')
    ax.set_ylabel('Среднее Total Time Spent')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y')
    fig.savefig('bar_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def box_plot(df, show_plots=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='Age', by='OS', ax=ax, grid=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    ax.set_title('Распределение Age по OS')
    fig.suptitle('') 
    ax.set_xlabel('OS')
    ax.set_ylabel('Age')
    fig.savefig('box_plot.png')
    if show_plots:
        plt.show()
    plt.close(fig)

def main(show_plots=False):
    file_path = "social_networks.csv"

    try:
        df = pd.read_csv(
            file_path,
            usecols=["UserID", "Age", "Gender", "Demographics", "Total Time Spent", "OS"]
        )
    except ValueError as e:
        raise ValueError(f"Ошибка при чтении CSV: {e}. Проверьте названия столбцов.")

    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Total Time Spent'] = pd.to_numeric(df['Total Time Spent'], errors='coerce')
    df = df.dropna()

    print(f"Форма данных: {df.shape}")

    scatter_plot(df, show_plots)
    bar_plot(df, show_plots)
    box_plot(df, show_plots)

if __name__ == "__main__":
    main(show_plots=True)
```
