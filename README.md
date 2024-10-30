# Project test task README


## Issues 

- Debug the uencoder, iencoder in preprocess_data

## Materials

### Dataset

[ML2 RecSys HA](https://github.com/esokolov/ml-course-hse/blob/master/2023-spring/homeworks-practice/homework-practice-13-recommendations/homework-practice-13-recommendations.ipynb)

The dataset will be retrieved from here, as well as some code since I have done the assignment myself

### MLOps 

[GirafeAI Course on MLOps](https://github.com/girafe-ai/mlops/tree/master)

I will probably check this course every now and then


## Solution 


### Metric

Для оценки качества рекомендаций мы будем использовать метрику $MAP@k$.

Tl;dr $MAP@k$ - усредненная по всем пользователям $AP@k$. $AP@k$ - средняя точность по предсказанию для каждого юзера

$$
MAP@k = \frac{1}{N} \sum_{u = 1}^N AP_u@k
$$
$$
AP_u@k = \frac{1}{\min(k, n_u)} \sum_{i=1}^k r_u(i) p_u@i
$$
$$p_u@k = \dfrac{1}{k}\sum_{j=1}^k r_u(j)$$


*   $N$ - количество пользователей.
*   $n_u$ - число релевантных треков пользователя $u$ на тестовом промежутке.
*   $r_u(i)$ - бинарная величина: относится ли трек на позиции $i$ к релевантным.

### Model

Сама модель представляет собой колаборативный фильтр. Это первый, базовый уровень рекомендации. Данный алгоритм наиболее быстро работает, так как по факту обучать там практически нечего (идея схожа с kNN). Алгоритм ищет наиболее близких соседей и выдает их треки, которых не было у объекта, которому и рекомендуем. Точность не сильно высока, но по $MAP@k$ она выше случайных рекомендаций треков и позволяет нагенерить много потенциальных рекомендаций, среди которых уже можно выбирать каким-то более сложным алгоритмом. 