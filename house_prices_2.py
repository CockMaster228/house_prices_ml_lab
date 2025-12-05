#%%
# 1. ВРЕМЕННОЕ РАЗДЕЛЕНИЕ ПО ГОДАМ
import numpy as np
import pandas as pd

# Загружаем данные
df = pd.read_csv('train.csv')

# Сортируем по году постройки
df = df.sort_values('YearBuilt')

# Разделяем по годам (например, 80% ранние годы - train, 20% поздние - test)
split_year = df['YearBuilt'].quantile(0.8)
train_df = df[df['YearBuilt'] < split_year]
test_df = df[df['YearBuilt'] >= split_year]

print(f"Разделение по году: train < {split_year}, test >= {split_year}")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
print(f"Train years: {train_df['YearBuilt'].min()} - {train_df['YearBuilt'].max()}")
print(f"Test years: {test_df['YearBuilt'].min()} - {test_df['YearBuilt'].max()}")

# Создаем целевую переменную
train_df = train_df.assign(log_SalePrice=np.log1p(train_df['SalePrice']))
test_df = test_df.assign(log_SalePrice=np.log1p(test_df['SalePrice']))

# Удаляем ненужные колонки
train_df = train_df.drop(['SalePrice', 'Id'], axis=1)
test_df = test_df.drop(['SalePrice', 'Id'], axis=1)

# Разделяем на X и y
X_train = train_df.drop('log_SalePrice', axis=1)
y_train = train_df['log_SalePrice']
X_test = test_df.drop('log_SalePrice', axis=1)
y_test = test_df['log_SalePrice']
#%%
y_train
#%%
# 2. ЗАПОЛНЕНИЕ ПРОПУСКОВ БЕЗ УТЕЧКИ (версия с временным разделением)
# Заполняем пропуски отдельно в train и test используя статистики только из train

# Определяем типы колонок
num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(include='object').columns

print(f"Числовых колонок: {len(num_cols)}, Категориальных: {len(cat_cols)}")

# Вычисляем статистики ТОЛЬКО на тренировочных данных (ранние годы)
num_medians = X_train[num_cols].median()
cat_modes = X_train[cat_cols].mode().iloc[0]

# Заполняем пропуски в train и test
X_train[num_cols] = X_train[num_cols].fillna(num_medians)
X_test[num_cols] = X_test[num_cols].fillna(num_medians)
X_train[cat_cols] = X_train[cat_cols].fillna(cat_modes)
X_test[cat_cols] = X_test[cat_cols].fillna(cat_modes)

# Проверяем что пропусков нет
print("Пропуски в train:", X_train.isna().sum().sum())
print("Пропуски в test:", X_test.isna().sum().sum())
#%%
# 3. TIME SERIES SPLIT ДЛЯ КРОСС-ВАЛИДАЦИИ
from sklearn.model_selection import TimeSeriesSplit

# Создаем сплиттер для временных рядов
ts_split = TimeSeriesSplit(n_splits=5)

# Для использования в GridSearch нужно отсортировать данные по времени
X_train_sorted = X_train.sort_values('YearBuilt')
y_train_sorted = y_train[X_train_sorted.index]
#%%
# 3. АНАЛИЗ КОРРЕЛЯЦИИ И УДАЛЕНИЕ ВЫСОКОКОРРЕЛИРОВАННЫХ ПРИЗНАКОВ
import seaborn as sns
import matplotlib.pyplot as plt

# Вычисляем корреляцию только для числовых признаков
num_cols = X_train.select_dtypes(include='number').columns
correlation_matrix = X_train[num_cols].corr()

# Визуализируем корреляционную матрицу
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False, fmt='.2f')
plt.title('Матрица корреляций числовых признаков')
plt.tight_layout()
plt.show()


# Находим высококоррелированные пары признаков
def find_high_correlations(df, threshold=0.85):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1))
    high_corr_pairs = []

    for column in upper_tri.columns:
        for idx, value in upper_tri[column].items():
            if value > threshold:
                high_corr_pairs.append((column, idx, value))

    return high_corr_pairs


high_corr = find_high_correlations(X_train[num_cols], threshold=0.85)
print("Высококоррелированные пары признаков (corr > 0.85):")
for pair in high_corr:
    print(f"{pair[0]} - {pair[1]}: {pair[2]:.3f}")

# Удаляем один признак из каждой высококоррелированной пары
columns_to_drop = set()
for col1, col2, corr in high_corr:
    # Сохраняем признак, который имеет большую корреляцию с целевой переменной
    corr_with_target1 = abs(X_train[col1].corr(y_train))
    corr_with_target2 = abs(X_train[col2].corr(y_train))

    if corr_with_target1 > corr_with_target2:
        columns_to_drop.add(col2)
    else:
        columns_to_drop.add(col1)

print(f"\nУдаляем признаки: {columns_to_drop}")

# Удаляем высококоррелированные признаки
X_train = X_train.drop(columns=columns_to_drop)
X_test = X_test.drop(columns=columns_to_drop)

print(f"После удаления высококоррелированных признаков:")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
#%%
# 4. VARIANCEThreshold ДЛЯ УДАЛЕНИЯ МАЛОИНФОРМАТИВНЫХ ПРИЗНАКОВ
from sklearn.feature_selection import VarianceThreshold

# Обновляем списки колонок после удаления высококоррелированных
num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(include='object').columns

print("Применяем VarianceThreshold к числовым признакам...")

# Применяем VarianceThreshold только к числовым признакам
selector = VarianceThreshold(threshold=0.01)  # Порог дисперсии 1%
X_train_num = X_train[num_cols]
X_test_num = X_test[num_cols]

selector.fit(X_train_num)
selected_num_cols = selector.get_feature_names_out()

print(f"Числовых признаков до: {len(num_cols)}")
print(f"Числовых признаков после: {len(selected_num_cols)}")

# Удаляем признаки с низкой дисперсией
low_variance_cols = set(num_cols) - set(selected_num_cols)
print(f"Удалены признаки с низкой дисперсией: {low_variance_cols}")

X_train = X_train.drop(columns=low_variance_cols)
X_test = X_test.drop(columns=low_variance_cols)

# Обновляем списки колонок
num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(include='object').columns

print(f"\nИтоговые размерности:")
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Числовых признаков: {len(num_cols)}, Категориальных: {len(cat_cols)}")
#%%
# 1. РАЗДЕЛЕНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ ПО ПОРОГУ 7
cat_cols = X_train.select_dtypes(include='object').columns

# Разделяем по количеству уникальных значений
low_cardinality_cats = [col for col in cat_cols if X_train[col].nunique() <= 7]
high_cardinality_cats = [col for col in cat_cols if X_train[col].nunique() > 7]

print(f"Категориальные признаки с <=7 уникальных значений (OneHot): {len(low_cardinality_cats)}")
print(f"Категориальные признаки с >7 уникальных значений (CatBoost): {len(high_cardinality_cats)}")

# Выводим примеры
print("\nLow cardinality (OneHotEncoder):")
for col in low_cardinality_cats:
    unique_vals = X_train[col].nunique()
    print(f"  {col}: {unique_vals} уникальных значений")

print("\nHigh cardinality (CatBoostEncoder):")
for col in high_cardinality_cats:
    unique_vals = X_train[col].nunique()
    print(f"  {col}: {unique_vals} уникальных значений")
#%%
# 2. ПАЙПЛАЙН С РАЗДЕЛЕННЫМИ ЭНКОДЕРАМИ
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import CatBoostEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat_low', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_cardinality_cats),
    ('cat_high', CatBoostEncoder(random_state=42), high_cardinality_cats)
])

# Базовый пайплайн для проверки
dt_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('dt', DecisionTreeRegressor(max_depth=10, random_state=42))
])

# Быстрая проверка
dt_pipe.fit(X_train, y_train)
train_score = dt_pipe.score(X_train, y_train)
test_score = dt_pipe.score(X_test, y_test)

print("Результаты с разделенными энкодерами:")
print(f"Train R²: {train_score:.4f}")
print(f"Test R²:  {test_score:.4f}")
print(f"Разница:  {train_score - test_score:.4f}")
#%%
# 4. RANDOMIZEDSEARCHCV С ОПТИМАЛЬНЫМ ПАЙПЛАЙНОМ
from sklearn.model_selection import RandomizedSearchCV

# Параметры для Decision Tree
param_dist = {
    'dt__max_depth': [5, 10, 15, 20, 25, 30, None],
    'dt__min_samples_split': [2, 5, 10, 20, 30, 40, 50],
    'dt__min_samples_leaf': [1, 2, 4, 6, 8, 10, 15, 20],
    'dt__max_features': [None, 'sqrt', 'log2', 0.3, 0.5, 0.7, 0.9],
    'dt__ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.05, 0.1],
    'dt__min_impurity_decrease': [0.0, 0.001, 0.005, 0.01, 0.05]
}

# RandomizedSearchCV с TimeSeriesSplit
dt_search = RandomizedSearchCV(
    dt_pipe,
    param_distributions=param_dist,
    n_iter=50,
    cv=ts_split,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Запускаем RandomizedSearchCV с оптимальными энкодерами...")
dt_search.fit(X_train_sorted, y_train_sorted)

print("\nЛучшие параметры:")
for param, value in dt_search.best_params_.items():
    print(f"{param}: {value}")

print(f"\nЛучший CV score: {dt_search.best_score_:.4f}")
#%%
# 5. ФИНАЛЬНАЯ ОЦЕНКА МОДЕЛИ
from sklearn.metrics import r2_score, mean_squared_error

# Лучшая модель
best_dt = dt_search.best_estimator_

# Предсказания
y_pred_train = best_dt.predict(X_train)
y_pred_test = best_dt.predict(X_test)

# Метрики
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
print(f"Train R²: {train_r2:.4f}")
print(f"Test R²:  {test_r2:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")
print(f"Разница Train/Test R²: {train_r2 - test_r2:.4f}")
#%%
# 3. ПОПРОБУЙ СЛУЧАЙНЫЙ ЛЕС - ОН ЛУЧШЕ ОБОБЩАЕТ
from sklearn.ensemble import RandomForestRegressor

rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipe.fit(X_train, y_train)
train_score = rf_pipe.score(X_train, y_train)
test_score = rf_pipe.score(X_test, y_test)

print("RANDOM FOREST (должен быть лучше):")
print(f"Train R²: {train_score:.4f}")
print(f"Test R²:  {test_score:.4f}")
print(f"Разница:  {train_score - test_score:.4f}")
#%%
param_dist = {
    'rf__n_estimators': [100, 200, 300, 500],           # Количество деревьев
    'rf__max_depth': [10, 15, 20, 25, None],           # Глубина
    'rf__min_samples_split': [2, 5, 10, 20],           # Минимум для разделения
    'rf__min_samples_leaf': [1, 2, 4, 6],              # Минимум в листе
    'rf__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7], # Доля признаков
    'rf__bootstrap': [True, False]                     # С заменой или без
}

rf_search = RandomizedSearchCV(
    rf_pipe,
    param_distributions=param_dist,
    n_iter=50,
    cv=ts_split,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("Запускаем RandomizedSearchCV с оптимальными энкодерами...")
rf_search.fit(X_train_sorted, y_train_sorted)

print("\nЛучшие параметры:")
for param, value in rf_search.best_params_.items():
    print(f"{param}: {value}")

print(f"\nЛучший CV score: {rf_search.best_score_:.4f}")
#%%
from sklearn.metrics import mean_squared_log_error

rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features=0.3,
        random_state=42,
        n_jobs=-1,
        bootstrap=False
    ))
])

rf_pipe.fit(X_train_sorted, y_train_sorted)
#%%
# Твои предсказания (они в log-шкале)
y_pred_train = rf_pipe.predict(X_train)
y_pred_test = rf_pipe.predict(X_test)

# ПРЕОБРАЗУЕМ обратно в исходную шкалу (доллары)
# потому что np.log1p(SalePrice) был твой таргет
true_prices_train = np.expm1(y_train)  # Исходные цены трейна
true_prices_test = np.expm1(y_test)  # Исходные цены теста
pred_prices_train = np.expm1(y_pred_train)  # Предсказания в долларах
pred_prices_test = np.expm1(y_pred_test)  # Предсказания в долларах

# Теперь вычисляем MSLE на исходных ценах
from sklearn.metrics import mean_squared_log_error

train_msle = mean_squared_log_error(true_prices_train, pred_prices_train)
test_msle = mean_squared_log_error(true_prices_test, pred_prices_test)

print(f"Train RMSLE: {np.sqrt(train_msle):.4f}")  # Root MSLE
print(f"Test RMSLE:  {np.sqrt(test_msle):.4f}")
#%%
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

pipe_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('Lasso', Lasso())
])

pipe_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('KNN', KNeighborsRegressor())
])

estimators = [
    ('decision_tree', rf_pipe),
    ('knn', pipe_knn),
    ('lasso', pipe_lr)
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=RandomForestRegressor(),
    cv=2
)

stacking.fit(X_train, y_train)
#%%
# Твои предсказания (они в log-шкале)
y_pred_train = stacking.predict(X_train)
y_pred_test = stacking.predict(X_test)

# ПРЕОБРАЗУЕМ обратно в исходную шкалу (доллары)
# потому что np.log1p(SalePrice) был твой таргет
true_prices_train = np.expm1(y_train)  # Исходные цены трейна
true_prices_test = np.expm1(y_test)  # Исходные цены теста
pred_prices_train = np.expm1(y_pred_train)  # Предсказания в долларах
pred_prices_test = np.expm1(y_pred_test)  # Предсказания в долларах

# Теперь вычисляем MSLE на исходных ценах
from sklearn.metrics import mean_squared_log_error

train_msle = mean_squared_log_error(true_prices_train, pred_prices_train)
test_msle = mean_squared_log_error(true_prices_test, pred_prices_test)

print(f"Train RMSLE: {np.sqrt(train_msle):.4f}")  # Root MSLE
print(f"Test RMSLE:  {np.sqrt(test_msle):.4f}")