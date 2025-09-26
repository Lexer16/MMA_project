import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import logging
import os
import joblib
import json
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def safe_fillna_grouped(df, group_col, fill_cols):
    """Безопасное заполнение пропусков по группам"""
    for col in fill_cols:
        # Если вся группа NaN, используем общую медиану
        df[col] = df.groupby(group_col)[col].transform(
            lambda x: x.fillna(x.median() if not x.isnull().all() else df[col].median())
        )
    return df

def data_quality_report(df):
    """Генерация отчета о качестве данных"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    return report

def preprocess_ufc_data(input_path, output_dir="./data"):
    """Основная функция предобработки данных UFC"""
    
    # Проверка существования файла
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл {input_path} не найден!")
    
    # Загрузка данных
    logger.info(f"Загрузка данных из {input_path}")
    try:
        df = pd.read_csv(input_path, sep=",", encoding="utf-8")
    except Exception as e:
        logger.error(f"Ошибка загрузки файла: {e}")
        raise
    
    data = df.copy()
    logger.info(f"Загружено {len(data)} строк, {len(data.columns)} столбцов")
    
    # --------------------
    # 1. Целевая переменная (Winner -> 1 = R, 0 = B)
    # --------------------
    initial_rows = len(data)
    data = data[data["Winner"].isin(["Red", "Blue"])]  # убираем ничьи
    removed_draws = initial_rows - len(data)
    logger.info(f"Удалено ничьих: {removed_draws}")
    
    data["target"] = (data["Winner"] == "Red").astype(int)
    logger.info(f"Целевая переменная создана. Распределение: {data['target'].value_counts().to_dict()}")
    
    # --------------------
    # 2. Извлекаем дату (год, месяц)
    # --------------------
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    null_dates = data["date"].isnull().sum()
    
    if null_dates > 0:
        logger.warning(f"Обнаружено {null_dates} строк с некорректной датой")
        # Заполняем минимальной датой из датасета
        data["date"] = data["date"].fillna(data["date"].min())
    
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    logger.info(f"Временные признаки добавлены: год {data['year'].min()}-{data['year'].max()}")
    
    # --------------------
    # 3. Обработка категориальных признаков
    # --------------------
    categorical_cols = ["weight_class", "R_Stance", "B_Stance"]
    
    # Проверяем наличие категориальных признаков
    missing_categorical = [col for col in categorical_cols if col not in data.columns]
    if missing_categorical:
        logger.warning(f"Отсутствуют категориальные признаки: {missing_categorical}")
        categorical_cols = [col for col in categorical_cols if col in data.columns]
    
    # Колонки для удаления
    drop_cols = ["R_fighter", "B_fighter", "Referee", "location", "Winner", "date"]
    drop_cols = [col for col in drop_cols if col in data.columns]  # Только существующие колонки
    
    # --------------------
    # 4. Заполнение пропусков
    # --------------------
    # Рост, вес, размах — медианой по весовой категории
    fill_cols = ["R_Height_cms", "R_Reach_cms", "R_Weight_lbs",
                "B_Height_cms", "B_Reach_cms", "B_Weight_lbs"]
    
    # Фильтруем только существующие колонки
    fill_cols = [col for col in fill_cols if col in data.columns]
    
    if fill_cols and "weight_class" in data.columns:
        logger.info("Заполнение пропусков в антропометрических данных...")
        data = safe_fillna_grouped(data, "weight_class", fill_cols)
    
    # Остальные числовые NaN заменим на 0
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["target"] and c in data.columns]  # исключаем target
    
    if num_cols:
        initial_nulls = data[num_cols].isnull().sum().sum()
        data[num_cols] = data[num_cols].fillna(0)
        logger.info(f"Заполнено {initial_nulls} пропусков в числовых признаках")
    
    # --------------------
    # 5. OneHot для категориальных признаков
    # --------------------
    if categorical_cols:
        logger.info(f"OneHot encoding для: {categorical_cols}")
        
        # Проверяем наличие данных в категориальных колонках
        for col in categorical_cols:
            null_count = data[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Заполняем {null_count} пропусков в {col} модой")
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else "Unknown")
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(data[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded, 
            columns=encoder.get_feature_names_out(categorical_cols), 
            index=data.index
        )
    else:
        logger.warning("Нет категориальных признаков для кодирования")
        encoded_df = pd.DataFrame()
        encoder = None
    
    # --------------------
    # 6. Объединяем данные
    # --------------------
    columns_to_drop = [col for col in drop_cols + categorical_cols if col in data.columns]
    final_df = pd.concat([data.drop(columns=columns_to_drop), encoded_df], axis=1)
    
    # Удаляем возможные дубликаты колонок
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    
    # --------------------
    # 7. Сохраняем подготовленный датасет
    # --------------------
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    output_path = f"{output_dir}/ufc_preprocessed.csv"
    final_df.to_csv(output_path, index=False)
    
    # --------------------
    # 8. Сохраняем энкодер и конфигурацию для Streamlit
    # --------------------
    if encoder is not None:
        encoder_path = "./models/onehot_encoder.pkl"
        joblib.dump(encoder, encoder_path)
        logger.info(f"✅ Энкодер сохранен: {encoder_path}")
    
    # Сохраняем конфигурацию признаков
    feature_config = {
        'feature_names': final_df.drop(columns=['target']).columns.tolist(),
        'categorical_cols': categorical_cols,
        'numeric_cols': [col for col in final_df.columns if col not in categorical_cols + ['target'] and col in final_df.columns],
        'original_columns': df.columns.tolist(),
        'preprocessing_date': datetime.now().isoformat(),
        'final_shape': final_df.shape
    }
    
    config_path = "./models/feature_config.json"
    with open(config_path, "w", encoding='utf-8') as f:
        json.dump(feature_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Конфигурация признаков сохранена: {config_path}")
    
    # --------------------
    # 9. Финальный отчет
    # --------------------
    quality_report = data_quality_report(final_df)
    logger.info("=== ОТЧЕТ О КАЧЕСТВЕ ДАННЫХ ===")
    for key, value in quality_report.items():
        logger.info(f"{key}: {value}")
    
    logger.info(f"✅ Предобработка завершена!")
    logger.info(f"✅ Исходные данные: {df.shape} -> Финальные данные: {final_df.shape}")
    logger.info(f"✅ Файл сохранен: {output_path}")
    
    return final_df

def main():
    """Основная функция"""
    try:
        input_file = "./data/ufc-master.csv"
        
        # Проверяем существование входного файла
        if not os.path.exists(input_file):
            logger.error(f"Входной файл не найден: {input_file}")
            logger.info("Убедитесь, что файл ufc-master.csv находится в папке ./data/")
            return
        
        # Запускаем предобработку
        final_data = preprocess_ufc_data(input_file)
        
        # Выводим основную информацию о данных
        logger.info("\n" + "="*50)
        logger.info("КРАТКАЯ СВОДКА ДАННЫХ:")
        logger.info(f"Всего признаков: {len(final_data.columns)}")
        logger.info(f"Числовых признаков: {len(final_data.select_dtypes(include=[np.number]).columns)}")
        logger.info(f"Целевая переменная: {final_data['target'].value_counts().to_dict()}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Ошибка во время выполнения: {e}")
        raise

if __name__ == "__main__":
    main()