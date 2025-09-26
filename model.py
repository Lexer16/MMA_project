import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_models_directory():
    """Создает директории для моделей и отчетов"""
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)

def load_preprocessed_data():
    """Загрузка предобработанных данных"""
    data_path = "./data/ufc_preprocessed.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"Файл {data_path} не найден!")
        logger.info("Сначала выполните preprocessing.py")
        raise FileNotFoundError(f"Файл {data_path} не найден!")
    
    logger.info(f"Загрузка данных из {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    return df

def analyze_class_balance(y):
    """Анализ дисбаланса классов"""
    target_distribution = y.value_counts().sort_index()
    total_samples = len(y)
    
    logger.info("=== АНАЛИЗ ДИСБАЛАНСА КЛАССОВ ===")
    for class_label, count in target_distribution.items():
        percentage = (count / total_samples) * 100
        logger.info(f"Класс {class_label} ({'BLUE' if class_label == 0 else 'RED'}): {count} samples ({percentage:.2f}%)")
    
    imbalance_ratio = target_distribution.max() / target_distribution.min()
    logger.info(f"Соотношение классов: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 1.5:
        logger.warning(f"Обнаружен дисбаланс классов: соотношение {imbalance_ratio:.2f}:1")
    
    return target_distribution, imbalance_ratio

def apply_sampling_strategy(X, y, strategy='smote', random_state=42):
    """Применение стратегии семплирования для балансировки классов"""
    
    target_distribution, imbalance_ratio = analyze_class_balance(y)
    
    # Если дисбаланс незначительный, не применяем семплирование
    if imbalance_ratio <= 1.2:
        logger.info("Дисбаланс незначительный, семплирование не применяется")
        return X, y, 'none'
    
    logger.info(f"Применяем стратегию семплирования: {strategy}")
    
    if strategy == 'smote':
        # SMOTE - создает синтетические примеры для миноритарного класса
        sampler = SMOTE(random_state=random_state, k_neighbors=min(5, target_distribution.min() - 1))
        method_name = 'SMOTE'
    elif strategy == 'oversample':
        # Случайное дублирование примеров миноритарного класса
        sampler = RandomOverSampler(random_state=random_state)
        method_name = 'Random Oversampling'
    elif strategy == 'undersample':
        # Случайное удаление примеров мажоритарного класса
        sampler = RandomUnderSampler(random_state=random_state)
        method_name = 'Random Undersampling'
    elif strategy == 'combined':
        # Комбинированный подход
        over = RandomOverSampler(sampling_strategy=0.5, random_state=random_state)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=random_state)
        steps = [('over', over), ('under', under)]
        sampler = ImbPipeline(steps=steps)
        method_name = 'Combined Sampling'
    else:
        logger.info("Семплирование не применяется")
        return X, y, 'none'
    
    try:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        new_distribution = y_resampled.value_counts().sort_index()
        
        logger.info("=== РЕЗУЛЬТАТ СЕМПЛИРОВАНИЯ ===")
        for class_label, count in new_distribution.items():
            percentage = (count / len(y_resampled)) * 100
            logger.info(f"Класс {class_label} ({'BLUE' if class_label == 0 else 'RED'}): {count} samples ({percentage:.2f}%)")
        
        new_imbalance_ratio = new_distribution.max() / new_distribution.min()
        logger.info(f"Новое соотношение: {new_imbalance_ratio:.2f}:1")
        logger.info(f"Размер данных до: {len(y)}, после: {len(y_resampled)}")
        
        return X_resampled, y_resampled, method_name
        
    except Exception as e:
        logger.error(f"Ошибка при применении {strategy}: {e}")
        logger.info("Возвращаем исходные данные")
        return X, y, 'none'

def validate_data(X, y):
    """Валидация данных перед обучением"""
    # Проверка на пропущенные значения
    if X.isnull().sum().sum() > 0:
        logger.warning(f"Обнаружено пропущенных значений: {X.isnull().sum().sum()}")
        # Заполняем пропуски медианами
        X = X.fillna(X.median())
    
    return X, y

def plot_feature_importance(model, feature_names, timestamp):
    """Визуализация важности признаков"""
    try:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(f'./plots/feature_importance_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("График важности признаков сохранен")
        return importance_df
    except Exception as e:
        logger.warning(f"Не удалось создать график важности признаков: {e}")
        return None

def plot_calibration_curve(y_true, y_proba, timestamp):
    """Калибровочная кривая"""
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Модель")
        plt.plot([0, 1], [0, 1], "--", label="Идеально калиброванная")
        plt.xlabel("Среднее предсказанное значение")
        plt.ylabel("Доля положительных классов")
        plt.title("Калибровочная кривая")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./plots/calibration_curve_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Калибровочная кривая сохранена")
    except Exception as e:
        logger.warning(f"Не удалось создать калибровочную кривую: {e}")

def plot_confusion_matrix(y_true, y_pred, timestamp):
    """Матрица ошибок"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Blue Win', 'Red Win'], 
                   yticklabels=['Blue Win', 'Red Win'])
        plt.title('Матрица ошибок')
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
        plt.tight_layout()
        plt.savefig(f'./plots/confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Матрица ошибок сохранена")
    except Exception as e:
        logger.warning(f"Не удалось создать матрицу ошибок: {e}")

def plot_class_distribution(y_train, y_test, y_resampled, sampling_method, timestamp):
    """Визуализация распределения классов до и после семплирования"""
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original distribution
        original_dist = y_train.value_counts().sort_index()
        axes[0].bar(['BLUE (0)', 'RED (1)'], original_dist.values, color=['blue', 'red'])
        axes[0].set_title('Оригинальное распределение\n(тренировочные данные)')
        axes[0].set_ylabel('Количество образцов')
        for i, v in enumerate(original_dist.values):
            axes[0].text(i, v, str(v), ha='center', va='bottom')
        
        # Test distribution
        test_dist = y_test.value_counts().sort_index()
        axes[1].bar(['BLUE (0)', 'RED (1)'], test_dist.values, color=['blue', 'red'])
        axes[1].set_title('Распределение в тестовых данных')
        for i, v in enumerate(test_dist.values):
            axes[1].text(i, v, str(v), ha='center', va='bottom')
        
        # Resampled distribution
        if sampling_method != 'none':
            resampled_dist = y_resampled.value_counts().sort_index()
            axes[2].bar(['BLUE (0)', 'RED (1)'], resampled_dist.values, color=['blue', 'red'])
            axes[2].set_title(f'После {sampling_method}')
            for i, v in enumerate(resampled_dist.values):
                axes[2].text(i, v, str(v), ha='center', va='bottom')
        else:
            axes[2].bar(['BLUE (0)', 'RED (1)'], original_dist.values, color=['blue', 'red'])
            axes[2].set_title('Семплирование не применялось')
            for i, v in enumerate(original_dist.values):
                axes[2].text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'./plots/class_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("График распределения классов сохранен")
    except Exception as e:
        logger.warning(f"Не удалось создать график распределения классов: {e}")

def train_model():
    """Основная функция обучения модели"""
    # Создаем директории
    create_models_directory()
    
    # Timestamp для версионирования
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # --------------------
        # 1. Загружаем подготовленные данные
        # --------------------
        df = load_preprocessed_data()
        
        # --------------------
        # 2. Разделяем X и y
        # --------------------
        if "target" not in df.columns:
            raise ValueError("Столбец 'target' не найден в данных!")
        
        X = df.drop(columns=["target"])
        y = df["target"]
        
        # --------------------
        # 3. Валидация данных
        # --------------------
        X, y = validate_data(X, y)
        
        # --------------------
        # 4. Трейн/тест сплит (стратифицированный)
        # --------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Разделение данных: train {X_train.shape}, test {X_test.shape}")
        
        # --------------------
        # 5. Балансировка классов
        # --------------------
        X_train_resampled, y_train_resampled, sampling_method = apply_sampling_strategy(
            X_train, y_train, strategy='smote'  # Можно изменить на 'oversample', 'undersample', 'combined'
        )
        
        # Визуализация распределения классов
        plot_class_distribution(y_train, y_test, y_train_resampled, sampling_method, timestamp)
        
        # --------------------
        # 6. Определяем модель и сетку гиперпараметров
        # --------------------
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=42,
            verbose=100,
            early_stopping_rounds=50,
            auto_class_weights='Balanced'  # Автоматические веса классов
        )

        # Сетка параметров
        param_grid = {
            'iterations': [500, 800],
            'learning_rate': [0.03, 0.05, 0.1],
            'depth': [6, 8],
            'l2_leaf_reg': [1, 3, 5],
        }

        logger.info("Начинаем подбор гиперпараметров...")
        
        # --------------------
        # 7. GridSearchCV
        # --------------------
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            verbose=2,
            n_jobs=-1,
            return_train_score=True
        )

        grid_search.fit(X_train_resampled, y_train_resampled)

        # --------------------
        # 8. Анализ результатов GridSearch
        # --------------------
        logger.info("=== РЕЗУЛЬТАТЫ GRIDSEARCH ===")
        logger.info(f"Лучшие параметры: {grid_search.best_params_}")
        logger.info(f"Лучшее CV AUC: {grid_search.best_score_:.4f}")
        
        # Результаты всех комбинаций
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_path = f"./reports/gridsearch_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Результаты GridSearch сохранены: {results_path}")

        # --------------------
        # 9. Оценка качества на тесте
        # --------------------
        best_model = grid_search.best_estimator_
        
        # Предсказания
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        logger.info("=== РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        
        # Детальная классификация
        class_report = classification_report(y_test, y_pred, output_dict=True)
        logger.info("\nClassification Report:")
        for class_name, metrics in class_report.items():
            if class_name in ['0', '1']:
                class_label = 'BLUE' if class_name == '0' else 'RED'
                logger.info(f"Class {class_label}: Precision={metrics['precision']:.3f}, "
                           f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

        # --------------------
        # 10. Визуализации
        # --------------------
        logger.info("Создание визуализаций...")
        
        # Важность признаков
        feature_importance_df = plot_feature_importance(best_model, X.columns.tolist(), timestamp)
        
        # Калибровочная кривая
        plot_calibration_curve(y_test, y_proba, timestamp)
        
        # Матрица ошибок
        plot_confusion_matrix(y_test, y_pred, timestamp)

        # --------------------
        # 11. Сохраняем модель и метаданные
        # --------------------
        model_path = f"./models/catboost_ufc_model_{timestamp}.pkl"
        joblib.dump(best_model, model_path)
        
        # Метаданные модели
        model_metadata = {
            'model_info': {
                'model_path': model_path,
                'model_type': 'CatBoostClassifier',
                'training_date': timestamp,
                'feature_names': X.columns.tolist(),
                'target_name': 'target',
                'n_features': X.shape[1],
                'best_params': grid_search.best_params_,
                'sampling_method': sampling_method,
                'auto_class_weights': True
            },
            'performance_metrics': {
                'best_cv_auc': float(grid_search.best_score_),
                'test_accuracy': float(accuracy),
                'test_roc_auc': float(roc_auc),
                'test_precision_0': float(class_report['0']['precision']),
                'test_recall_0': float(class_report['0']['recall']),
                'test_f1_0': float(class_report['0']['f1-score']),
                'test_precision_1': float(class_report['1']['precision']),
                'test_recall_1': float(class_report['1']['recall']),
                'test_f1_1': float(class_report['1']['f1-score'])
            },
            'data_info': {
                'original_shape': [int(X.shape[0]), int(X.shape[1])],
                'train_shape': [int(X_train.shape[0]), int(X_train.shape[1])],
                'train_resampled_shape': [int(X_train_resampled.shape[0]), int(X_train_resampled.shape[1])],
                'test_shape': [int(X_test.shape[0]), int(X_test.shape[1])],
                'original_target_distribution': y.value_counts().to_dict(),
                'train_target_distribution': y_train.value_counts().to_dict(),
                'train_resampled_distribution': y_train_resampled.value_counts().to_dict() if sampling_method != 'none' else y_train.value_counts().to_dict(),
                'test_target_distribution': y_test.value_counts().to_dict(),
                'imbalance_ratio': float(y_train.value_counts().max() / y_train.value_counts().min())
            },
            'feature_importance': feature_importance_df.to_dict('records') if feature_importance_df is not None else []
        }
        
        # Сохраняем метаданные
        metadata_path = f"./models/model_metadata_{timestamp}.json"
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(model_metadata, f, indent=2, ensure_ascii=False)
        
        # Создаем симлинк на лучшую модель
        latest_model_path = "./models/catboost_ufc_model_latest.pkl"
        if os.path.exists(latest_model_path):
            os.remove(latest_model_path)
        os.symlink(os.path.basename(model_path), latest_model_path)
        
        # --------------------
        # 12. Финальный отчет
        # --------------------
        logger.info("="*60)
        logger.info("✅ ОБУЧЕНИЕ МОДЕЛИ ЗАВЕРШЕНО!")
        logger.info("="*60)
        logger.info(f"📁 Модель сохранена: {model_path}")
        logger.info(f"📊 Метод балансировки: {sampling_method}")
        logger.info(f"📈 Метрики:")
        logger.info(f"   - CV AUC: {grid_search.best_score_:.4f}")
        logger.info(f"   - Test Accuracy: {accuracy:.4f}")
        logger.info(f"   - Test ROC AUC: {roc_auc:.4f}")
        logger.info(f"🔗 Ссылка на последнюю модель: {latest_model_path}")
        
        return model_metadata
        
    except Exception as e:
        logger.error(f"Ошибка во время обучения модели: {e}")
        raise

def main():
    """Основная функция"""
    try:
        metadata = train_model()
        
        # Выводим краткую сводку
        print("\n" + "="*50)
        print("🎯 КРАТКАЯ СВОДКА ОБУЧЕНИЯ:")
        print(f"📊 Метод балансировки: {metadata['model_info']['sampling_method']}")
        print(f"📈 Лучшее CV AUC: {metadata['performance_metrics']['best_cv_auc']:.4f}")
        print(f"🎯 Test Accuracy: {metadata['performance_metrics']['test_accuracy']:.4f}")
        print(f"👥 Исходное соотношение: {metadata['data_info']['imbalance_ratio']:.2f}:1")
        
        # Информация о классах
        original_dist = metadata['data_info']['original_target_distribution']
        blue_wins = original_dist.get(0, 0)
        red_wins = original_dist.get(1, 0)
        total = blue_wins + red_wins
        print(f"🔵 Побед BLUE: {blue_wins} ({blue_wins/total*100:.1f}%)")
        print(f"🔴 Побед RED: {red_wins} ({red_wins/total*100:.1f}%)")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Программа завершена с ошибкой: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())