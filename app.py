import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="UFC Fight Predictor",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("🥊 UFC Fight Predictor")
st.markdown("""
Прогнозирование вероятности победы бойцов на основе их статистики.
Введите параметры бойцов и получите прогноз!
""")

# Функции загрузки модели и конфигурации
@st.cache_resource
def load_model_and_metadata():
    """Загрузка обученной модели и метаданных"""
    try:
        # Пытаемся загрузить последнюю модель через симлинк
        model_path = "./models/catboost_ufc_model_latest.pkl"
        
        if not os.path.exists(model_path):
            # Если симлинка нет, ищем самую свежую модель
            model_files = [f for f in os.listdir("./models") 
                          if f.startswith("catboost_ufc_model_") and f.endswith(".pkl")
                          and "latest" not in f]
            
            if not model_files:
                st.error("❌ Модели не найдены! Сначала обучите модель.")
                return None, None
            
            # Берем самую свежую модель
            latest_model = sorted(model_files)[-1]
            model_path = f"./models/{latest_model}"
        
        model = joblib.load(model_path)
        
        # Загружаем метаданные
        model_name = os.path.basename(model_path).replace(".pkl", "")
        metadata_path = f"./models/{model_name.replace('catboost_ufc_model', 'model_metadata')}.json"
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = None
            
        st.success(f"✅ Модель загружена: {os.path.basename(model_path)}")
        return model, metadata
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None, None

@st.cache_resource
def load_feature_config():
    """Загрузка конфигурации признаков"""
    try:
        with open("./models/feature_config.json", "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ Конфигурация признаков не найдена!")
        return None

@st.cache_resource
def load_encoder():
    """Загрузка энкодера"""
    try:
        encoder = joblib.load("./models/onehot_encoder.pkl")
        return encoder
    except FileNotFoundError:
        return None

def get_expected_features(metadata, feature_config):
    """Получаем список ожидаемых признаков в правильном порядке"""
    if metadata and 'model_info' in metadata:
        return metadata['model_info']['feature_names']
    elif feature_config:
        return feature_config['feature_names']
    else:
        return None

def prepare_fight_data(red_fighter, blue_fighter, encoder, expected_features):
    """Подготовка данных для предсказания"""
    
    # Создаем словарь со всеми признаками
    fight_dict = {}
    
    # Заполняем числовые признаки RED
    red_features = {k: v for k, v in red_fighter.items() if k.startswith('R_')}
    fight_dict.update(red_features)
    
    # Заполняем числовые признаки BLUE
    blue_features = {k: v for k, v in blue_fighter.items() if k.startswith('B_')}
    fight_dict.update(blue_features)
    
    # Добавляем временные признаки (берём текущие значения)
    fight_dict['year'] = datetime.now().year
    fight_dict['month'] = datetime.now().month
    
    # Создаем DataFrame
    fight_data = pd.DataFrame([fight_dict])
    
    # Кодируем категориальные признаки
    if encoder is not None:
        categorical_data = pd.DataFrame([{
            'weight_class': red_fighter['weight_class'],
            'R_Stance': red_fighter['R_Stance'],
            'B_Stance': blue_fighter['B_Stance']
        }])
        
        encoded_data = encoder.transform(categorical_data)
        encoded_df = pd.DataFrame(
            encoded_data, 
            columns=encoder.get_feature_names_out(['weight_class', 'R_Stance', 'B_Stance'])
        )
        
        # Объединяем с числовыми признаками
        fight_data = pd.concat([fight_data.reset_index(drop=True), 
                              encoded_df.reset_index(drop=True)], axis=1)
    
    # Создаем финальный DataFrame с правильным порядком колонок
    final_data = pd.DataFrame(columns=expected_features)
    
    # Заполняем существующие колонки
    for col in expected_features:
        if col in fight_data.columns:
            final_data[col] = fight_data[col]
        else:
            final_data[col] = 0  # Заполняем нулями отсутствующие колонки
    
    return final_data.fillna(0)

def display_model_info(metadata):
    """Отображение информации о модели с учетом балансировки"""
    if metadata:
        with st.expander("ℹ️ Информация о модели"):
            # Информация о балансировке
            model_info = metadata.get('model_info', {})
            sampling_method = model_info.get('sampling_method', 'none')
            auto_weights = model_info.get('auto_class_weights', False)
            
            st.write("**Балансировка классов:**")
            if sampling_method != 'none':
                st.write(f"📊 Метод: {sampling_method}")
            if auto_weights:
                st.write("⚖️ Автовеса: Да")
            
            # Распределение данных
            data_info = metadata.get('data_info', {})
            st.write("**Данные обучения:**")
            
            original_dist = data_info.get('original_target_distribution', {})
            blue_original = original_dist.get('0', 0)
            red_original = original_dist.get('1', 0)
            total_original = blue_original + red_original
            
            if total_original > 0:
                st.write(f"🔵 Исходные победы BLUE: {blue_original} ({blue_original/total_original*100:.1f}%)")
                st.write(f"🔴 Исходные победы RED: {red_original} ({red_original/total_original*100:.1f}%)")
            
            # После балансировки
            if sampling_method != 'none':
                resampled_dist = data_info.get('train_resampled_distribution', {})
                blue_resampled = resampled_dist.get('0', 0)
                red_resampled = resampled_dist.get('1', 0)
                total_resampled = blue_resampled + red_resampled
                
                if total_resampled > 0:
                    st.write(f"🔵 После балансировки BLUE: {blue_resampled} ({blue_resampled/total_resampled*100:.1f}%)")
                    st.write(f"🔴 После балансировки RED: {red_resampled} ({red_resampled/total_resampled*100:.1f}%)")
            
            # Качество модели
            st.write("**Качество модели:**")
            metrics = metadata.get('performance_metrics', {})
            st.write(f"📈 Точность: {metrics.get('test_accuracy', 'N/A'):.1%}")
            st.write(f"🎯 ROC AUC: {metrics.get('test_roc_auc', 'N/A'):.3f}")

# Загружаем ресурсы при старте
model, metadata = load_model_and_metadata()
feature_config = load_feature_config()
encoder = load_encoder()

# Получаем ожидаемые признаки
expected_features = get_expected_features(metadata, feature_config)

if expected_features:
    st.sidebar.success(f"✅ Модель готова к работе")
else:
    st.sidebar.error("❌ Не удалось загрузить конфигурацию признаков")

# Отображаем информацию о модели
if metadata:
    display_model_info(metadata)

# Интерфейс ввода данных
st.sidebar.header("📊 Параметры боя")

# Весовая категория
weight_classes = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight", 
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight"
]

weight_class = st.sidebar.selectbox("Весовая категория", weight_classes, index=4)

# Стили стоек
stances = ["Orthodox", "Southpaw", "Switch"]

# Создаем две колонки для бойцов
col1, col2 = st.columns(2)

with col1:
    st.header("🔴 Красный угол")
    st.subheader("Боец RED")
    
    red_fighter = {
        'weight_class': weight_class,
        'R_Stance': st.selectbox("Стойка RED", stances, key="red_stance", index=0),
        'R_Height_cms': st.slider("Рост RED (см)", 150, 220, 175, key="red_height"),
        'R_Reach_cms': st.slider("Размах рук RED (см)", 150, 220, 178, key="red_reach"),
        'R_Weight_lbs': st.slider("Вес RED (фунты)", 100, 300, 170, key="red_weight"),
        'R_age': st.slider("Возраст RED", 18, 50, 25, key="red_age"),
        'R_win_streak': st.slider("Серия побед RED", 0, 10, 1, key="red_streak"),
        'R_wins': st.slider("Всего побед RED", 0, 50, 5, key="red_wins"),
        'R_losses': st.slider("Всего поражений RED", 0, 30, 3, key="red_losses"),
        'R_avg_SIG_STR_landed': st.slider("Среднее число точных ударов RED", 0.0, 10.0, 2.5, 0.1, key="red_strikes"),
        'R_avg_SUB_ATT': st.slider("Среднее число попыток сабмишена RED", 0.0, 10.0, 0.5, 0.1, key="red_subs"),
        'R_avg_TD_landed': st.slider("Среднее число тейкдаунов RED", 0.0, 10.0, 1.2, 0.1, key="red_takedowns")
    }

with col2:
    st.header("🔵 Синий угол") 
    st.subheader("Боец BLUE")
    
    blue_fighter = {
        'weight_class': weight_class,
        'B_Stance': st.selectbox("Стойка BLUE", stances, key="blue_stance", index=0),
        'B_Height_cms': st.slider("Рост BLUE (см)", 150, 220, 180, key="blue_height"),
        'B_Reach_cms': st.slider("Размах рук BLUE (см)", 150, 220, 185, key="blue_reach"),
        'B_Weight_lbs': st.slider("Вес BLUE (фунты)", 100, 300, 170, key="blue_weight"),
        'B_age': st.slider("Возраст BLUE", 18, 50, 28, key="blue_age"),
        'B_win_streak': st.slider("Серия побед BLUE", 0, 10, 5, key="blue_streak"),
        'B_wins': st.slider("Всего побед BLUE", 0, 50, 15, key="blue_wins"),
        'B_losses': st.slider("Всего поражений BLUE", 0, 30, 2, key="blue_losses"),
        'B_avg_SIG_STR_landed': st.slider("Среднее число точных ударов BLUE", 0.0, 10.0, 4.2, 0.1, key="blue_strikes"),
        'B_avg_SUB_ATT': st.slider("Среднее число попыток сабмишена BLUE", 0.0, 10.0, 1.8, 0.1, key="blue_subs"),
        'B_avg_TD_landed': st.slider("Среднее число тейкдаунов BLUE", 0.0, 10.0, 3.1, 0.1, key="blue_takedowns")
    }

# Кнопка предсказания
if st.button("🎯 Сделать прогноз", type="primary", use_container_width=True):
    
    if model is None or expected_features is None:
        st.error("❌ Модель или конфигурация признаков не загружены!")
    else:
        with st.spinner("🤖 Анализируем данные бойцов..."):
            try:
                # Подготавливаем данные
                fight_data = prepare_fight_data(red_fighter, blue_fighter, encoder, expected_features)
                
                # Предсказание
                probability = model.predict_proba(fight_data)[0, 1]
                red_win_prob = probability * 100
                blue_win_prob = (1 - probability) * 100
                
                error = None
                
            except Exception as e:
                red_win_prob, blue_win_prob, error = None, None, str(e)
        
        if error:
            st.error(f"❌ Ошибка предсказания: {error}")
        else:
            st.success("✅ Прогноз готов!")
            
            # Визуализация результатов
            st.markdown("---")
            st.subheader("📊 Результаты прогноза")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🔴 RED: {red_win_prob:.1f}%")
                st.progress(red_win_prob/100)
                
            with col2:
                st.markdown(f"### 🔵 BLUE: {blue_win_prob:.1f}%")
                st.progress(blue_win_prob/100)
            
            # Вердикт
            difference = abs(red_win_prob - blue_win_prob)
            
            if difference < 10:
                st.info(f"⚖️ **Близкий бой!** (разница: {difference:.1f}%)")
                st.write("Этот бой может закончиться любым исходом - всё решится в октагоне!")
            elif red_win_prob > blue_win_prob:
                st.success(f"🏆 **Победа RED!** (преимущество: {difference:.1f}%)")
                st.write("Красный угол имеет значительное преимущество по статистике.")
            else:
                st.success(f"🏆 **Победа BLUE!** (преимущество: {difference:.1f}%)")
                st.write("Синий угол демонстрирует лучшие показатели.")

# Боковая панель с вспомогательной информацией
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ О приложении")

st.sidebar.info("""
**UFC Fight Predictor** использует сбалансированную модель машинного обучения для объективного прогнозирования исходов боев.

**Особенности модели:**
- Балансировка классов для объективных прогнозов
- Учет антропометрических данных
- Анализ статистики предыдущих боев
- Учет стиля боя и текущей формы
""")

st.sidebar.markdown("---")
st.sidebar.header("📋 Советы по вводу данных")

st.sidebar.info("""
**Для точных прогнозов:**

• Указывайте реальные антропометрические данные
• Используйте актуальную статистику бойцов  
• Учитывайте текущую форму (серию побед/поражений)
• Сравнивайте стили боя (стойка, предпочтения)
""")

st.sidebar.markdown("---")
st.sidebar.header("🎯 О балансировке модели")

st.sidebar.info("""
**Сбалансированная модель учитывает:**

• Историческое распределение побед
• Объективную оценку шансов обоих бойцов
• Реальную конкурентность боев

*Модель обучена на сбалансированных данных для минимизации смещения прогнозов.*
""")

# Запуск приложения
if __name__ == "__main__":
    pass