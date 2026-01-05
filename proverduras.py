import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc, balanced_accuracy_score
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from collections import Counter
import pickle
import time
import io
import os
import random
import gc

st.set_page_config(
    page_title="🥬Aprendizaje Supervisado",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 15px;
        border: none;
        transition: all 0.3s;
        font-size: 16px;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'training_data' not in st.session_state:
    st.session_state.training_data = []
if 'labels' not in st.session_state:
    st.session_state.labels = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'captured_images' not in st.session_state:
    st.session_state.captured_images = []
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

def augment_image(image, num_augmentations=5):
    """Genera versiones aumentadas de una imagen"""
    augmented = []
    
    # Convertir a PIL si es numpy array
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image
    
    for _ in range(num_augmentations):
        img = pil_img.copy()
        
        
        if random.random() > 0.3:
            angle = random.randint(-20, 20)
            img = img.rotate(angle, fillcolor=(255, 255, 255))
        
        # 2. Voltear horizontalmente
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 3. Ajustar brillo
        if random.random() > 0.3:
            enhancer = ImageEnhance.Brightness(img)
            factor = random.uniform(0.7, 1.3)
            img = enhancer.enhance(factor)
        
        # 4. Ajustar contraste
        if random.random() > 0.3:
            enhancer = ImageEnhance.Contrast(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        
        # 5. Ajustar saturación
        if random.random() > 0.3:
            enhancer = ImageEnhance.Color(img)
            factor = random.uniform(0.8, 1.2)
            img = enhancer.enhance(factor)
        
        # 6. Zoom aleatorio
        if random.random() > 0.5:
            w, h = img.size
            zoom = random.uniform(0.9, 1.1)
            new_w, new_h = int(w * zoom), int(h * zoom)
            img = img.resize((new_w, new_h))
            
            # Recortar al tamaño original
            if zoom > 1:
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                img = img.crop((left, top, left + w, top + h))
            else:
                # Rellenar si es más pequeño
                new_img = Image.new('RGB', (w, h), (255, 255, 255))
                paste_x = (w - new_w) // 2
                paste_y = (h - new_h) // 2
                new_img.paste(img, (paste_x, paste_y))
                img = new_img
        
        augmented.append(np.array(img))
    
    return augmented

def apply_data_augmentation(images, labels, multiplier=5):
    """Aplica data augmentation a todo el dataset"""
    augmented_images = []
    augmented_labels = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(images)
    
    for idx, (img, label) in enumerate(zip(images, labels)):
        # Agregar imagen original
        augmented_images.append(img)
        augmented_labels.append(label)
        
        # Generar versiones aumentadas
        aug_imgs = augment_image(img, num_augmentations=multiplier)
        augmented_images.extend(aug_imgs)
        augmented_labels.extend([label] * len(aug_imgs))
        
        # Actualizar progreso
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Aumentando datos: {idx+1}/{total} imágenes procesadas")
    
    progress_bar.empty()
    status_text.empty()
    
    return augmented_images, augmented_labels

# Clases de verduras
VEGGIE_CLASSES = ['Tomate', 'Zanahoria', 'Lechuga', 'Brócoli', 'Cebolla', 'Pimiento', 'Papa', 'Pepino', 'Calabaza', 'Espinaca']

def extract_features(image):
    """Extrae características MEJORADAS de la imagen para el modelo"""
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image
    else:
        img_rgb = np.array(image)
        if len(img_rgb.shape) == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    
    if len(img_rgb.shape) == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    features = []
    
    # 1. Estadísticas de color por canal (RGB)
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
        for i in range(3):
            channel = img_resized[:, :, i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel),
                np.min(channel),
                np.max(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75)
            ])
    else:
        for i in range(3):
            features.extend([
                np.mean(img_resized),
                np.std(img_resized),
                np.median(img_resized),
                np.min(img_resized),
                np.max(img_resized),
                np.percentile(img_resized, 25),
                np.percentile(img_resized, 75)
            ])
    
    # 2. Histograma de color (más detallado)
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
        for i in range(3):
            hist = cv2.calcHist([img_resized], [i], None, [8], [0, 256])
            features.extend(hist.flatten())
    else:
        hist = cv2.calcHist([img_resized], [0], None, [8], [0, 256])
        features.extend(hist.flatten())
        features.extend(hist.flatten())
        features.extend(hist.flatten())
    
    # 3. Conversión a HSV para mejor distinción de colores
    try:
        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        for i in range(3):
            channel = img_hsv[:, :, i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.median(channel)
            ])
    except:
        for i in range(3):
            features.extend([0, 0, 0])
    
    # 4. Textura (desviación estándar local)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) if len(img_resized.shape) == 3 else img_resized
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.median(gray),
        np.var(gray)
    ])
    
    # 5. Bordes (útil para detectar formas)
    try:
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.sum(edges) / edges.size,
            np.mean(edges),
            np.std(edges)
        ])
    except:
        features.extend([0, 0, 0])
    
    return np.array(features)

# FUNCIÓN NUEVA: CALCULAR PESOS DE CLASE
def calculate_class_weights(y):
    """
    Calcula pesos de clase según fórmula del PDF: w_c = N / (C * N_c)
    Donde:
    - N = número total de muestras
    - C = número de clases
    - N_c = número de muestras en la clase c
    """
    classes, counts = np.unique(y, return_counts=True)
    N = len(y)
    C = len(classes)
    
    weights = {}
    for cls, count in zip(classes, counts):
        weights[cls] = N / (C * count)
    
    return weights

#ENTRENAMIENTO CON BALANCEO 
def train_model(X, y, balancing_method="SMOTE"):
    """
    Entrena el modelo de clasificación MEJORADO con BALANCEO DE CLASES
    
    Parámetros:
    - balancing_method: "SMOTE", "SMOTETomek", "ClassWeights", "Undersampling", "None"
    """
    from sklearn.preprocessing import StandardScaler
    
    # ✅ PASO 1: Normalizar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ✅ PASO 2: Mostrar distribución ANTES del balanceo
    st.write("### 📊 Distribución de Clases ANTES del Balanceo:")
    counter_before = Counter(y)
    df_before = pd.DataFrame({
        'Clase': list(counter_before.keys()),
        'Cantidad': list(counter_before.values())
    })
    st.dataframe(df_before, hide_index=True)
    
    # Calcular Imbalance Ratio (IR)
    counts = list(counter_before.values())
    IR = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    
    col_ir1, col_ir2 = st.columns(2)
    with col_ir1:
        st.metric("📉 Imbalance Ratio (IR)", f"{IR:.2f}")
    with col_ir2:
        if IR > 10:
            st.error("⚠️ Desbalance SEVERO detectado")
        elif IR > 3:
            st.warning("⚠️ Desbalance MODERADO detectado")
        else:
            st.success("✅ Desbalance LEVE")
    
    # PASO 3: APLICAR TÉCNICA DE BALANCEO SELECCIONADA
    X_balanced = X_scaled
    y_balanced = y
    
    if balancing_method == "SMOTE":
        st.info("🔄 Aplicando SMOTE (Synthetic Minority Over-sampling Technique)")
        st.write("📝 Genera ejemplos sintéticos de la clase minoritaria mediante interpolación")
        try:
            # Verificar si hay suficientes muestras para SMOTE
            min_samples = min(counts)
            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
            
            if k_neighbors >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
                st.success("✅ SMOTE aplicado correctamente")
            else:
                st.warning("⚠️ SMOTE requiere al menos 2 muestras por clase. Usando Class Weights.")
                balancing_method = "ClassWeights"
        except Exception as e:
            st.warning(f"⚠️ SMOTE no pudo aplicarse: {e}. Usando Class Weights.")
            balancing_method = "ClassWeights"
    
    elif balancing_method == "SMOTETomek":
        st.info("🔄 Aplicando SMOTE + Tomek Links")
        st.write("📝 SMOTE + limpieza de ejemplos ambiguos en la frontera de decisión")
        try:
            smote_tomek = SMOTETomek(random_state=42)
            X_balanced, y_balanced = smote_tomek.fit_resample(X_scaled, y)
            st.success("✅ SMOTE-Tomek aplicado correctamente")
        except Exception as e:
            st.warning(f"⚠️ SMOTE-Tomek falló: {e}. Usando SMOTE simple.")
            try:
                smote = SMOTE(random_state=42)
                X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
            except:
                balancing_method = "ClassWeights"
    
    elif balancing_method == "Undersampling":
        st.info("🔄 Aplicando Random Undersampling")
        st.write("📝 Reduce aleatoriamente ejemplos de la clase mayoritaria")
        rus = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = rus.fit_resample(X_scaled, y)
        st.success("✅ Undersampling aplicado correctamente")
    
    # Mostrar distribución DESPUÉS del balanceo (si se aplicó SMOTE/Undersampling)
    if balancing_method in ["SMOTE", "SMOTETomek", "Undersampling"]:
        st.write("### 📊 Distribución de Clases DESPUÉS del Balanceo:")
        counter_after = Counter(y_balanced)
        df_after = pd.DataFrame({
            'Clase': list(counter_after.keys()),
            'Cantidad': list(counter_after.values())
        })
        st.dataframe(df_after, hide_index=True)
        
        # Nuevo IR
        counts_after = list(counter_after.values())
        IR_after = max(counts_after) / min(counts_after) if min(counts_after) > 0 else 1
        st.success(f"✅ Nuevo Imbalance Ratio: {IR_after:.2f} (antes: {IR:.2f})")
    
    # PASO 4: Split train/test ESTRATIFICADO
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.25, random_state=42, stratify=y_balanced
    )
    
    # PASO 5: Configurar modelo con o sin class_weight
    if balancing_method == "ClassWeights" or balancing_method == "None":
        if balancing_method == "ClassWeights":
            st.info("⚖️ Aplicando Class Weights")
            st.write("📝 Penaliza más los errores en clases minoritarias: w_c = N / (C * N_c)")
            class_weights = calculate_class_weights(y_train)
            
            # Mostrar pesos calculados
            st.write("**Pesos calculados por clase:**")
            weight_df = pd.DataFrame({
                'Clase': list(class_weights.keys()),
                'Peso': [f"{w:.3f}" for w in class_weights.values()]
            })
            st.dataframe(weight_df, hide_index=True)
        else:
            class_weights = None
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
            class_weight=class_weights  # AQUÍ SE APLICAN LOS PESOS
        )
    else:
        # Ya balanceamos con SMOTE/Undersampling, no necesitamos class_weight
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    
    # Simular progreso de entrenamiento
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        time.sleep(0.03)
        progress_bar.progress(i + 1)
        status_text.text(f"Entrenando modelo... {i+1}%")
    
    model.fit(X_train, y_train)
    
    status_text.text("✅ Modelo entrenado exitosamente!")
    
    # PASO 6: Calcular métricas (incluyendo BALANCED ACCURACY)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),  # ⭐ NUEVA MÉTRICA
        'accuracy_train': accuracy_score(y_train, y_pred_train),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'scaler': scaler,
        'balancing_method': balancing_method,  # ⭐ Guardar método usado
        'IR_before': IR  # ⭐ Guardar IR inicial
    }
    
    return model, metrics

def predict_class(frame, model, scaler, label_encoder):
    """Función independiente para clasificar - evita caching"""
    import gc
    
    # Crear copia completamente nueva
    img_copy = np.array(frame, dtype=np.uint8).copy()
    
    # Extraer características
    features = extract_features(img_copy)
    features = np.array(features, dtype=np.float64).reshape(1, -1)
    
    # Normalizar si existe scaler
    if scaler is not None:
        features = scaler.transform(features)
    
    # Crear array nuevo para predicción
    features_final = np.array(features, dtype=np.float64)
    
    # Predecir
    prediction = int(model.predict(features_final)[0])
    probabilities = np.array(model.predict_proba(features_final)[0], dtype=np.float64)
    
    # Convertir a tipos nativos de Python
    predicted_class = str(label_encoder.inverse_transform([prediction])[0])
    confidence = float(np.max(probabilities) * 100)
    
    # Top 3
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_classes = [str(c) for c in label_encoder.inverse_transform(top_indices)]
    top_probs = [float(probabilities[i] * 100) for i in top_indices]
    
    # Limpiar memoria
    gc.collect()
    
    return predicted_class, confidence, top_classes, top_probs

def capture_from_droidcam():
    """Captura imagen desde DroidCam (cámara 0)"""
    try:
        captura = cv2.VideoCapture(0)
        
        if not captura.isOpened():
            st.error("❌ No se pudo abrir la cámara. Verifica que DroidCam esté conectado.")
            return None
        
        ret, frame = captura.read()
        captura.release()
        
        if ret:
            return frame
        else:
            st.error("❌ No se pudo capturar la imagen")
            return None
            
    except Exception as e:
        st.error(f"❌ Error al capturar: {str(e)}")
        return None

# Pantalla de Login
def login_page():
    st.markdown("<h1 style='text-align: center; color: white;'>🥬 Mis Verduras Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>Sistema Inteligente de Clasificación de Verduras</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div style='background: white; padding: 40px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>", unsafe_allow_html=True)
        
        username = st.text_input("👤 Usuario", placeholder="Ingresa tu usuario")
        password = st.text_input("🔒 Contraseña", type="password", placeholder="Ingresa tu contraseña")
        
        if st.button("🚀 Iniciar Sesión"):
            if username and password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Por favor, completa todos los campos")
        
        st.info("No recuerdas tu contraseña Mimi? Es una pena, no te podemos ayudar")
        st.markdown("</div>", unsafe_allow_html=True)

# Dashboard principal
def main_dashboard():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/vegetables.png", width=150)
        st.title(f"👋 {st.session_state.username}")
        
        menu = st.radio(
            "📋 Menú Principal",
            ["🏠 Dashboard", "📊 Entrenamiento", "📸 Clasificar", "📈 Resultados"],
            key="menu"
        )
        
        st.divider()
        
        # Estadísticas
        st.metric("📷 Imágenes", len(st.session_state.training_data))
        st.metric("🧠 Modelo", "✅ Entrenado" if st.session_state.model else "❌ No entrenado")
        
        if st.session_state.metrics:
            st.metric("🎯 Accuracy", f"{st.session_state.metrics['accuracy']*100:.1f}%")
            st.metric("⚖️ Balanced Acc", f"{st.session_state.metrics['balanced_accuracy']*100:.1f}%")
        
        st.divider()
        
        if st.button("🚪 Cerrar Sesión"):
            st.session_state.logged_in = False
            st.rerun()
    
    # Contenido principal
    if menu == "🏠 Dashboard":
        show_dashboard()
    elif menu == "📊 Entrenamiento":
        show_training()
    elif menu == "📸 Clasificar":
        show_classification()
    elif menu == "📈 Resultados":
        show_results()

def show_dashboard():
    st.title("🏠 Dashboard Principal")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                <h2 style='color: white; margin: 0;'>📷</h2>
                <h1 style='color: white; margin: 10px 0;'>{len(st.session_state.training_data)}</h1>
                <p style='color: white; margin: 0;'>Imágenes</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "✅ Entrenado" if st.session_state.model else "❌ Sin entrenar"
        color = "#11998e" if st.session_state.model else "#f5576c"
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, {color} 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                <h2 style='color: white; margin: 0;'>🧠</h2>
                <h3 style='color: white; margin: 10px 0;'>{status}</h3>
                <p style='color: white; margin: 0;'>Modelo</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        accuracy = st.session_state.metrics['accuracy'] * 100 if st.session_state.metrics else 0
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                <h2 style='color: white; margin: 0;'>🎯</h2>
                <h1 style='color: white; margin: 10px 0;'>{accuracy:.1f}%</h1>
                <p style='color: white; margin: 0;'>Precisión</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                <h2 style='color: white; margin: 0;'>📊</h2>
                <h1 style='color: white; margin: 10px 0;'>{len(st.session_state.predictions_history)}</h1>
                <p style='color: white; margin: 0;'>Predicciones</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Información del sistema
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ℹ️ Estado del Sistema")
        status_df = pd.DataFrame({
            'Componente': ['Dataset', 'Modelo', 'Métricas', 'Cámara DroidCam'],
            'Estado': [
                '✅ Listo' if len(st.session_state.training_data) >= 50 else f'⚠️ {len(st.session_state.training_data)}/50 mín',
                '✅ Entrenado' if st.session_state.model else '❌ Sin entrenar',
                '✅ Disponibles' if st.session_state.metrics else '❌ No disponibles',
                '✅ Conectada (ID: 0)'
            ],
            'Detalles': [
                f'{len(st.session_state.training_data)} imágenes',
                'Random Forest' if st.session_state.model else 'N/A',
                f"{st.session_state.metrics['accuracy']*100:.1f}%" if st.session_state.metrics else 'N/A',
                'cv2.VideoCapture(0)'
            ]
        })
        st.dataframe(status_df, hide_index=True)
    
    with col2:
        st.subheader("🎓 Clases de Verduras")
        veggie_df = pd.DataFrame({
            'ID': range(1, len(VEGGIE_CLASSES)+1),
            'Verdura': VEGGIE_CLASSES
        })
        st.dataframe(veggie_df, hide_index=True)

def show_training():
    st.title("📊 Entrenamiento del Modelo")
    
    st.info("📌 **Recomendación:** Mínimo 500 imágenes (ideal 1000+) para mejores resultados")
    
    tab1, tab2 = st.tabs(["📤 Cargar Imágenes", "🚀 Entrenar Modelo"])
    
    with tab1:
        st.subheader("📤 Cargar Dataset de Imágenes")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Selecciona múltiples imágenes",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Puedes seleccionar muchas imágenes a la vez (Ctrl+Click o Cmd+Click)",
                key=f"file_uploader_{len(st.session_state.training_data)}"
            )
        
        with col2:
            st.info(f"**Total cargadas:**\n# {len(st.session_state.training_data)} imágenes")
            
            if len(st.session_state.training_data) >= 50:
                st.success("✅ Dataset suficiente")
            else:
                st.warning(f"⚠️ Faltan {50-len(st.session_state.training_data)}")
        
        if uploaded_files:
            selected_class = st.selectbox(
                "🏷️ Asignar todas las imágenes subidas a:",
                VEGGIE_CLASSES,
                key="bulk_label"
            )
            
            if st.button("✅ Agregar Imágenes al Dataset", type="primary", key="add_images_btn"):
                progress = st.progress(0)
                status = st.empty()
                
                added = 0
                for idx, uploaded_file in enumerate(uploaded_files):
                    try:
                        image = Image.open(uploaded_file)
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        img_array = np.array(image)
                        if len(img_array.shape) >= 2:
                            st.session_state.training_data.append(img_array)
                            st.session_state.labels.append(selected_class)
                            added += 1
                        progress.progress((idx + 1) / len(uploaded_files))
                        status.text(f"Procesando: {idx+1}/{len(uploaded_files)}")
                    except Exception as e:
                        st.error(f"Error con {uploaded_file.name}: {str(e)}")
                
                progress.empty()
                status.empty()
                st.success(f"✅ {added} imágenes agregadas como '{selected_class}'")
                st.balloons()
                time.sleep(1)
                st.rerun()
        
        if len(st.session_state.labels) > 0:
            st.divider()
            st.subheader("📊 Distribución del Dataset")
            
            label_counts = pd.Series(st.session_state.labels).value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                label_counts.plot(kind='bar', ax=ax, color='#667eea')
                ax.set_title('Imágenes por Clase', fontsize=16, fontweight='bold')
                ax.set_xlabel('Clase')
                ax.set_ylabel('Cantidad')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.dataframe(
                    pd.DataFrame({
                        'Clase': label_counts.index,
                        'Cantidad': label_counts.values,
                        'Porcentaje': [f"{(v/len(st.session_state.labels)*100):.1f}%" for v in label_counts.values]
                    }),
                    hide_index=True
                )
    
    with tab2:
        st.subheader("🚀 Entrenar Modelo de Clasificación")
        
        if len(st.session_state.training_data) < 50:
            st.error(f"❌ Necesitas al menos 50 imágenes. Actualmente tienes {len(st.session_state.training_data)}")
            st.info("💡 Ve a la pestaña 'Cargar Imágenes' para agregar más datos")
            return
        
        if len(set(st.session_state.labels)) < 2:
            st.error("❌ Necesitas al menos 2 clases diferentes para entrenar")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Imágenes", len(st.session_state.training_data))
        with col2:
            st.metric("🎯 Clases Únicas", len(set(st.session_state.labels)))
        with col3:
            st.metric("📈 Estado", "✅ Listo")
        
        st.divider()
        
        # ⭐⭐⭐ SECCIÓN DE DATA AUGMENTATION ⭐⭐⭐
        st.subheader("🎨 Data Augmentation (Multiplicar Dataset)")
        
        col_aug1, col_aug2 = st.columns([2, 1])
        
        with col_aug1:
            st.info("🔥 **¡RECOMENDADO!** Multiplica tus imágenes automáticamente sin buscar más")
            
            use_augmentation = st.checkbox(
                "✅ Aplicar Data Augmentation",
                value=True,
                help="Genera versiones modificadas de tus imágenes (rotación, brillo, zoom, etc.)"
            )
            
            if use_augmentation:
                multiplier = st.slider(
                    "🔢 Factor de multiplicación",
                    min_value=3,
                    max_value=10,
                    value=7,
                    help="Cada imagen generará X versiones adicionales"
                )
                
                estimated_total = len(st.session_state.training_data) * (multiplier + 1)
                st.success(f"📈 Dataset resultante: **~{estimated_total} imágenes** (de {len(st.session_state.training_data)} originales)")
        
        with col_aug2:
            st.markdown("### 🎨 Transformaciones:")
            st.write("✅ Rotación ±20°")
            st.write("✅ Volteo horizontal")
            st.write("✅ Ajuste de brillo")
            st.write("✅ Ajuste de contraste")
            st.write("✅ Ajuste de color")
            st.write("✅ Zoom aleatorio")
        
        st.divider()
        
        # SECCIÓN DE BALANCEO DE CLASES
        st.subheader("⚖️ Técnica de Balanceo de Clases")
        
        col_bal1, col_bal2 = st.columns([2, 1])
        
        with col_bal1:
            st.warning("⚠️ **IMPORTANTE:** El balanceo ayuda cuando algunas clases tienen pocas imágenes")
            
            balancing_method = st.selectbox(
                "Selecciona método de balanceo:",
                ["SMOTE", "SMOTETomek", "ClassWeights", "Undersampling", "None"],
                index=0,
                help="""
                - SMOTE: Genera ejemplos sintéticos (RECOMENDADO)
                - SMOTETomek: SMOTE + limpieza de frontera
                - ClassWeights: Penaliza errores en clases minoritarias
                - Undersampling: Reduce clase mayoritaria
                - None: Sin balanceo
                """
            )
        
        with col_bal2:
            st.markdown("### 📚 Descripción:")
            if balancing_method == "SMOTE":
                st.info("Genera nuevos ejemplos sintéticos de clases minoritarias mediante interpolación. **Más recomendado.**")
            elif balancing_method == "SMOTETomek":
                st.info("SMOTE + elimina ejemplos ambiguos en la frontera. **Muy efectivo.**")
            elif balancing_method == "ClassWeights":
                st.info("Penaliza más los errores en clases con pocas muestras (w = N/(C*Nc))")
            elif balancing_method == "Undersampling":
                st.warning("Reduce ejemplos de clase mayoritaria. **Pierdes datos.**")
            else:
                st.error("Sin balanceo. **Solo si dataset ya está balanceado.**")
        
        st.divider()
        
        if st.button("🧠 INICIAR ENTRENAMIENTO", type="primary"):
            with st.spinner("🔄 Preparando datos..."):
                
                images_to_train = st.session_state.training_data.copy()
                labels_to_train = st.session_state.labels.copy()
                
                # APLICAR DATA AUGMENTATION
                if use_augmentation:
                    st.info(f"🎨 Aplicando Data Augmentation (esto puede tomar 1-2 minutos)...")
                    images_to_train, labels_to_train = apply_data_augmentation(
                        images_to_train,
                        labels_to_train,
                        multiplier=multiplier
                    )
                    st.success(f"✅ Dataset aumentado: {len(images_to_train)} imágenes totales!")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                X = []
                for idx, img in enumerate(images_to_train):
                    features = extract_features(img)
                    X.append(features)
                    if idx % 50 == 0:
                        progress_bar.progress((idx + 1) / len(images_to_train))
                        status_text.text(f"Extrayendo características: {idx+1}/{len(images_to_train)}")
                
                X = np.array(X)
                
                le = LabelEncoder()
                y = le.fit_transform(labels_to_train)
                
                st.session_state.label_encoder = le
                
                status_text.text("🧠 Entrenando modelo con balanceo de clases...")
                
                # LLAMAR A LA FUNCIÓN DE ENTRENAMIENTO CON BALANCEO
                model, metrics = train_model(X, y, balancing_method=balancing_method)
                
                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.scaler = metrics['scaler']
                
                st.success("✅ ¡Modelo entrenado exitosamente!")
                st.balloons()
                
                st.subheader("📊 Resultados del Entrenamiento")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.2f}%")
                col2.metric("⚖️ Balanced Acc", f"{metrics['balanced_accuracy']*100:.2f}%")
                col3.metric("🎯 Precision", f"{metrics['precision']*100:.2f}%")
                col4.metric("🎯 F1-Score", f"{metrics['f1']*100:.2f}%")
                
                st.info(f"""
                **Configuración usada:**
                - Data Augmentation: {'Sí (x' + str(multiplier) + ')' if use_augmentation else 'No'}
                - Método de Balanceo: {metrics['balancing_method']}
                - Imbalance Ratio inicial: {metrics['IR_before']:.2f}
                - Total imágenes entrenadas: {len(images_to_train)}
                """)
                
                st.info("💡 Ve a la pestaña '📈 Resultados' para ver métricas detalladas")
                
                time.sleep(2)

def show_classification():
    st.title("📸 Clasificación con DroidCam")
    
    if st.session_state.model is None:
        st.error("❌ Primero debes entrenar un modelo en la sección '📊 Entrenamiento'")
        return
    
    st.success("✅ Modelo cargado y listo para clasificar")
    
    st.info("📱 **Instrucciones:** 1) Click en 'ABRIR CÁMARA' para ver el video en vivo. 2) Click 'CAPTURAR' cuando veas la verdura. 3) La ventana se cerrará automáticamente.")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📷 Control de Cámara DroidCam")
        
        # Botones de control
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🎥 ABRIR CÁMARA", type="primary"):
                st.info("🔹 Abriendo cámara DroidCam en ventana nueva...")
                
                try:
                    # Abrir ventana de cámara
                    cv2.namedWindow("DroidCam - Presiona ESPACIO para capturar, ESC para salir")
                    captura = cv2.VideoCapture(0)
                    
                    if not captura.isOpened():
                        st.error("❌ No se pudo abrir DroidCam. Verifica la conexión.")
                    else:
                        st.success("✅ Cámara abierta. Presiona ESPACIO para capturar o ESC para salir.")
                        
                        captured_frame = None
                        
                        while True:
                            ret, frame = captura.read()
                            
                            if not ret:
                                st.error("❌ Error al leer frame")
                                break
                            
                            # CREAR COPIA DEL FRAME INMEDIATAMENTE
                            display_frame = frame.copy()
                            
                            # Agregar texto instructivo en el frame
                            cv2.putText(display_frame, "ESPACIO = Capturar | ESC = Salir", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.7, (0, 255, 0), 2)
                            
                            cv2.imshow("DroidCam - Presiona ESPACIO para capturar, ESC para salir", display_frame)
                            
                            key = cv2.waitKey(1) & 0xFF
                            
                            # ESPACIO para capturar
                            if key == 32:
                                # IMPORTANTE: Crear copia profunda del frame original
                                captured_frame = frame.copy()
                                captured_display = captured_frame.copy()
                                cv2.putText(captured_display, "CAPTURADO!", 
                                          (captured_display.shape[1]//2 - 100, captured_display.shape[0]//2), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                          1.5, (0, 255, 0), 3)
                                cv2.imshow("DroidCam - Presiona ESPACIO para capturar, ESC para salir", captured_display)
                                cv2.waitKey(500)
                                break
                            
                            # ESC para salir
                            elif key == 27:
                                break
                        
                        captura.release()
                        cv2.destroyAllWindows()
                        
                        # Si se capturó algo, clasificar
                        if captured_frame is not None:
                            # GUARDAR COPIA INDEPENDIENTE
                            frame_to_save = np.array(captured_frame, dtype=np.uint8).copy()
                            st.session_state.captured_images.append(frame_to_save)
                            
                            # USAR FUNCIÓN DE PREDICCIÓN INDEPENDIENTE
                            predicted_class, confidence, top_classes, top_probs = predict_class(
                                captured_frame,
                                st.session_state.model,
                                st.session_state.scaler,
                                st.session_state.label_encoder
                            )
                            
                            # Guardar predicción
                            pred_data = {
                                'imagen': frame_to_save.copy(),
                                'clase': predicted_class,
                                'confianza': confidence,
                                'top_classes': list(top_classes),
                                'top_probs': list(top_probs),
                                'timestamp': time.strftime("%H:%M:%S"),
                                'id': time.time()
                            }
                            st.session_state.predictions_history.insert(0, pred_data)
                            
                            if len(st.session_state.predictions_history) > 20:
                                st.session_state.predictions_history = st.session_state.predictions_history[:20]
                            
                            st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col_btn2:
            if st.button("📸 CAPTURA RÁPIDA"):
                with st.spinner("📸 Capturando..."):
                    frame = capture_from_droidcam()
                    
                    if frame is not None:
                        # CREAR COPIA INDEPENDIENTE
                        frame_to_save = np.array(frame, dtype=np.uint8).copy()
                        st.session_state.captured_images.append(frame_to_save)
                        
                        # USAR FUNCIÓN DE PREDICCIÓN INDEPENDIENTE
                        predicted_class, confidence, top_classes, top_probs = predict_class(
                            frame,
                            st.session_state.model,
                            st.session_state.scaler,
                            st.session_state.label_encoder
                        )
                        
                        pred_data = {
                            'imagen': frame_to_save.copy(),
                            'clase': predicted_class,
                            'confianza': confidence,
                            'top_classes': list(top_classes),
                            'top_probs': list(top_probs),
                            'timestamp': time.strftime("%H:%M:%S"),
                            'id': time.time()
                        }
                        st.session_state.predictions_history.insert(0, pred_data)
                        
                        if len(st.session_state.predictions_history) > 20:
                            st.session_state.predictions_history = st.session_state.predictions_history[:20]
                        
                        st.success(f"✅ {predicted_class} ({confidence:.1f}%)")
                        st.rerun()
        
        with col_btn3:
            if st.button("🗑️ Limpiar"):
                st.session_state.predictions_history = []
                st.session_state.captured_images = []
                st.rerun()
        
        st.divider()
        
        # Mostrar última captura
        if len(st.session_state.captured_images) > 0:
            st.subheader("📸 Última Captura")
            st.image(
                cv2.cvtColor(st.session_state.captured_images[-1], cv2.COLOR_BGR2RGB),
                caption=f"Clasificada como: {st.session_state.predictions_history[0]['clase']}" if st.session_state.predictions_history else "Captura",
                use_column_width=True
            )
        else:
            st.info("👆 Haz clic en 'ABRIR CÁMARA' para comenzar")
    
    with col2:
        st.subheader("📊 Historial de Predicciones")
        
        if len(st.session_state.predictions_history) == 0:
            st.info("👆 Captura una imagen para comenzar")
        else:
            # Mostrar última predicción destacada
            last_pred = st.session_state.predictions_history[0]
            
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 25px; border-radius: 15px; color: white; text-align: center; margin-bottom: 20px;'>
                    <h2 style='color: white; margin: 0;'>🥬 {last_pred['clase']}</h2>
                    <h3 style='color: white; margin: 10px 0;'>Confianza: {last_pred['confianza']:.1f}%</h3>
                    <p style='color: white; margin: 0;'>⏰ {last_pred['timestamp']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Historial completo
            st.write("### 📝 Últimas clasificaciones:")
            
            for idx, pred in enumerate(st.session_state.predictions_history[:10]):
                with st.expander(f"#{idx+1} - {pred['clase']} ({pred['confianza']:.1f}%) - {pred['timestamp']}"):
                    col_img, col_info = st.columns([1, 1])
                    
                    with col_img:
                        st.image(
                            cv2.cvtColor(pred['imagen'], cv2.COLOR_BGR2RGB),
                            use_column_width=True
                        )
                    
                    with col_info:
                        st.write(f"**🥇 Predicción Principal:**")
                        st.write(f"**{pred['clase']}** - {pred['confianza']:.1f}%")
                        
                        if 'top_classes' in pred and len(pred['top_classes']) > 1:
                            st.write(f"")
                            st.write(f"**📊 Top 3 Predicciones:**")
                            for i, (cls, prob) in enumerate(zip(pred['top_classes'], pred['top_probs']), 1):
                                st.write(f"{i}. {cls}: {prob:.1f}%")
                        
                        st.write(f"")
                        st.write(f"**⏰ Hora:** {pred['timestamp']}")

def show_results():
    st.title("📈 Resultados y Métricas del Modelo")
    
    if st.session_state.metrics is None:
        st.error("❌ No hay métricas disponibles. Primero entrena un modelo.")
        st.info("💡 Ve a la sección '📊 Entrenamiento' para entrenar el modelo")
        return
    
    metrics = st.session_state.metrics
    
    # Métricas principales
    st.subheader("🎯 Métricas Generales de Rendimiento")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Accuracy", f"{metrics['accuracy']*100:.2f}%", 
                  delta=f"{(metrics['accuracy']-metrics['accuracy_train'])*100:.2f}%" if 'accuracy_train' in metrics else None)
    with col2:
        st.metric("⚖️ Balanced Accuracy", f"{metrics['balanced_accuracy']*100:.2f}%")
    with col3:
        st.metric("🎯 Precision", f"{metrics['precision']*100:.2f}%")
    with col4:
        st.metric("🔍 Recall", f"{metrics['recall']*100:.2f}%")
    
    st.divider()
    
    # Información del modelo
    st.subheader("ℹ️ Información del Entrenamiento")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Train Set", metrics['train_size'])
    with col2:
        st.metric("🧪 Test Set", metrics['test_size'])
    with col3:
        st.metric("⚖️ Método Balanceo", metrics['balancing_method'])
    with col4:
        st.metric("📉 IR Inicial", f"{metrics['IR_before']:.2f}")
    
    st.divider()
    
    # Matriz de confusión y reporte
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📊 Matriz de Confusión")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        classes = st.session_state.label_encoder.classes_
        
        sns.heatmap(
            metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='RdYlGn',
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            cbar_kws={'label': 'Cantidad'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        ax.set_ylabel('Clase Verdadera', fontsize=12, fontweight='bold')
        ax.set_xlabel('Clase Predicha', fontsize=12, fontweight='bold')
        ax.set_title('Matriz de Confusión del Modelo', fontsize=14, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("📋 Reporte Detallado")
        st.text(metrics['classification_report'])
        
        st.divider()
        
        st.info(f"""
        **Configuración del Modelo:**
        
        - **Algoritmo:** Random Forest
        - **n_estimators:** 200
        - **max_depth:** 30
        - **min_samples_split:** 3
        - **min_samples_leaf:** 1
        - **Balanceo:** {metrics['balancing_method']}
        """)
    
    st.divider()
    
    # Distribución de confianza
    st.subheader("📊 Análisis de Confianza en Predicciones")
    
    if len(st.session_state.predictions_history) > 0:
        confidences = [pred['confianza'] for pred in st.session_state.predictions_history]
        classes_pred = [pred['clase'] for pred in st.session_state.predictions_history]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(confidences, bins=20, color='#667eea', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Confianza (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
            ax.set_title('Distribución de Confianza', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            pred_counts = pd.Series(classes_pred).value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            pred_counts.plot(kind='barh', ax=ax, color='#38ef7d')
            ax.set_xlabel('Cantidad', fontsize=12, fontweight='bold')
            ax.set_ylabel('Clase', fontsize=12, fontweight='bold')
            ax.set_title('Predicciones Realizadas', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("📸 Realiza algunas predicciones para ver el análisis de confianza")

# Función principal
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        main_dashboard()

if __name__ == "__main__":
    main()


