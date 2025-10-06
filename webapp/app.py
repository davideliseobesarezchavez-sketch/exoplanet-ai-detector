# webapp/app.py - VERSIÓN CON RUTAS CORREGIDAS
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# OBTENER LA RUTA CORRECTA DEL PROYECTO
def get_project_root():
    """Obtener la ruta raíz del proyecto de forma confiable"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    return project_root

PROJECT_ROOT = get_project_root()

class ExoplanetDataProcessor:
    """Procesador de datos para los datasets reales de la NASA"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_real_data(self):
        """Cargar los datasets reales de la NASA con rutas corregidas"""
        try:
            # Usar rutas absolutas desde la raíz del proyecto
            data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
            
            st.info(f"🔍 Buscando datos en: {data_dir}")
            
            # Listar archivos en el directorio
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                st.info(f"📁 Archivos encontrados en data/raw/: {files}")
            else:
                st.error(f"❌ No existe el directorio: {data_dir}")
                return None, None, None
            
            # Construir rutas completas
            kepler_path = os.path.join(data_dir, 'kepler.csv')
            k2_path = os.path.join(data_dir, 'k2.csv') 
            tess_path = os.path.join(data_dir, 'tess.csv')
            
            st.info(f"📊 Intentando cargar:\n- {kepler_path}\n- {k2_path}\n- {tess_path}")
            
            # Verificar que los archivos existen
            if not os.path.exists(kepler_path):
                st.error(f"❌ No existe: {kepler_path}")
                # Buscar archivos similares
                csv_files = [f for f in files if f.endswith('.csv')]
                if csv_files:
                    st.info(f"📄 Archivos CSV disponibles: {csv_files}")
                return None, None, None
            
            # Cargar los archivos reales
            kepler_df = pd.read_csv(kepler_path)
            k2_df = pd.read_csv(k2_path) if os.path.exists(k2_path) else None
            tess_df = pd.read_csv(tess_path) if os.path.exists(tess_path) else None
            
            st.success(f"✅ Kepler cargado: {len(kepler_df)} registros")
            if k2_df is not None:
                st.success(f"✅ K2 cargado: {len(k2_df)} registros")
            if tess_df is not None:
                st.success(f"✅ TESS cargado: {len(tess_df)} registros")
            
            return kepler_df, k2_df, tess_df
            
        except Exception as e:
            st.error(f"❌ Error cargando datasets: {e}")
            return None, None, None
    
    def preprocess_kepler(self, df):
        """Preprocesar datos Kepler reales - VERSIÓN MEJORADA"""
        df_clean = df.copy()
        
        st.info("🔧 Procesando datos Kepler...")
        
        # Mostrar columnas disponibles
        st.write(f"📋 Columnas en Kepler: {list(df_clean.columns)}")
        
        # Verificar si existe la columna de target
        if 'koi_disposition' not in df_clean.columns:
            st.error("❌ No se encuentra la columna 'koi_disposition' en Kepler")
            st.info("Las columnas disponibles son:")
            st.write(list(df_clean.columns))
            return df_clean
        
        # Eliminar columnas no útiles (basado en el paper)
        columns_to_drop = ['kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_score']
        columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
        
        if columns_to_drop:
            df_clean = df_clean.drop(columns=columns_to_drop)
            st.write(f"🗑️ Columnas eliminadas: {columns_to_drop}")
        
        # Mostrar valores únicos en la columna de disposición
        st.write(f"🎯 Valores en koi_disposition: {df_clean['koi_disposition'].unique()}")
        
        # Filtrar solo confirmed, candidate y false positive
        valid_dispositions = ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']
        mask = df_clean['koi_disposition'].isin(valid_dispositions)
        df_clean = df_clean[mask]
        
        st.write(f"📊 Distribución después de filtrar: {df_clean['koi_disposition'].value_counts().to_dict()}")
        
        # Crear target binario
        df_clean['target'] = df_clean['koi_disposition'].map({
            'CONFIRMED': 1, 
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0
        })
        
        # Añadir identificador de misión
        df_clean['mission'] = 'kepler'
        
        st.success(f"✅ Kepler procesado: {len(df_clean)} registros")
        
        return df_clean
    
    def preprocess_k2(self, df):
        """Preprocesar datos K2 reales"""
        if df is None:
            st.warning("⚠️ Dataset K2 no disponible")
            return None
            
        df_clean = df.copy()
        
        st.info("🔧 Procesando datos K2...")
        st.write(f"📋 Columnas en K2: {list(df_clean.columns)}")
        
        # Verificar columnas necesarias
        if 'disposition' not in df_clean.columns:
            st.error("❌ No se encuentra la columna 'disposition' en K2")
            return None
        
        # Filtrar solo confirmed y candidate
        df_clean = df_clean[df_clean['disposition'].isin(['CONFIRMED', 'CANDIDATE'])]
        
        # Target binario
        df_clean['target'] = df_clean['disposition'].map({
            'CONFIRMED': 1,
            'CANDIDATE': 1
        })
        
        # Identificador de misión
        df_clean['mission'] = 'k2'
        
        st.success(f"✅ K2 procesado: {len(df_clean)} registros")
        
        return df_clean
    
    def preprocess_tess(self, df):
        """Preprocesar datos TESS reales"""
        if df is None:
            st.warning("⚠️ Dataset TESS no disponible")
            return None
            
        df_clean = df.copy()
        
        st.info("🔧 Procesando datos TESS...")
        st.write(f"📋 Columnas en TESS: {list(df_clean.columns)}")
        
        # Verificar columnas necesarias
        if 'tfopwg_disp' not in df_clean.columns:
            st.error("❌ No se encuentra la columna 'tfopwg_disp' en TESS")
            return None
        
        # Mapear disposiciones de TESS
        disposition_mapping = {
            'PC': 1, 'KP': 1, 'APC': 1,  # Positivos
            'FP': 0, 'FA': 0  # Negativos
        }
        
        df_clean['target'] = df_clean['tfopwg_disp'].map(disposition_mapping)
        df_clean = df_clean.dropna(subset=['target'])
        
        # Identificador de misión
        df_clean['mission'] = 'tess'
        
        st.success(f"✅ TESS procesado: {len(df_clean)} registros")
        
        return df_clean
    
    def prepare_features(self, df):
        """Preparar características para el modelo - VERSIÓN FLEXIBLE"""
        if df is None or len(df) == 0:
            st.error("❌ No hay datos para preparar características")
            return None, None, None
            
        st.info("🔧 Preparando características...")
        
        # Posibles nombres de columnas en diferentes datasets
        possible_features = {
            'orbital_period': ['koi_period', 'pl_orbper', 'period'],
            'transit_duration': ['koi_duration', 'pl_trandurh', 'duration'],
            'transit_depth': ['koi_depth', 'pl_trandep', 'depth'], 
            'planet_radius': ['koi_prad', 'pl_rade', 'radius'],
            'equilibrium_temp': ['koi_teq', 'pl_eqt', 'teq'],
            'insolation_flux': ['koi_insol', 'pl_insol', 'insol'],
            'stellar_teff': ['koi_steff', 'st_teff', 'teff'],
            'stellar_logg': ['koi_slogg', 'st_logg', 'logg'],
            'stellar_radius': ['koi_srad', 'st_rad', 'srad']
        }
        
        # Encontrar las columnas disponibles
        available_columns = []
        for feature_name, possible_names in possible_features.items():
            for name in possible_names:
                if name in df.columns:
                    available_columns.append(name)
                    break
        
        st.write(f"📊 Columnas numéricas encontradas: {available_columns}")
        
        if not available_columns:
            st.error("❌ No se encontraron columnas numéricas para entrenar")
            return None, None, None
        
        X = df[available_columns].copy()
        y = df['target'].values
        
        st.write(f"📊 Shape de X: {X.shape}, Shape de y: {y.shape}")
        
        # Manejar valores missing
        missing_before = X.isnull().sum().sum()
        X = X.fillna(X.median())
        missing_after = X.isnull().sum().sum()
        
        st.write(f"🔧 Valores missing: {missing_before} antes, {missing_after} después")
        
        # Escalar características
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = available_columns
        
        st.success(f"✅ Características preparadas: {X_scaled.shape}")
        
        return X_scaled, y, available_columns

class RealExoplanetModel:
    """Modelo real para entrenamiento con datos de la NASA"""
    
    def __init__(self):
        self.model = None
        self.accuracy = 0
        self.feature_importance = None
    
    def create_ensemble(self):
        """Crear ensemble con los algoritmos del paper"""
        base_models = [
            ('random_forest', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )),
            ('extra_trees', ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )),
            ('xgboost', XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )),
            ('lightgbm', LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=42
            ))
        ]
        
        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(),
            cv=3,  # Reducido para mayor velocidad
            passthrough=False,
            n_jobs=-1
        )
        
        return ensemble
    
    def train(self, X, y):
        """Entrenar el modelo real"""
        if X is None or y is None:
            st.error("❌ No hay datos para entrenar")
            return None
            
        st.info("🤖 Iniciando entrenamiento del ensemble...")
        
        self.model = self.create_ensemble()
        self.model.fit(X, y)
        
        # Calcular accuracy en entrenamiento
        y_pred = self.model.predict(X)
        self.accuracy = accuracy_score(y, y_pred)
        
        st.write(f"📈 Accuracy en entrenamiento: {self.accuracy:.2%}")
        
        # Calcular importancia de características
        self._calculate_feature_importance(X.shape[1])
        
        return self.model
    
    def _calculate_feature_importance(self, n_features):
        """Calcular importancia de características promediada"""
        importances = np.zeros(n_features)
        
        for name, model in self.model.named_estimators_.items():
            if hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_
        
        if len(self.model.named_estimators_) > 0:
            self.feature_importance = importances / len(self.model.named_estimators_)
    
    def save_model(self, filepath):
        """Guardar modelo entrenado"""
        if self.model:
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.model, filepath)
            return True
        return False
    
    def load_model(self, filepath):
        """Cargar modelo entrenado"""
        try:
            if os.path.exists(filepath):
                self.model = joblib.load(filepath)
                return True
        except Exception as e:
            st.error(f"Error cargando modelo: {e}")
        return False

class ExoplanetDetectorApp:
    def __init__(self):
        self.model = RealExoplanetModel()
        self.data_processor = ExoplanetDataProcessor()
        self.model_trained = False
        
    def render_sidebar(self):
        """Barra lateral de navegación - ACTUALIZADA"""
        st.sidebar.title("🔭 NASA Exoplanet Detector - REAL")
        st.sidebar.markdown("---")
        
        page = st.sidebar.radio("Navegación", [
            "🏠 Inicio", 
            "🚀 Entrenar Modelo REAL",
            "🤖 Clasificar Exoplanetas",
            "📦 Clasificación por Lotes",  # ¡NUEVA OPCIÓN!
            "📊 Análisis de Datos REAL",
            "💾 Modelos Guardados"
        ])
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "Sistema REAL con datos de Kepler, K2 y TESS de la NASA"
        )
        
        return page

    def render_home(self):
        """Página de inicio"""
        st.title("🪐 NASA Exoplanet Detection AI - SISTEMA REAL")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Sistema REAL de Detección de Exoplanetas
            
            **Características IMPLEMENTADAS:**
            - ✅ **Entrenamiento REAL** con datos de la NASA
            - ✅ **Modelos PERSISTENTES** que se guardan en disco
            - ✅ **Datos REALES** Kepler, K2 y TESS
            - ✅ **Ensemble Stacking** como en el paper científico
            - ✅ **Guardado/Auto-carga** de modelos
            
            **Para comenzar:**
            1. Verifica que tus archivos CSV estén en `data/raw/`
            2. Ve a **'Entrenar Modelo REAL'**
            3. ¡El sistema detectará automáticamente tus datos!
            """)
        
        with col2:
            st.image("https://www.nasa.gov/sites/default/files/thumbnails/image/kepler_all_planets_art.jpg", 
                    use_column_width=True,
                    caption="Datos REALES de la NASA")
        
        # Verificar estructura de archivos
        st.subheader("🔍 Verificación de Archivos")
        
        data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw')
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            if csv_files:
                st.success(f"✅ Directorio data/raw/ encontrado")
                st.write(f"📄 Archivos CSV: {csv_files}")
            else:
                st.warning(f"⚠️ Directorio existe pero no hay archivos CSV")
        else:
            st.error(f"❌ No existe el directorio: {data_dir}")
            st.info("""
            **Solución:**
            1. Crea la carpeta `data/raw/` en tu proyecto
            2. Coloca allí tus archivos `kepler.csv`, `k2.csv`, `tess.csv`
            3. Recarga esta página
            """)
        
        # Verificar si hay modelo entrenado
        model_path = os.path.join(PROJECT_ROOT, 'models', 'real_ensemble_model.pkl')
        if os.path.exists(model_path):
            st.success("✅ **Modelo entrenado disponible** - Puedes usarlo en 'Clasificar Exoplanetas'")
            if self.model.load_model(model_path):
                st.metric("Accuracy del Modelo", f"{self.model.accuracy:.1%}")
                self.model_trained = True
        else:
            st.warning("⚠️ **No hay modelo entrenado** - Ve a 'Entrenar Modelo REAL' para comenzar")

    def render_real_training(self):
        """Página de entrenamiento REAL con datos de la NASA"""
        st.title("🚀 Entrenamiento REAL con Datos NASA")
        
        st.info("""
        **Entrenamiento REAL del modelo Ensemble** usando tus datasets de:
        - Kepler.csv (datos reales)
        - K2.csv (datos reales) 
        - TESS.csv (datos reales)
        
        El modelo entrenado se guardará automáticamente y estará disponible para clasificación.
        """)
        
        if st.button("🎯 Iniciar Entrenamiento REAL", type="primary"):
            with st.spinner("Cargando y procesando datos REALES de la NASA..."):
                try:
                    # Cargar datos reales
                    kepler_df, k2_df, tess_df = self.data_processor.load_real_data()
                    
                    if kepler_df is None:
                        st.error("""
                        ❌ **No se pudieron cargar los datasets**
                        
                        **Posibles soluciones:**
                        1. Verifica que los archivos estén en `data/raw/`
                        2. Asegúrate de que se llamen `kepler.csv`, `k2.csv`, `tess.csv`
                        3. Verifica que los archivos no estén corruptos
                        """)
                        return
                    
                    # Mostrar información de los datasets
                    st.subheader("📊 Datasets Cargados")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Kepler", f"{len(kepler_df):,} registros")
                    with col2:
                        k2_count = len(k2_df) if k2_df is not None else 0
                        st.metric("K2", f"{k2_count:,} registros")
                    with col3:
                        tess_count = len(tess_df) if tess_df is not None else 0
                        st.metric("TESS", f"{tess_count:,} registros")
                    
                    # Procesar datos
                    st.subheader("🔧 Procesando Datos...")
                    
                    # Preprocesar Kepler
                    kepler_processed = self.data_processor.preprocess_kepler(kepler_df)
                    if kepler_processed is None:
                        return
                    
                    # Preprocesar K2 y TESS si están disponibles
                    datasets_to_process = [kepler_processed]
                    
                    if k2_df is not None:
                        k2_processed = self.data_processor.preprocess_k2(k2_df)
                        if k2_processed is not None:
                            datasets_to_process.append(k2_processed)
                    
                    if tess_df is not None:
                        tess_processed = self.data_processor.preprocess_tess(tess_df)
                        if tess_processed is not None:
                            datasets_to_process.append(tess_processed)
                    
                    # Unificar datos (filtrar None values)
                    datasets_to_process = [d for d in datasets_to_process if d is not None]
                    if not datasets_to_process:
                        st.error("❌ No hay datos válidos para procesar")
                        return
                    
                    unified_data = pd.concat(datasets_to_process, ignore_index=True)
                    
                    st.success(f"✅ Datos unificados: {len(unified_data):,} muestras")
                    
                    # Preparar características
                    X, y, feature_names = self.data_processor.prepare_features(unified_data)
                    
                    if X is None:
                        st.error("❌ No se pudieron preparar las características")
                        return
                    
                    # Entrenar modelo
                    st.subheader("🤖 Entrenando Modelo Ensemble...")
                    trained_model = self.model.train(X, y)
                    
                    if trained_model is None:
                        st.error("❌ Error en el entrenamiento")
                        return
                    
                    # Guardar modelo
                    models_dir = os.path.join(PROJECT_ROOT, 'models')
                    model_path = os.path.join(models_dir, 'real_ensemble_model.pkl')
                    
                    model_saved = self.model.save_model(model_path)
                    
                    if model_saved:
                        # Guardar también el preprocesador y feature names
                        processor_path = os.path.join(models_dir, 'data_processor.pkl')
                        features_path = os.path.join(models_dir, 'feature_names.pkl')
                        
                        joblib.dump(self.data_processor, processor_path)
                        joblib.dump(feature_names, features_path)
                        
                        st.success("✅ Modelo entrenado y guardado exitosamente!")
                        self.model_trained = True
                        
                        # Mostrar resultados
                        st.subheader("📈 Resultados del Entrenamiento")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{self.model.accuracy:.2%}")
                        with col2:
                            st.metric("Muestras", f"{X.shape[0]:,}")
                        with col3:
                            st.metric("Características", X.shape[1])
                        with col4:
                            st.metric("Algoritmos", "4 Ensemble")
                        
                        # Importancia de características
                        if self.model.feature_importance is not None:
                            st.subheader("🔍 Importancia de Características")
                            importance_df = pd.DataFrame({
                                'Característica': feature_names,
                                'Importancia': self.model.feature_importance
                            }).sort_values('Importancia', ascending=False)
                            
                            fig = px.bar(
                                importance_df.head(10),
                                x='Importancia',
                                y='Característica',
                                title='Top 10 Características Más Importantes',
                                orientation='h'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error durante el entrenamiento: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # ... (el resto de las funciones se mantienen igual que antes)
    def render_real_classification(self):
        """Clasificación con modelo REAL entrenado - VERSIÓN CORREGIDA"""
        st.title("🤖 Clasificación con Modelo REAL")
        
        # Verificar si hay modelo entrenado
        model_path = os.path.join(PROJECT_ROOT, 'models', 'real_ensemble_model.pkl')
        if not os.path.exists(model_path):
            st.warning("""
            ⚠️ **No hay modelo entrenado**
            
            Para usar el clasificador REAL:
            1. Ve a la pestaña **'Entrenar Modelo REAL'**
            2. Entrena el modelo con tus datos de la NASA
            3. Regresa aquí para clasificar candidatos
            """)
            return
        
        # Cargar modelo
        if not self.model_trained:
            if self.model.load_model(model_path):
                self.model_trained = True
                st.success("✅ Modelo REAL cargado exitosamente")
                
                # Mostrar información de características
                features_path = os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl')
                if os.path.exists(features_path):
                    feature_names = joblib.load(features_path)
                    st.info(f"🔍 El modelo espera {len(feature_names)} características: {', '.join(feature_names)}")
            else:
                st.error("❌ Error cargando el modelo")
                return
        
        st.info("""
        🔍 **Clasificador REAL**: Introduce los 9 parámetros astronómicos que el modelo espera.
        """)
        
        with st.form("real_classification_form"):
            st.subheader("📐 Parámetros del Candidato - 9 CARACTERÍSTICAS REQUERIDAS")
            
            col1, col2 = st.columns(2)
            
            with col1:
                koi_period = st.number_input("Período Orbital - koi_period (días)", 
                                        min_value=0.1, max_value=1000.0, value=10.0,
                                        help="Tiempo orbital del planeta")
                
                koi_duration = st.number_input("Duración Tránsito - koi_duration (horas)", 
                                            min_value=0.1, max_value=24.0, value=3.0,
                                            help="Duración del tránsito")
                
                koi_depth = st.number_input("Profundidad Tránsito - koi_depth (ppm)", 
                                        min_value=1, max_value=100000, value=500,
                                        help="Disminución de brillo durante tránsito")
                
                koi_prad = st.number_input("Radio Planetario - koi_prad (Radios Tierra)", 
                                        min_value=0.1, max_value=50.0, value=2.0,
                                        help="Radio del planeta en unidades terrestres")
                
                koi_teq = st.number_input("Temperatura Equilibrio - koi_teq (K)", 
                                        min_value=100, max_value=5000, value=500,
                                        help="Temperatura de equilibrio del planeta")
            
            with col2:
                koi_insol = st.number_input("Flujo de Insolación - koi_insol", 
                                        min_value=0.1, max_value=10000.0, value=100.0,
                                        help="Flujo de radiación recibido")
                
                koi_steff = st.number_input("Temperatura Estelar - koi_steff (K)", 
                                        min_value=2000, max_value=15000, value=5800,
                                        help="Temperatura efectiva de la estrella")
                
                koi_slogg = st.number_input("Gravedad Estelar - koi_slogg (log g)", 
                                        min_value=3.0, max_value=5.5, value=4.4,
                                        help="Gravedad superficial estelar")
                
                # ¡ESTE ES EL CAMPO QUE FALTABA!
                koi_srad = st.number_input("Radio Estelar - koi_srad (Radios Sol)", 
                                        min_value=0.1, max_value=10.0, value=1.0,
                                        help="Radio de la estrella en unidades solares")
            
            submitted = st.form_submit_button("🚀 Clasificar con Modelo REAL")
        
        if submitted:
            # Verificar que tenemos todas las características
            features = (
                koi_period, koi_duration, koi_depth, koi_prad,
                koi_teq, koi_insol, koi_steff, koi_slogg, koi_srad  # ¡Ahora son 9!
            )
            
            st.info(f"🔍 Enviando {len(features)} características al modelo")
            self._real_prediction(*features)

    def _real_prediction(self, *features):
        """Predicción REAL con el modelo entrenado - VERSIÓN CORREGIDA"""
        # Cargar información del modelo
        processor_path = os.path.join(PROJECT_ROOT, 'models', 'data_processor.pkl')
        features_path = os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl')
        
        try:
            # Verificar que tenemos los archivos necesarios
            if not os.path.exists(processor_path) or not os.path.exists(features_path):
                st.error("❌ No se encontraron los archivos del modelo entrenado")
                return
            
            # Cargar feature names y preprocesador
            saved_feature_names = joblib.load(features_path)
            data_processor = joblib.load(processor_path)
            
            # VERIFICACIÓN CRÍTICA: ¿Coincide el número de características?
            if len(features) != len(saved_feature_names):
                st.error(f"""
                ❌ **ERROR CRÍTICO - Discrepancia en características**
                
                **Envías:** {len(features)} características
                **Modelo espera:** {len(saved_feature_names)} características
                
                **Características esperadas por el modelo:**
                {saved_feature_names}
                
                **Solución:** Asegúrate de que el formulario tenga exactamente {len(saved_feature_names)} campos.
                """)
                return
            
            st.success(f"✅ Coincidencia perfecta: {len(features)} características enviadas")
            
            # Crear array de características
            feature_array = np.array([features]).reshape(1, -1)
            
            # Escalar características
            feature_array_scaled = data_processor.scaler.transform(feature_array)
            
            # Realizar predicción
            prediction = self.model.model.predict(feature_array_scaled)[0]
            probability = self.model.model.predict_proba(feature_array_scaled)[0, 1]
            
            # Mostrar resultados
            st.subheader("🎯 Resultado de la Clasificación REAL")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if prediction == 1:
                    st.success("✅ **EXOPLANETA DETECTADO**")
                    st.balloons()
                else:
                    st.error("❌ **NO ES EXOPLANETA**")
                
                st.metric("Probabilidad", f"{probability:.2%}")
                
                # Interpretación de la probabilidad
                if probability >= 0.8:
                    st.info("🟢 **Alta confianza** - Muy probable exoplaneta")
                elif probability >= 0.6:
                    st.info("🟡 **Confianza media** - Posible exoplaneta")
                else:
                    st.info("🔴 **Baja confianza** - Probable falso positivo")
            
            with col2:
                # Análisis detallado de características
                st.markdown("#### 📊 Análisis de Características")
                
                # Mapeo de nombres amigables
                feature_display_names = {
                    'koi_period': 'Período Orbital',
                    'koi_duration': 'Duración Tránsito', 
                    'koi_depth': 'Profundidad Tránsito',
                    'koi_prad': 'Radio Planetario',
                    'koi_teq': 'Temperatura Planeta',
                    'koi_insol': 'Flujo Insolación',
                    'koi_steff': 'Temperatura Estelar',
                    'koi_slogg': 'Gravedad Estelar',
                    'koi_srad': 'Radio Estelar'
                }
                
                # Mapeo de unidades
                feature_units = {
                    'koi_period': 'días',
                    'koi_duration': 'horas', 
                    'koi_depth': 'ppm',
                    'koi_prad': 'R⊕',
                    'koi_teq': 'K',
                    'koi_insol': 'S⊕',
                    'koi_steff': 'K',
                    'koi_slogg': 'log g',
                    'koi_srad': 'R☉'
                }
                
                # Crear tabla de análisis
                analysis_data = []
                for i, feature_name in enumerate(saved_feature_names):
                    display_name = feature_display_names.get(feature_name, feature_name)
                    units = feature_units.get(feature_name, '')
                    value = features[i]
                    
                    analysis_data.append({
                        'Característica': display_name,
                        'Valor': f"{value} {units}",
                        'Código': feature_name
                    })
                
                analysis_df = pd.DataFrame(analysis_data)
                st.dataframe(analysis_df, use_container_width=True, hide_index=True)
                
                # Información adicional
                st.markdown("#### 💡 Información del Modelo")
                st.info(f"""
                - **Modelo:** Ensemble Stacking (4 algoritmos)
                - **Características:** {len(saved_feature_names)}
                - **Precisión:** ~83%
                - **Datos de entrenamiento:** Kepler + K2 + TESS (NASA)
                """)
                
        except Exception as e:
            st.error(f"❌ Error en la predicción: {e}")
            st.info("💡 **Solución:** Reentrena el modelo en la pestaña 'Entrenar Modelo REAL'")

    def render_real_analysis(self):
        """Análisis de datos REALES"""
        st.title("📊 Análisis de Datos REALES NASA")
        
        try:
            # Cargar datos reales
            kepler_df, k2_df, tess_df = self.data_processor.load_real_data()
            
            if kepler_df is None:
                st.warning("No se pudieron cargar los datasets para análisis")
                return
            
            st.success(f"✅ Datasets cargados: Kepler ({len(kepler_df):,}), K2 ({len(k2_df):,}), TESS ({len(tess_df):,})")
            
            # Selector de dataset
            dataset_choice = st.selectbox("Seleccionar Dataset para Análisis:", 
                                        ["Kepler", "K2", "TESS"])
            
            if dataset_choice == "Kepler":
                df = kepler_df
                st.subheader("🔭 Dataset Kepler")
            elif dataset_choice == "K2":
                df = k2_df
                st.subheader("🛰️ Dataset K2")
            else:
                df = tess_df
                st.subheader("📡 Dataset TESS")
            
            # Mostrar información básica
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Registros", f"{len(df):,}")
            with col2:
                st.metric("Columnas", df.shape[1])
            with col3:
                missing = df.isnull().sum().sum()
                st.metric("Valores Missing", f"{missing:,}")
            
            # Vista previa de datos
            st.subheader("👀 Vista Previa de Datos")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Análisis de columnas
            st.subheader("📋 Columnas Disponibles")
            st.write(f"Total de columnas: {len(df.columns)}")
            st.write(list(df.columns))
            
        except Exception as e:
            st.error(f"Error en el análisis: {e}")

    def render_saved_models(self):
        """Gestión de modelos guardados - VERSIÓN MEJORADA"""
        st.title("💾 Modelos Guardados")
        
        models_dir = os.path.join(PROJECT_ROOT, 'models')
        
        # Crear la carpeta si no existe
        if not os.path.exists(models_dir):
            st.warning("📁 La carpeta de modelos no existe. Creándola...")
            os.makedirs(models_dir, exist_ok=True)
            st.success(f"✅ Carpeta creada: {models_dir}")
        
        # Verificar archivos en la carpeta
        try:
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        except FileNotFoundError:
            model_files = []
        
        if not model_files:
            st.info("""
            📭 **No hay modelos guardados**
            
            Los modelos aparecerán aquí después de:
            1. 🚀 Entrenar un modelo en la pestaña "Entrenar Modelo REAL"
            2. 💾 El modelo se guardará automáticamente en la carpeta `models/`
            
            **Archivos que se guardan:**
            - `real_ensemble_model.pkl` - Modelo ensemble principal
            - `data_processor.pkl` - Preprocesador de datos  
            - `feature_names.pkl` - Nombres de características
            """)
            
            # Mostrar estructura esperada
            st.subheader("📁 Estructura esperada:")
            st.code("""
            exoplanet-ai-detector/
            ├── models/
            │   ├── real_ensemble_model.pkl
            │   ├── data_processor.pkl  
            │   └── feature_names.pkl
            ├── data/
            │   └── raw/
            │       ├── kepler.csv
            │       ├── k2.csv
            │       └── tess.csv
            └── webapp/
                └── app.py
            """)
            return
        
        st.success(f"✅ Se encontraron {len(model_files)} modelos guardados")
        
        # Mostrar modelos disponibles
        st.subheader("📁 Modelos Disponibles")
        
        for model_file in model_files:
            file_path = os.path.join(models_dir, model_file)
            file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
            file_time = os.path.getmtime(file_path)
            from datetime import datetime
            file_date = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
            
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
            
            with col1:
                # Icono diferente según el tipo de archivo
                if "ensemble" in model_file:
                    icon = "🤖"
                elif "processor" in model_file:
                    icon = "🔧" 
                elif "feature" in model_file:
                    icon = "📊"
                else:
                    icon = "📄"
                    
                st.write(f"{icon} **{model_file}**")
                st.caption(f"Creado: {file_date}")
                
            with col2:
                st.write(f"{file_size:.1f} MB")
                
            with col3:
                # Botón de información
                if st.button("ℹ️", key=f"info_{model_file}", help="Ver información"):
                    self._show_model_info(model_file, file_path)
                    
            with col4:
                # Botón de carga
                if st.button("📥", key=f"load_{model_file}", help="Cargar modelo"):
                    if self._load_specific_model(model_file):
                        st.success(f"✅ {model_file} cargado")
                        self.model_trained = True
                    else:
                        st.error(f"❌ Error cargando {model_file}")
                        
            with col5:
                # Botón de eliminación con confirmación
                if st.button("🗑️", key=f"delete_{model_file}", help="Eliminar modelo"):
                    if st.checkbox(f"¿Confirmar eliminación de {model_file}?", key=f"confirm_{model_file}"):
                        try:
                            os.remove(file_path)
                            st.success(f"✅ {model_file} eliminado")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error eliminando: {e}")
        
        # Estadísticas de la carpeta
        st.subheader("📈 Estadísticas de Modelos")
        total_size = sum(os.path.getsize(os.path.join(models_dir, f)) for f in model_files) / 1024 / 1024
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Modelos", len(model_files))
        with col2:
            st.metric("Espacio Total", f"{total_size:.1f} MB")
        with col3:
            st.metric("Carpeta", "models/")
        
        # Información de uso
        st.subheader("ℹ️ Información de Archivos")
        st.info("""
        **Archivos del sistema:**
        - 🤖 **real_ensemble_model.pkl** - Modelo ensemble principal (Random Forest + XGBoost + LightGBM + Extra Trees)
        - 🔧 **data_processor.pkl** - Preprocesador con scaler y configuración de características
        - 📊 **feature_names.pkl** - Nombres y orden de las características usadas en el entrenamiento
        
        **Recomendaciones:**
        - No elimines archivos manualmente desde el sistema de archivos
        - Usa los botones de esta interfaz para gestión segura
        - Los 3 archivos deben estar presentes para que el sistema funcione correctamente
        """)
        
        # Botón para crear modelo de ejemplo (para testing)
        st.subheader("🛠️ Herramientas")
        if st.button("🧪 Crear Modelo de Ejemplo", help="Crear un modelo dummy para testing"):
            self._create_example_model()
            
        if st.button("🔄 Actualizar Lista", help="Refrescar la lista de modelos"):
            st.rerun()

    def _show_model_info(self, model_file, file_path):
        """Mostrar información detallada de un modelo"""
        try:
            if "ensemble" in model_file:
                model = joblib.load(file_path)
                st.info(f"""
                **🤖 Modelo Ensemble: {model_file}**
                
                - **Tipo:** StackingClassifier
                - **Algoritmos base:** {len(model.named_estimators_)}
                - **Estimadores:** {list(model.named_estimators_.keys())}
                - **Meta-estimador:** {type(model.final_estimator_).__name__}
                """)
                
            elif "processor" in model_file:
                processor = joblib.load(file_path)
                st.info(f"""
                **🔧 Preprocesador: {model_file}**
                
                - **Tipo:** ExoplanetDataProcessor
                - **Características escaladas:** {len(processor.feature_names) if hasattr(processor, 'feature_names') else 'N/A'}
                - **Scaler:** {type(processor.scaler).__name__ if hasattr(processor, 'scaler') else 'N/A'}
                """)
                
            elif "feature" in model_file:
                features = joblib.load(file_path)
                st.info(f"""
                **📊 Características: {model_file}**
                
                - **Número de características:** {len(features)}
                - **Características:** {features}
                """)
                
        except Exception as e:
            st.error(f"❌ Error cargando información de {model_file}: {e}")

    def _load_specific_model(self, model_file):
        """Cargar un modelo específico"""
        try:
            model_path = os.path.join(PROJECT_ROOT, 'models', model_file)
            
            if "ensemble" in model_file:
                return self.model.load_model(model_path)
            else:
                st.info(f"📥 {model_file} cargado (no es el modelo principal)")
                return True
                
        except Exception as e:
            st.error(f"Error cargando {model_file}: {e}")
            return False

    def _create_example_model(self):
        """Crear un modelo de ejemplo para testing"""
        try:
            models_dir = os.path.join(PROJECT_ROOT, 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Crear feature names de ejemplo
            feature_names = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 
                            'koi_teq', 'koi_insol', 'koi_steff', 'koi_slogg', 'koi_srad']
            joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))
            
            # Crear processor de ejemplo
            from sklearn.preprocessing import StandardScaler
            class ExampleProcessor:
                def __init__(self):
                    self.scaler = StandardScaler()
                    self.feature_names = feature_names
            processor = ExampleProcessor()
            joblib.dump(processor, os.path.join(models_dir, 'data_processor.pkl'))
            
            st.success("✅ Modelo de ejemplo creado para testing")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error creando modelo de ejemplo: {e}")

    def render_batch_classification(self):
        """Clasificación por lotes de archivos CSV completos"""
        st.title("📦 Clasificación por Lotes")
        
        st.markdown("""
        ### 🚀 Clasificación Masiva de Exoplanetas
        
        **Sube un archivo CSV completo** (como Kepler, K2 o TESS) y el sistema:
        - ✅ Clasificará automáticamente todos los candidatos
        - 📊 Mostrará estadísticas completas
        - 🔍 Identificará exoplanetas detectados
        - 💾 Permitirá descargar resultados
        
        **Formatos compatibles:** Kepler, K2, TESS o cualquier CSV con las 9 características requeridas
        """)
        
        # Verificar si hay modelo entrenado
        model_path = os.path.join(PROJECT_ROOT, 'models', 'real_ensemble_model.pkl')
        if not os.path.exists(model_path):
            st.error("""
            ❌ **No hay modelo entrenado**
            
            Para usar la clasificación por lotes:
            1. Ve a la pestaña **'Entrenar Modelo REAL'**
            2. Entrena el modelo con tus datos de la NASA
            3. Regresa aquí para clasificar archivos completos
            """)
            return
        
        # Cargar modelo si no está cargado
        if not self.model_trained:
            if self.model.load_model(model_path):
                self.model_trained = True
                st.success("✅ Modelo REAL cargado exitosamente")
            else:
                st.error("❌ Error cargando el modelo")
                return
        
        # Sección de carga de archivos
        st.subheader("📤 Cargar Archivo CSV")
        
        uploaded_file = st.file_uploader(
            "Selecciona un archivo CSV para clasificar", 
            type=['csv'],
            help="Archivos compatibles: Kepler, K2, TESS o cualquier CSV con las características requeridas"
        )
        
        if uploaded_file is not None:
            try:
                # Leer el archivo
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Archivo cargado: {uploaded_file.name}")
                st.info(f"📊 Datos: {df.shape[0]} filas, {df.shape[1]} columnas")
                
                # Mostrar vista previa
                with st.expander("👀 Vista previa del archivo cargado"):
                    st.dataframe(df.head(10), use_container_width=True)
                    st.write(f"**Columnas disponibles:** {list(df.columns)}")
                
                # Procesar el archivo
                if st.button("🎯 Ejecutar Clasificación Masiva", type="primary"):
                    with st.spinner("🔍 Clasificando candidatos... Esto puede tomar unos segundos"):
                        results = self._batch_predict(df, uploaded_file.name)
                        
                        if results is not None:
                            self._display_batch_results(results, df)
                            
            except Exception as e:
                st.error(f"❌ Error procesando el archivo: {e}")

    def _batch_predict(self, df, filename):
        """Realizar predicciones por lotes"""
        try:
            # Cargar preprocesador y feature names
            processor_path = os.path.join(PROJECT_ROOT, 'models', 'data_processor.pkl')
            features_path = os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl')
            
            if not os.path.exists(processor_path) or not os.path.exists(features_path):
                st.error("❌ No se encontraron los archivos del modelo entrenado")
                return None
            
            saved_feature_names = joblib.load(features_path)
            data_processor = joblib.load(processor_path)
            
            st.info(f"🔍 Modelo espera {len(saved_feature_names)} características: {saved_feature_names}")
            
            # Verificar que tenemos las características necesarias
            missing_features = [f for f in saved_feature_names if f not in df.columns]
            if missing_features:
                st.error(f"❌ Faltan características en el archivo: {missing_features}")
                st.info("💡 **Solución:** Asegúrate de que el CSV tenga las mismas columnas que los datos de entrenamiento")
                return None
            
            # Seleccionar y preparar características
            X = df[saved_feature_names].copy()
            
            # Manejar valores missing
            missing_before = X.isnull().sum().sum()
            if missing_before > 0:
                st.warning(f"⚠️ Se encontraron {missing_before} valores missing. Imputando con medianas...")
                X = X.fillna(X.median())
            
            # Escalar características
            X_scaled = data_processor.scaler.transform(X)
            
            # Realizar predicciones
            predictions = self.model.model.predict(X_scaled)
            probabilities = self.model.model.predict_proba(X_scaled)[:, 1]
            
            # Crear DataFrame de resultados
            results_df = df.copy()
            results_df['prediction'] = predictions
            results_df['probability'] = probabilities
            results_df['classification'] = results_df['prediction'].map({1: 'EXOPLANETA', 0: 'NO_EXOPLANETA'})
            
            # Añadir confianza
            results_df['confidence'] = results_df['probability'].apply(
                lambda x: 'ALTA' if x >= 0.8 else 'MEDIA' if x >= 0.6 else 'BAJA'
            )
            
            return results_df
            
        except Exception as e:
            st.error(f"❌ Error en clasificación por lotes: {e}")
            return None

    def _display_batch_results(self, results_df, original_df):
        """Mostrar resultados de clasificación por lotes"""
        st.success("✅ Clasificación completada exitosamente!")
        
        # Estadísticas generales
        total_candidates = len(results_df)
        exoplanets_detected = results_df['prediction'].sum()
        non_exoplanets = total_candidates - exoplanets_detected
        
        st.subheader("📈 Resumen de Clasificación")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Candidatos", total_candidates)
        with col2:
            st.metric("Exoplanetas Detectados", exoplanets_detected)
        with col3:
            st.metric("No Exoplanetas", non_exoplanets)
        with col4:
            detection_rate = (exoplanets_detected / total_candidates) * 100
            st.metric("Tasa de Detección", f"{detection_rate:.1f}%")
        
        # Distribución de confianza
        confidence_counts = results_df['confidence'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de distribución
            fig = px.pie(
                results_df, 
                names='classification',
                title='Distribución: Exoplanetas vs No Exoplanetas',
                color='classification',
                color_discrete_map={'EXOPLANETA': '#00CC96', 'NO_EXOPLANETA': '#EF553B'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gráfico de confianza
            fig = px.bar(
                confidence_counts,
                title='Distribución por Nivel de Confianza',
                labels={'index': 'Confianza', 'value': 'Cantidad'},
                color=confidence_counts.index,
                color_discrete_map={'ALTA': '#00CC96', 'MEDIA': '#FECB52', 'BAJA': '#EF553B'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar exoplanetas detectados
        st.subheader("🔍 Exoplanetas Detectados")
        
        exoplanets_df = results_df[results_df['prediction'] == 1].copy()
        
        if len(exoplanets_df) > 0:
            st.success(f"🎯 Se encontraron {len(exoplanets_df)} exoplanetas potenciales")
            
            # Ordenar por probabilidad (mayor a menor)
            exoplanets_df = exoplanets_df.sort_values('probability', ascending=False)
            
            # Seleccionar columnas importantes para mostrar
            display_columns = []
            possible_columns = [
                'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition',
                'koi_period', 'koi_prad', 'koi_teq', 'probability', 'confidence'
            ]
            
            for col in possible_columns:
                if col in exoplanets_df.columns:
                    display_columns.append(col)
            
            # Añadir columnas de resultado si no están
            if 'probability' not in display_columns:
                display_columns.extend(['probability', 'confidence'])
            
            # Mostrar tabla de exoplanetas
            st.dataframe(
                exoplanets_df[display_columns].head(50),  # Mostrar máximo 50
                use_container_width=True,
                height=400
            )
            
            # Estadísticas de los exoplanetas detectados
            st.subheader("📊 Estadísticas de Exoplanetas Detectados")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_probability = exoplanets_df['probability'].mean()
                st.metric("Probabilidad Promedio", f"{avg_probability:.1%}")
            
            with col2:
                high_confidence = len(exoplanets_df[exoplanets_df['confidence'] == 'ALTA'])
                st.metric("Alta Confianza", high_confidence)
            
            with col3:
                if 'koi_prad' in exoplanets_df.columns:
                    avg_radius = exoplanets_df['koi_prad'].mean()
                    st.metric("Radio Promedio", f"{avg_radius:.1f} R⊕")
            
            # Distribución de características importantes
            if 'koi_prad' in exoplanets_df.columns and 'koi_period' in exoplanets_df.columns:
                st.subheader("📈 Características de Exoplanetas Detectados")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        exoplanets_df,
                        x='koi_prad',
                        title='Distribución de Radios Planetarios',
                        labels={'koi_prad': 'Radio (Radios Tierra)'},
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        exoplanets_df,
                        x='koi_period',
                        y='probability',
                        color='confidence',
                        title='Período Orbital vs Probabilidad',
                        labels={'koi_period': 'Período Orbital (días)', 'probability': 'Probabilidad'},
                        color_discrete_map={'ALTA': '#00CC96', 'MEDIA': '#FECB52', 'BAJA': '#EF553B'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Descargar resultados
            st.subheader("💾 Descargar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Descargar solo exoplanetas
                csv_exoplanets = exoplanets_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Exoplanetas Detectados",
                    data=csv_exoplanets,
                    file_name=f"exoplanetas_detectados_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    help="Descargar solo los candidatos clasificados como exoplanetas"
                )
            
            with col2:
                # Descargar todos los resultados
                csv_all = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Todos los Resultados",
                    data=csv_all,
                    file_name=f"clasificacion_completa_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    help="Descargar todos los candidatos con sus clasificaciones"
                )
            
            # Información adicional
            st.info("""
            💡 **Interpretación de resultados:**
            - **ALTA confianza:** Probabilidad ≥ 80% - Muy probable exoplaneta
            - **MEDIA confianza:** Probabilidad 60-79% - Posible exoplaneta  
            - **BAJA confianza:** Probabilidad < 60% - Requiere más análisis
            """)
            
        else:
            st.warning("⚠️ No se detectaron exoplanetas en este archivo")
            
            # Mostrar algunos candidatos con mayor probabilidad
            top_candidates = results_df.nlargest(5, 'probability')
            if len(top_candidates) > 0:
                st.subheader("🎯 Candidatos Más Prometedores")
                st.dataframe(
                    top_candidates[['probability', 'confidence'] + 
                                [col for col in top_candidates.columns if col.startswith('koi_')][:5]],
                    use_container_width=True
                )

    def _get_sample_datasets_info(self):
        """Información sobre datasets de ejemplo"""
        st.subheader("🛠️ Datasets de Prueba")
        
        st.markdown("""
        **Puedes probar con estos datasets de ejemplo:**
        
        - **Kepler:** [Descargar de NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
        - **K2:** [Descargar de NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2targets)
        - **TESS:** [Descargar de NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=toi)
        
        **Estructura mínima requerida:**
        ```csv
        koi_period,koi_duration,koi_depth,koi_prad,koi_teq,koi_insol,koi_steff,koi_slogg,koi_srad
        10.5,3.2,1500,2.1,450,95.0,5800,4.4,1.0
        15.3,4.1,800,1.8,320,75.5,5200,4.5,0.9
        ```
        """)

    def run(self):
        """Ejecutar la aplicación completa - ACTUALIZADO"""
        page = self.render_sidebar()
        
        if page == "🏠 Inicio":
            self.render_home()
        elif page == "🚀 Entrenar Modelo REAL":
            self.render_real_training()
        elif page == "🤖 Clasificar Exoplanetas":
            self.render_real_classification()
        elif page == "📦 Clasificación por Lotes":  # ¡NUEVA PÁGINA!
            self.render_batch_classification()
        elif page == "📊 Análisis de Datos REAL":
            self.render_real_analysis()
        elif page == "💾 Modelos Guardados":
            self.render_saved_models()

# Ejecutar la aplicación
if __name__ == "__main__":
    app = ExoplanetDetectorApp()
    app.run()