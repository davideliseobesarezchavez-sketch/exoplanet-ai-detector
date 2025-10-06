# Placeholder for data_preprocessing.py
# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class ExoplanetDataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.feature_names = []
        
    def load_and_unify_datasets(self, kepler_path, k2_path, tess_path):
        """Cargar y unificar los tres datasets de la NASA"""
        print("📥 Cargando datasets de la NASA...")
        
        # Cargar datos
        kepler_df = pd.read_csv(kepler_path)
        k2_df = pd.read_csv(k2_path) 
        tess_df = pd.read_csv(tess_path)
        
        # Procesar cada dataset
        kepler_processed = self._process_kepler(kepler_df)
        k2_processed = self._process_k2(k2_df)
        tess_processed = self._process_tess(tess_df)
        
        # Combinar
        unified_df = pd.concat([kepler_processed, k2_processed, tess_processed], 
                              ignore_index=True)
        
        print(f"✅ Datasets unificados: {unified_df.shape}")
        return unified_df
    
    def _process_kepler(self, df):
        """Procesar datos Kepler basado en el paper"""
        df_clean = df.copy()
        
        # 1. Eliminar columnas según paper (identificadores no útiles)
        columns_to_drop = [
            'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 
            'koi_score', 'rowid', 'koi_teq_err1', 'koi_teq_err2'
        ]
        df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])
        
        # 2. Filtrar solo CANDIDATE y CONFIRMED (eliminar FALSE POSITIVE)
        df_clean = df_clean[df_clean['koi_disposition'].isin(['CANDIDATE', 'CONFIRMED'])]
        
        # 3. Transformar target a binario (1 = exoplaneta, 0 = no exoplaneta)
        df_clean['target'] = df_clean['koi_disposition'].map({
            'CONFIRMED': 1, 
            'CANDIDATE': 1,
            'FALSE POSITIVE': 0
        })
        
        # 4. Añadir identificador de misión
        df_clean['mission'] = 'kepler'
        
        # 5. Renombrar columnas clave para unificación
        column_mapping = {
            'koi_period': 'orbital_period',
            'koi_duration': 'transit_duration', 
            'koi_depth': 'transit_depth',
            'koi_prad': 'planet_radius',
            'koi_teq': 'equilibrium_temp',
            'koi_insol': 'insolation_flux',
            'koi_steff': 'stellar_teff',
            'koi_slogg': 'stellar_logg',
            'koi_srad': 'stellar_radius',
            'koi_kepmag': 'magnitude'
        }
        df_clean = df_clean.rename(columns=column_mapping)
        
        return df_clean
    
    def _process_k2(self, df):
        """Procesar datos K2"""
        df_clean = df.copy()
        
        # Filtrar solo confirmed y candidate
        df_clean = df_clean[df_clean['disposition'].isin(['CONFIRMED', 'CANDIDATE'])]
        
        # Target binario
        df_clean['target'] = df_clean['disposition'].map({
            'CONFIRMED': 1,
            'CANDIDATE': 1
        })
        
        # Identificador de misión
        df_clean['mission'] = 'k2'
        
        # Renombrar columnas
        column_mapping = {
            'pl_orbper': 'orbital_period',
            'pl_trandurh': 'transit_duration',
            'pl_rade': 'planet_radius', 
            'pl_insol': 'insolation_flux',
            'pl_eqt': 'equilibrium_temp',
            'st_teff': 'stellar_teff',
            'st_logg': 'stellar_logg',
            'st_rad': 'stellar_radius',
            'st_mass': 'stellar_mass',
            'sy_vmag': 'magnitude'
        }
        df_clean = df_clean.rename(columns=column_mapping)
        
        return df_clean
    
    def _process_tess(self, df):
        """Procesar datos TESS"""
        df_clean = df.copy()
        
        # Mapear disposiciones de TESS
        disposition_mapping = {
            'PC': 1, 'KP': 1, 'APC': 1,  # Positivos
            'FP': 0, 'FA': 0, 'CP': 0     # Negativos
        }
        
        df_clean['target'] = df_clean['tfopwg_disp'].map(disposition_mapping)
        df_clean = df_clean.dropna(subset=['target'])
        
        # Identificador de misión
        df_clean['mission'] = 'tess'
        
        # Renombrar columnas
        column_mapping = {
            'pl_orbper': 'orbital_period',
            'pl_trandurh': 'transit_duration',
            'pl_trandep': 'transit_depth',
            'pl_rade': 'planet_radius',
            'pl_insol': 'insolation_flux', 
            'pl_eqt': 'equilibrium_temp',
            'st_teff': 'stellar_teff',
            'st_logg': 'stellar_logg',
            'st_rad': 'stellar_radius',
            'st_tmag': 'magnitude'
        }
        df_clean = df_clean.rename(columns=column_mapping)
        
        return df_clean
    
    def engineer_features(self, df):
        """Ingeniería de características avanzada basada en dominio astronómico"""
        df_eng = df.copy()
        
        # 1. Razones y proporciones físicas clave
        df_eng['period_duration_ratio'] = df_eng['orbital_period'] / df_eng['transit_duration']
        df_eng['depth_duration_ratio'] = df_eng['transit_depth'] / (df_eng['transit_duration'] + 1e-8)
        df_eng['radius_flux_ratio'] = df_eng['planet_radius'] / (df_eng['insolation_flux'] + 1e-8)
        
        # 2. Características estelares avanzadas
        df_eng['teff_logg_ratio'] = df_eng['stellar_teff'] / (df_eng['stellar_logg'] + 1e-8)
        
        # 3. Señal-ruido estimada (como en el paper)
        df_eng['estimated_snr'] = df_eng['transit_depth'] / np.sqrt(df_eng['transit_duration'] + 1e-8)
        
        # 4. Zona habitable estimada (simplificada)
        df_eng['habitable_zone_estimate'] = np.sqrt(df_eng['stellar_teff'] / 5772) * (df_eng['insolation_flux'] ** -0.5)
        
        # 5. Binning de período orbital (característica categórica)
        df_eng['period_category'] = pd.cut(
            df_eng['orbital_period'],
            bins=[0, 10, 50, 100, 365, np.inf],
            labels=['ultra_short', 'short', 'medium', 'long', 'very_long']
        )
        
        return df_eng
    
    def prepare_final_dataset(self, df):
        """Preparar dataset final para entrenamiento"""
        # Seleccionar características numéricas
        numeric_features = [
            'orbital_period', 'transit_duration', 'transit_depth', 'planet_radius',
            'equilibrium_temp', 'insolation_flux', 'stellar_teff', 'stellar_logg',
            'stellar_radius', 'magnitude', 'period_duration_ratio', 'depth_duration_ratio',
            'radius_flux_ratio', 'teff_logg_ratio', 'estimated_snr', 'habitable_zone_estimate'
        ]
        
        # Filtrar características disponibles
        available_features = [f for f in numeric_features if f in df.columns]
        X = df[available_features].copy()
        y = df['target'].values
        missions = df['mission'].values
        
        # Imputar valores missing
        for feature in available_features:
            if X[feature].isnull().any():
                imputer = SimpleImputer(strategy='median')
                X[feature] = imputer.fit_transform(X[[feature]]).ravel()
                self.imputers[feature] = imputer
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['main'] = scaler
        self.feature_names = available_features
        
        return X_scaled, y, missions, available_features