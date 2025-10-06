# train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from src.data_preprocessing import ExoplanetDataPreprocessor
from src.ensemble_model import AdvancedExoplanetEnsemble

def main():
    print("🚀 INICIANDO ENTRENAMIENTO DEL MODELO EXOPLANET AI")
    print("=" * 60)
    
    # 1. Preprocesamiento de datos
    print("📥 Cargando y preprocesando datos...")
    preprocessor = ExoplanetDataPreprocessor()
    
    # Cargar datasets (ajusta las rutas según tus archivos)
    unified_data = preprocessor.load_and_unify_datasets(
        'data/raw/kepler.csv',
        'data/raw/k2.csv',
        'data/raw/tess.csv'
    )
    
    # Guardar dataset unificado
    os.makedirs('data/processed', exist_ok=True)
    unified_data.to_csv('data/processed/unified_dataset.csv', index=False)
    print(f"✅ Dataset unificado guardado: {unified_data.shape}")
    
    # 2. Ingeniería de características
    print("🔧 Aplicando ingeniería de características...")
    engineered_data = preprocessor.engineer_features(unified_data)
    
    # 3. Preparar datos finales
    print("⚙️ Preparando datos para entrenamiento...")
    X, y, missions, feature_names = preprocessor.prepare_final_dataset(engineered_data)
    
    print(f"📊 Datos finales: {X.shape}")
    print(f"🎯 Distribución de targets: {np.unique(y, return_counts=True)}")
    
    # 4. Entrenar modelo ensemble
    print("🤖 Entrenando modelo Ensemble avanzado...")
    ensemble_trainer = AdvancedExoplanetEnsemble()
    
    # Opcional: Optimizar hiperparámetros (quitar comentario para usar)
    # print("🎯 Optimizando hiperparámetros...")
    # best_params = ensemble_trainer.optimize_hyperparameters(X, y, n_trials=50)
    
    # Entrenar ensemble
    model = ensemble_trainer.train_ensemble(X, y, missions)
    
    # 5. Evaluar modelo
    print("📈 Evaluando modelo...")
    results = ensemble_trainer.evaluate_model(X, y, missions, feature_names)
    
    # 6. Guardar modelos
    print("💾 Guardando modelos...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'models/best_ensemble_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print(f"🎯 Accuracy final: {results['accuracy']:.4f}")
    print(f"🎯 F1-Score final: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()



    