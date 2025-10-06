# src/ensemble_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import optuna
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

class AdvancedExoplanetEnsemble:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def create_ensemble(self):
        """Crear ensemble stacking basado en el paper optimizado"""
        
        # Modelos base (optimizados según paper)
        base_models = [
            ('random_forest', RandomForestClassifier(
                n_estimators=1000,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )),
            ('extra_trees', ExtraTreesClassifier(
                n_estimators=800,
                max_depth=25,
                min_samples_split=3,
                random_state=42
            )),
            ('xgboost', XGBClassifier(
                n_estimators=900,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                random_state=42,
                eval_metric='logloss'
            )),
            ('lightgbm', LGBMClassifier(
                n_estimators=850,
                learning_rate=0.05,
                max_depth=10,
                num_leaves=31,
                random_state=42
            ))
        ]
        
        # Meta-model (como en el paper)
        meta_model = LogisticRegressionCV(
            cv=5,
            random_state=42,
            max_iter=1000
        )
        
        # Ensemble Stacking (mejor performance según paper)
        stacking_model = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            passthrough=True,
            n_jobs=-1
        )
        
        self.models['stacking'] = stacking_model
        return stacking_model
    
    def optimize_hyperparameters(self, X, y, n_trials=100):
        """Optimización avanzada con Optuna"""
        
        def objective(trial):
            model_name = trial.suggest_categorical('model', ['rf', 'xgb', 'lgbm'])
            
            if model_name == 'rf':
                n_estimators = trial.suggest_int('rf_n_estimators', 500, 1500)
                max_depth = trial.suggest_int('rf_max_depth', 10, 30)
                min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 10)
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42,
                    n_jobs=-1
                )
                
            elif model_name == 'xgb':
                n_estimators = trial.suggest_int('xgb_n_estimators', 500, 1200)
                learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3)
                max_depth = trial.suggest_int('xgb_max_depth', 3, 12)
                
                model = XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=42,
                    eval_metric='logloss'
                )
                
            else:  # lgbm
                n_estimators = trial.suggest_int('lgbm_n_estimators', 500, 1200)
                learning_rate = trial.suggest_float('lgbm_learning_rate', 0.01, 0.2)
                num_leaves = trial.suggest_int('lgbm_num_leaves', 20, 50)
                
                model = LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    random_state=42
                )
            
            # Validación cruzada
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
            
            return scores.mean()
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"🎯 Mejores hiperparámetros: {study.best_params}")
        print(f"🎯 Mejor F1-score: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_ensemble(self, X, y, missions):
        """Entrenar el ensemble final"""
        print("🚀 Entrenando ensemble avanzado...")
        
        # Crear ensemble
        ensemble_model = self.create_ensemble()
        
        # Entrenar con validación por misión
        mission_groups = missions  # Para GroupKFold
        
        # Fit del modelo
        ensemble_model.fit(X, y)
        
        self.best_model = ensemble_model
        
        # Calcular importancia de características (promedio de modelos base)
        self._calculate_feature_importance(ensemble_model, X.shape[1])
        
        print("✅ Ensemble entrenado exitosamente!")
        return ensemble_model
    
    def _calculate_feature_importance(self, ensemble_model, n_features):
        """Calcular importancia de características promediada"""
        importances = np.zeros(n_features)
        
        for name, model in ensemble_model.named_estimators_.items():
            if hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_
        
        self.feature_importance = importances / len(ensemble_model.named_estimators_)
    
    def evaluate_model(self, X, y, missions, feature_names):
        """Evaluación comprehensiva del modelo"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Predicciones
        y_pred = self.best_model.predict(X)
        y_pred_proba = self.best_model.predict_proba(X)[:, 1]
        
        # Métricas generales
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        print("📊 EVALUACIÓN DEL MODELO")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}") 
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\n" + classification_report(y, y_pred))
        
        # Evaluación por misión
        self._evaluate_per_mission(y, y_pred, missions)
        
        # Importancia de características
        self._plot_feature_importance(feature_names)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1, 
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def _evaluate_per_mission(self, y_true, y_pred, missions):
        """Evaluar performance por misión individual"""
        print("\n🔬 EVALUACIÓN POR MISIÓN:")
        print("-" * 30)
        
        for mission in np.unique(missions):
            mask = missions == mission
            mission_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            mission_f1 = f1_score(y_true[mask], y_pred[mask])
            
            print(f"{mission.upper():<10} | Accuracy: {mission_accuracy:.4f} | F1: {mission_f1:.4f}")
    
    def _plot_feature_importance(self, feature_names):
        """Visualizar importancia de características"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Importancia de Características - Ensemble Promediado')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("📈 Importancia de características (top 10):")
            for idx in np.argsort(self.feature_importance)[-10:][::-1]:
                print(f"  {feature_names[idx]}: {self.feature_importance[idx]:.4f}")