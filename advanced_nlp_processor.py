
import pandas as pd
import numpy as np
import torch
import os
import warnings
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sentence_transformers import SentenceTransformer
import model_visualizer as mv

warnings.filterwarnings('ignore')

class AdvancedNLPProcessor:
    def __init__(self, base_dir="c:/Users/adief/OneDrive/Dokumen/Lomba/Dataquest 4.0/"):
        self.base_dir = base_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.train_path = os.path.join(base_dir, "train.csv")
        self.test_path = os.path.join(base_dir, "test.csv")
        self.features_path = os.path.join(base_dir, "preprocessed_extracted_features.csv")
        
        self.scaler = StandardScaler()
        self.sentence_model = None
        self.feature_columns = []
        self.nlp_feature_names = []

    def load_data(self):
        print("Loading data...")
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        
        if os.path.exists(self.features_path):
            self.features_df = pd.read_csv(self.features_path)
        else:
            raise FileNotFoundError(f"Features file not found at {self.features_path}")

        self.train_merged = self.train_df.merge(self.features_df, left_on='id', right_on='doc_id', how='left')
        self.test_merged = self.test_df.merge(self.features_df, left_on='id', right_on='doc_id', how='left')
        print("Data loading complete.")

    def feature_engineering(self):
        print("Performing feature engineering...")
        self.train_merged = self.train_merged.fillna('')
        self.test_merged = self.test_merged.fillna('')

        # Basic numerical features from the original script
        self.feature_columns = [
            'cooperation_score', 'fine_payment_score', 'behavioral_score',
            'fine_amount_numeric', 'text_length', 'mitigating_count', 'aggravating_count'
        ]
        
        # Re-create necessary columns if they don't exist
        self.train_merged['fine_amount_numeric'] = self.train_merged['fine_amount'].apply(self._extract_fine_numeric)
        self.test_merged['fine_amount_numeric'] = self.test_merged['fine_amount'].apply(self._extract_fine_numeric)
        
        cooperation_mapping = {'cooperative': 1, 'mixed': 0.5, 'not_cooperative': -1, 'uncooperative': -1, 'unknown': 0}
        self.train_merged['cooperation_score'] = self.train_merged['cooperation_status'].map(cooperation_mapping).fillna(0)
        self.test_merged['cooperation_score'] = self.test_merged['cooperation_status'].map(cooperation_mapping).fillna(0)

        fine_payment_mapping = {'paid': 1, 'not_paid': -1, 'unknown': 0}
        self.train_merged['fine_payment_score'] = self.train_merged['fine_payment_status'].map(fine_payment_mapping).fillna(0)
        self.test_merged['fine_payment_score'] = self.test_merged['fine_payment_status'].map(fine_payment_mapping).fillna(0)

        behavioral_mapping = {'mitigating': 1, 'aggravating': -1, 'both': 0, 'none': 0}
        self.train_merged['behavioral_score'] = self.train_merged['behavioral_impact'].map(behavioral_mapping).fillna(0)
        self.test_merged['behavioral_score'] = self.test_merged['behavioral_impact'].map(behavioral_mapping).fillna(0)

        self.train_merged['text_length'] = self.train_merged['extracted_key_points_text'].str.len()
        self.test_merged['text_length'] = self.test_merged['extracted_key_points_text'].str.len()

        self.train_merged['mitigating_count'] = self.train_merged['mitigating_reasons'].str.count('||')
        self.test_merged['mitigating_count'] = self.test_merged['mitigating_reasons'].str.count('||')

        self.train_merged['aggravating_count'] = self.train_merged['aggravating_reasons'].str.count('||')
        self.test_merged['aggravating_count'] = self.test_merged['aggravating_reasons'].str.count('||')

        print("Feature engineering complete.")

    def _extract_fine_numeric(self, fine_str):
        import re
        if pd.isna(fine_str) or str(fine_str).strip() == '':
            return 0
        # Remove all non-digit characters
        digits_only = re.sub(r'\D', '', str(fine_str))
        if digits_only:
            try:
                return float(digits_only)
            except ValueError:
                return 0
        return 0

    def prepare_model_data(self):
        print("Preparing model data with advanced NLP features...")
        
        # 1. Basic Numerical Features
        self.X_num_train = self.scaler.fit_transform(self.train_merged[self.feature_columns].fillna(0))
        self.X_num_test = self.scaler.transform(self.test_merged[self.feature_columns].fillna(0))

        # 2. Advanced NLP Features (Sentence Transformer)
        print("Generating sentence embeddings... (This may take a while)")
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=self.device)
        
        train_texts = self.train_merged['extracted_key_points_text'].astype(str).fillna('').tolist()
        test_texts = self.test_merged['extracted_key_points_text'].astype(str).fillna('').tolist()
        
        self.X_nlp_train = self.sentence_model.encode(train_texts, show_progress_bar=True, batch_size=32)
        self.X_nlp_test = self.sentence_model.encode(test_texts, show_progress_bar=True, batch_size=32)
        self.nlp_feature_names = [f'embed_{i}' for i in range(self.X_nlp_train.shape[1])]

        # 3. Combine Features
        self.X_combined = np.hstack([self.X_num_train, self.X_nlp_train])
        self.X_test_combined = np.hstack([self.X_num_test, self.X_nlp_test])
        self.y = self.train_merged['lama hukuman (bulan)'].astype(float)
        self.feature_names = self.feature_columns + self.nlp_feature_names

        print(f"Combined training data shape: {self.X_combined.shape}")
        print("Model data preparation complete.")

    def train_and_evaluate(self):
        print("Training LightGBM model and evaluating...")
        
        X_train, X_val, y_train, y_val = train_test_split(self.X_combined, self.y, test_size=0.2, random_state=42)

        lgb_params = {
            'objective': 'regression_l1',
            'metric': 'rmse',
            'n_estimators': 2000,
            'learning_rate': 0.01,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'num_leaves': 31,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
            'boosting_type': 'gbdt',
        }

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])

        # Evaluation
        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Validation RMSE with Advanced NLP Features: {rmse:.4f}")

        # Generate Visualizations
        print("Generating visualizations...")
        mv.plot_error_analysis(y_val, val_preds, "LightGBM", os.path.join(self.base_dir, 'error_analysis.png'))
        mv.plot_feature_importance(model, self.feature_names, "LightGBM", os.path.join(self.base_dir, 'feature_importance.png'))
        
        # For learning curve, we use the full dataset for a more complete picture
        print("Generating learning curve... (This may take a while)")
        lc_model = lgb.LGBMRegressor(**lgb_params)
        mv.plot_learning_curve(lc_model, self.X_combined, self.y, "LightGBM", os.path.join(self.base_dir, 'learning_curve.png'))

    def run_pipeline(self):
        self.load_data()
        self.feature_engineering()
        self.prepare_model_data()
        self.train_and_evaluate()
        print("\nPipeline finished successfully!")

if __name__ == "__main__":
    processor = AdvancedNLPProcessor()
    processor.run_pipeline()
