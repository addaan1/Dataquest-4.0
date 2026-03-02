import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import re
import os
import warnings
from joblib import parallel_backend
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import StackingRegressor
try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except Exception:
    _CATBOOST_AVAILABLE = False
warnings.filterwarnings('ignore')
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True  
except ImportError:
    _ST_AVAILABLE = False 
try:
    from transformers import AutoModel, AutoTokenizer
    _HF_AVAILABLE = True  # Set to True if transformers library is available
except ImportError:
    _HF_AVAILABLE = False  # Set to False if transformers library is not available 

class SentencePredictionModel:
    def __init__(self, base_dir="c:/Users/adief/OneDrive/Dokumen/Lomba/Dataquest 4.0/"):
        self.base_dir = base_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # File paths
        self.train_path = os.path.join(base_dir, "train.csv")
        self.test_path = os.path.join(base_dir, "test.csv")
        self.features_path = os.path.join(base_dir, "preprocessed_extracted_features2.csv")
        self.preprocessed_dir = os.path.join(base_dir, "file_putusan_preprocessed")
        self.original_dir = os.path.join(base_dir, "file_putusan")
        self.sample_submission_path = os.path.join(base_dir, "sample_submission.csv")
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        # Control flag: keep NN off to avoid instability and memory spikes
        self.use_nn = False
        # Embedding components
        self.tfidf_vectorizer = None
        self.svd = None
    def load_data(self):
        """Load and merge all necessary data"""
        print("Loading data...")
        
        # Load train and test data
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        
        # Load extracted features if available
        if os.path.exists(self.features_path):
            print("Loading extracted features...")
            self.features_df = pd.read_csv(self.features_path)
            # Pastikan tipe data selaras untuk merge dan numerik
            if 'fine_amount_value' in self.features_df.columns:
                self.features_df['fine_amount_value'] = pd.to_numeric(
                    self.features_df['fine_amount_value'], errors='coerce'
                ).fillna(0.0)
            # Samakan tipe ID agar merge aman
            self.train_df['id'] = self.train_df['id'].astype(str)
            self.test_df['id'] = self.test_df['id'].astype(str)
            if 'doc_id' in self.features_df.columns:
                self.features_df['doc_id'] = self.features_df['doc_id'].astype(str)
        else:
            print("Features file not found, creating basic features...")
            self.features_df = self._create_basic_features()
            # Samakan tipe ID setelah pembuatan fitur dasar
            self.train_df['id'] = self.train_df['id'].astype(str)
            self.test_df['id'] = self.test_df['id'].astype(str)
            self.features_df['doc_id'] = self.features_df['doc_id'].astype(str)
        
        # Merge features dengan train/test
        self.train_merged = self.train_df.merge(self.features_df, left_on='id', right_on='doc_id', how='left')
        self.test_merged = self.test_df.merge(self.features_df, left_on='id', right_on='doc_id', how='left')
        
        print(f"Train data shape: {self.train_merged.shape}")
        print(f"Test data shape: {self.test_merged.shape}")
        
        def _init_text_preprocessor(self):
            # Stopwords ringkas Bahasa Indonesia (daftar inti untuk noise reduction)
            self._ind_stopwords = {
                'yang','dan','di','ke','dari','dengan','untuk','pada','adalah','atau','itu','ini','karena','bahwa',
                'agar','dalam','sebagai','oleh','akan','tidak','bukan','kini','para','terhadap','atas','jika',
                'saat','telah','sudah','harus','dapat','juga','lebih','kurang','setelah','sebelum','kembali',
                'maka','kepada','antara','bagi','pun','serta','atau','sehingga','hingga','oleh','tentang','suatu'
            }
            self._re_url = re.compile(r'(http|www)\S+')
            self._re_non_alnum = re.compile(r'[^a-z0-9\s.,;:!?-]+')
            self._re_multispace = re.compile(r'\s+')

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = text.lower()
        t = self._re_url.sub(" ", t)
        t = self._re_non_alnum.sub(" ", t)
        # Token filter: hapus stopwords dan token pendek
        tokens = [w for w in t.split() if (len(w) > 2 and w not in self._ind_stopwords)]
        t = " ".join(tokens)
        t = self._re_multispace.sub(" ", t).strip()
        return t

    def _ensure_st_model(self) -> bool:
        if not self.use_sentence_transformer or not _ST_AVAILABLE:
            return False
        if self.st_model is None:
            try:
                self.st_model = SentenceTransformer(self.st_model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
                print(f"Loaded SentenceTransformer: {self.st_model_name}")
            except Exception as e:
                print(f"Failed to load SentenceTransformer: {e}")
                self.use_sentence_transformer = False
                return False
        return True
        
    def _create_basic_features(self):
        """Create basic features if extracted features are not available"""
        print("Creating basic features from original files...")
        
        # Get all document IDs from train and test
        all_ids = list(self.train_df['id']) + list(self.test_df['id'])
        
        features_data = []
        for doc_id in all_ids:
            # Check preprocessed directory first
            preprocessed_file = os.path.join(self.preprocessed_dir, f"{doc_id}.txt")
            original_file = os.path.join(self.original_dir, f"{doc_id}.txt")
            
            text_content = ""
            if os.path.exists(preprocessed_file):
                with open(preprocessed_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            elif os.path.exists(original_file):
                with open(original_file, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            
            # Extract basic features from text
            features = self._extract_basic_features_from_text(doc_id, text_content)
            features_data.append(features)
            
        return pd.DataFrame(features_data)
    
    def _extract_basic_features_from_text(self, doc_id, text):
        """Extract basic features from text content"""
        text_lower = text.lower()
        
        # Basic cooperation indicators
        cooperation_keywords = ['mengaku', 'menyesal', 'berjanji', 'kooperatif', 'sopan']
        cooperation_score = sum(1 for keyword in cooperation_keywords if keyword in text_lower)
        
        # Basic fine indicators
        fine_pattern = r'denda.*?rp\.?\s*([\d.,]+)'
        fine_matches = re.findall(fine_pattern, text_lower)
        fine_amount = 0
        if fine_matches:
            try:
                fine_str = fine_matches[0].replace('.', '').replace(',', '')
                fine_amount = float(fine_str)
            except:
                fine_amount = 0
        
        # Basic aggravating/mitigating factors
        mitigating_keywords = ['meringankan', 'belum pernah dihukum', 'perdamaian']
        aggravating_keywords = ['memberatkan', 'pernah dihukum', 'merugikan']
        
        mitigating_score = sum(1 for keyword in mitigating_keywords if keyword in text_lower)
        aggravating_score = sum(1 for keyword in aggravating_keywords if keyword in text_lower)
        
        return {
            'doc_id': doc_id,
            'cooperation_status': 'cooperative' if cooperation_score > 0 else 'unknown',
            'cooperation_evidence': '',
            'fine_payment_status': 'unknown',
            'fine_subsidiary_clause_present': 'subsidair' in text_lower,
            'fine_evidence': '',
            'fine_amount': f'rp.{fine_amount}' if fine_amount > 0 else '',
            'raw_fine_amount': f'rp.{fine_amount}' if fine_amount > 0 else '',
            'fine_amount_evidence': '',
            'mitigating_reasons': '',
            'aggravating_reasons': '',
            'behavioral_impact': 'mitigating' if mitigating_score > aggravating_score else 'aggravating' if aggravating_score > mitigating_score else 'none',
            'extracted_key_points_text': text[:500]  # First 500 chars
        }
    
    def feature_engineering(self):
        """Convert categorical features to numerical and create new features"""
        print("Performing feature engineering...")
        
        # Fill missing values
        self.train_merged = self.train_merged.fillna('')
        self.test_merged = self.test_merged.fillna('')
        
        # Gunakan fine_amount_value bila tersedia (sudah numerik), fallback ke parsing fine_amount
        if 'fine_amount_value' in self.train_merged.columns and 'fine_amount_value' in self.test_merged.columns:
            self.train_merged['fine_amount_numeric'] = pd.to_numeric(
                self.train_merged['fine_amount_value'], errors='coerce'
            ).fillna(0.0)
            self.test_merged['fine_amount_numeric'] = pd.to_numeric(
                self.test_merged['fine_amount_value'], errors='coerce'
            ).fillna(0.0)
        else:
            self.train_merged['fine_amount_numeric'] = self.train_merged['fine_amount'].apply(self._extract_fine_numeric)
            self.test_merged['fine_amount_numeric'] = self.test_merged['fine_amount'].apply(self._extract_fine_numeric)
        
        # Transformasi log jumlah denda
        self.train_merged['fine_amount_log'] = np.log1p(self.train_merged['fine_amount_numeric'])
        self.test_merged['fine_amount_log'] = np.log1p(self.test_merged['fine_amount_numeric'])
        
        # Tambahkan flag untuk klausul subsidair denda
        def _to_bool_flag(v):
            s = str(v).strip().lower()
            if s in ('true', '1', 'ya', 'yes'):
                return 1
            if s in ('false', '0', 'tidak', 'no'):
                return 0
            # Jika sudah boolean True/False
            if v is True:
                return 1
            if v is False:
                return 0
            return 0
        
        if 'fine_subsidiary_clause_present' in self.train_merged.columns:
            self.train_merged['fine_subsidiary_flag'] = self.train_merged['fine_subsidiary_clause_present'].apply(_to_bool_flag)
        else:
            self.train_merged['fine_subsidiary_flag'] = 0
        
        if 'fine_subsidiary_clause_present' in self.test_merged.columns:
            self.test_merged['fine_subsidiary_flag'] = self.test_merged['fine_subsidiary_clause_present'].apply(_to_bool_flag)
        else:
            self.test_merged['fine_subsidiary_flag'] = 0
        
        # Mapping status kerja sama sesuai CSV
        cooperation_mapping = {'cooperative': 1, 'mixed': 0.5, 'not_cooperative': -1, 'uncooperative': -1, 'unknown': 0}
        self.train_merged['cooperation_score'] = self.train_merged['cooperation_status'].map(cooperation_mapping).fillna(0)
        self.test_merged['cooperation_score'] = self.test_merged['cooperation_status'].map(cooperation_mapping).fillna(0)
        
        # Mapping status pembayaran denda
        fine_payment_mapping = {'paid': 1, 'not_paid': -1, 'unknown': 0}
        self.train_merged['fine_payment_score'] = self.train_merged['fine_payment_status'].map(fine_payment_mapping).fillna(0)
        self.test_merged['fine_payment_score'] = self.test_merged['fine_payment_status'].map(fine_payment_mapping).fillna(0)
        
        # Mapping dampak perilaku
        behavioral_mapping = {'mitigating': 1, 'aggravating': -1, 'both': 0, 'none': 0}
        self.train_merged['behavioral_score'] = self.train_merged['behavioral_impact'].map(behavioral_mapping).fillna(0)
        self.test_merged['behavioral_score'] = self.test_merged['behavioral_impact'].map(behavioral_mapping).fillna(0)
        
        # Fitur panjang teks
        self.train_merged['text_length'] = self.train_merged['extracted_key_points_text'].str.len()
        self.test_merged['text_length'] = self.test_merged['extracted_key_points_text'].str.len()
        
        # Fitur tekstual sederhana (tidak diubah)
        def _text_feats(s: str):
            s = s if isinstance(s, str) else ""
            wc = len(s.split())
            sc = s.count('.') + s.count('!') + s.count('?')
            uniq = len(set(s.split()))
            unique_ratio = (uniq / wc) if wc > 0 else 0.0
            avg_word_len = (sum(len(w) for w in s.split()) / wc) if wc > 0 else 0.0
            punct = s.count(',') + s.count(';') + s.count(':') + s.count('-')
            return pd.Series([wc, sc, unique_ratio, avg_word_len, punct])
        self.train_merged[['word_count','sentence_count','unique_ratio','avg_word_len','punct_count']] = \
            self.train_merged['extracted_key_points_text'].apply(_text_feats)
        self.test_merged[['word_count','sentence_count','unique_ratio','avg_word_len','punct_count']] = \
            self.test_merged['extracted_key_points_text'].apply(_text_feats)
        
        # Evidence count features (tetap, menyesuaikan delimiter '||')
        self.train_merged['cooperation_evidence_count'] = self.train_merged['cooperation_evidence'].str.count(r'\|\|')
        self.test_merged['cooperation_evidence_count'] = self.test_merged['cooperation_evidence'].str.count(r'\|\|')
        self.train_merged['mitigating_count'] = self.train_merged['mitigating_reasons'].str.count(r'\|\|')
        self.test_merged['mitigating_count'] = self.test_merged['mitigating_reasons'].str.count(r'\|\|')
        self.train_merged['aggravating_count'] = self.train_merged['aggravating_reasons'].str.count(r'\|\|')
        self.test_merged['aggravating_count'] = self.test_merged['aggravating_reasons'].str.count(r'\|\|')
        
        # Kalkulasi faktor penyesuaian hukuman
        self.train_merged['sentence_adjustment'] = self._calculate_sentence_adjustment(self.train_merged)
        self.test_merged['sentence_adjustment'] = self._calculate_sentence_adjustment(self.test_merged)
        
        # Pilih fitur akhir (ditambah fine_subsidiary_flag dan memakai fine_amount_numeric dari file)
        self.feature_columns = [
            'cooperation_score', 'fine_payment_score', 'behavioral_score',
            'fine_amount_numeric', 'fine_amount_log', 'fine_subsidiary_flag',
            'text_length', 'word_count','sentence_count','unique_ratio','avg_word_len','punct_count',
            'cooperation_evidence_count', 'mitigating_count', 'aggravating_count',
            'sentence_adjustment'
        ]
        
        print(f"Selected features: {self.feature_columns}")
        
    def _extract_fine_numeric(self, fine_str):
        """Extract numeric value from fine amount string"""
        if pd.isna(fine_str) or fine_str == '':
            return 0
        
        # Remove 'rp.' and extract numbers
        numbers = re.findall(r'[\d.,]+', str(fine_str))
        if numbers:
            try:
                # Take the first number found and clean it
                num_str = numbers[0].replace('.', '').replace(',', '')
                return float(num_str)
            except:
                return 0
        return 0
    
    def _calculate_sentence_adjustment(self, df):
        """Calculate sentence adjustment factor based on judge-like reasoning.
        Negative values lower the sentence, positive values increase it.
        """
        adjustment = np.zeros(len(df), dtype=float)
        
        # Cooperation should REDUCE sentence (e.g., -10%)
        adjustment -= df['cooperation_score'].astype(float) * 0.10
        
        # Large unpaid fines should INCREASE sentence (e.g., +15%)
        large_fine_threshold = 100_000_000  # 100 million rupiah
        large_unpaid_fine = ((df['fine_amount_numeric'] > large_fine_threshold) & (df['fine_payment_score'] <= 0)).astype(float)
        adjustment += large_unpaid_fine * 0.15
        
        # Behavioral impact: mitigating reduces, aggravating increases via sign of behavioral_score
        adjustment -= df['behavioral_score'].astype(float) * 0.05
        
        # Evidence strength: more mitigating vs aggravating should reduce
        evidence_strength = (df['cooperation_evidence_count'] + df['mitigating_count'] - df['aggravating_count']).astype(float) / 10.0
        adjustment -= evidence_strength * 0.02
        
        # Clip to keep adjustments moderate
        adjustment = np.clip(adjustment, -0.30, 0.30)
        return adjustment
    
    def prepare_model_data(self):
        """Prepare data for model training with log-target for stability and TF-IDF SVD embeddings"""
        print("Preparing model data...")
        
        # Get features and target
        X_num_all = self.train_merged[self.feature_columns].fillna(0)
        y_train_raw_series = self.train_merged['lama hukuman (bulan)'].astype(float)
        X_num_test = self.test_merged[self.feature_columns].fillna(0)
        
        texts_all = self.train_merged['extracted_key_points_text'].astype(str).fillna('')
        texts_test = self.test_merged['extracted_key_points_text'].astype(str).fillna('')
        
        # Persist raw target stats for clipping
        self.target_min = float(np.nanmin(y_train_raw_series.values))
        self.target_p99 = float(np.nanpercentile(y_train_raw_series.values, 99))
        
        # Hapus scaling penuh (ini menyebabkan leakage), lakukan split dulu
        y_all_log = np.log1p(y_train_raw_series.values)
        y_all_raw = y_train_raw_series.values
        
        # Split indices for train/val to avoid leakage
        idx = np.arange(len(X_num_all))
        train_idx, val_idx, y_train_log, y_val_log, y_train_raw, y_val_raw = train_test_split(
            idx, y_all_log, y_all_raw, test_size=0.2, random_state=42
        )
        
        # Fit scaler HANYA pada numeric train
        X_num_train = X_num_all.iloc[train_idx]
        X_num_val = X_num_all.iloc[val_idx]
        self.scaler = StandardScaler()
        X_num_train_scaled = self.scaler.fit_transform(X_num_train)
        X_num_val_scaled = self.scaler.transform(X_num_val)
        X_num_test_scaled = self.scaler.transform(X_num_test)
        
        # Fit TF-IDF pada teks training saja, lalu SVD
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            lowercase=True
        )
        tfidf_train = self.tfidf_vectorizer.fit_transform(texts_all.iloc[train_idx])
        tfidf_val = self.tfidf_vectorizer.transform(texts_all.iloc[val_idx])
        tfidf_test = self.tfidf_vectorizer.transform(texts_test)
        
        self.svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
        svd_train = self.svd.fit_transform(tfidf_train)
        svd_val = self.svd.transform(tfidf_val)
        svd_test = self.svd.transform(tfidf_test)
        
        # Combine numeric and text embeddings
        X_train_combined = np.hstack([X_num_train_scaled, svd_train])
        X_val_combined = np.hstack([X_num_val_scaled, svd_val])
        X_test_combined = np.hstack([X_num_test_scaled, svd_test])
        
        # Persist
        self.X_train = X_train_combined
        self.X_val = X_val_combined
        self.X_test = X_test_combined
        self.y_train = y_train_log
        self.y_val = y_val_log
        self.y_train_raw = y_train_raw
        self.y_val_raw = y_val_raw
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_val.shape}")
        print(f"Test set shape: {self.X_test.shape}")
    
    def train_ensemble_model(self):
        """Train ensemble of models with log-target, evaluate on raw scale, and compute dynamic weights"""
        print("Training ensemble models...")
        
        self.models = {}
        self.predictions = {}
        self.val_metrics = {}
        
        # XGBoost (light, regularized)
        print("Training XGBoost...")
        xgb_params = {
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist',
            'gpu_id': 0 if torch.cuda.is_available() else None,
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
        }
        self.models['xgb'] = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50,)
        self.models['xgb'].fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )
        
        # LightGBM (light, robust)
        print("Training LightGBM...")
        lgb_params = {
            "objective": "regression_l1",  
            "metric": "mae",
            "learning_rate": 0.05,
            "n_estimators": 1500,
            "num_leaves": 63,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,   # L1
            "reg_lambda": 0.5,  # L2
            "random_state": 42,
            "device": "gpu" if torch.cuda.is_available() else "cpu",
        }
        self.models['lgb'] = lgb.LGBMRegressor(**lgb_params)
        self.models['lgb'].fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        # Random Forest (on log target)
        print("Training Random Forest...")
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.models['rf'].fit(self.X_train, self.y_train)
        
        # Gradient Boosting (on log target)
        print("Training Gradient Boosting...")
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.8,
            random_state=42,
        )
        self.models['gb'].fit(self.X_train, self.y_train)
        
        # CatBoost (minimal yet effective hyperparameters)
        if _CATBOOST_AVAILABLE:
            print("Training CatBoost...")
            cat_params = {
                'loss_function': 'RMSE',
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3.0,
                'iterations': 1000,
                'random_seed': 42,
                'verbose': False,
            }
            if torch.cuda.is_available():
                cat_params['task_type'] = 'GPU'
            self.models['cat'] = CatBoostRegressor(**cat_params)
            # Early stopping via eval set
            self.models['cat'].fit(self.X_train, self.y_train, eval_set=(self.X_val, self.y_val), use_best_model=True, verbose=False)
        else:
            print("CatBoost not available. Skipping CatBoost model.")
        
        # Optional: Neural Network disabled by default to keep predictions calibrated
        if self.use_nn and torch.cuda.is_available():
            print("Training Neural Network on GPU...")
            self.models['nn'] = self._train_neural_network()
        
        # Stacking Regressor (combine base learners)
        print("Training Stacking Regressor...")
        base_estimators = []
        base_estimators.append(('xgb', xgb.XGBRegressor(**xgb_params)))
        base_estimators.append(('lgb', lgb.LGBMRegressor(**lgb_params)))
        base_estimators.append(('rf', RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)))
        base_estimators.append(('gb', GradientBoostingRegressor(n_estimators=250, learning_rate=0.08, random_state=42)))
        if _CATBOOST_AVAILABLE:
            cb_base_params = {
                'loss_function': 'RMSE',
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 3.0,
                'iterations': 600,
                'random_seed': 42,
                'verbose': False,
            }
            if torch.cuda.is_available():
                cb_base_params['task_type'] = 'GPU'
            base_estimators.append(('cat', CatBoostRegressor(**cb_base_params)))
        meta = RidgeCV(alphas=(0.1, 1.0, 10.0))
        self.models['stack'] = StackingRegressor(estimators=base_estimators, final_estimator=meta, n_jobs=1, passthrough=False)
        self.models['stack'].fit(self.X_train, self.y_train)
        
        with parallel_backend('threading', n_jobs=1):
            self.models['stack'].fit(self.X_train, self.y_train)
        
        # Evaluate models on validation (convert back from log)
        print("\nModel Performance on Validation Set (raw months):")
        rmse_for_weight = {}
        for name, model in self.models.items():
            if name == 'nn':
                val_pred_log = self._predict_neural_network(self.X_val)
                val_pred_raw = val_pred_log  # already raw if NN returns raw
            else:
                val_pred_log = model.predict(self.X_val)
                val_pred_raw = np.expm1(val_pred_log)
            
            mae = mean_absolute_error(self.y_val_raw, val_pred_raw)
            rmse = np.sqrt(mean_squared_error(self.y_val_raw, val_pred_raw))
            print(f"{name.upper()}: MAE={mae:.2f}, RMSE={rmse:.2f}")
            self.val_metrics[name] = {'MAE': mae, 'RMSE': rmse}
            rmse_for_weight[name] = rmse
            self.predictions[name] = val_pred_raw
        
        # Summary table sorted by RMSE
        try:
            summary_df = pd.DataFrame(self.val_metrics).T.sort_values('RMSE')
            print("\nValidation summary (sorted by RMSE):")
            print(summary_df)
        except Exception:
            pass
        
        # Dynamic inverse-RMSE weights (more accurate models get higher weight)
        inv = {k: (1.0 / (v + 1e-6)) for k, v in rmse_for_weight.items()}
        s = sum(inv.values())
        self.ensemble_weights = {k: (v / s) for k, v in inv.items()}
        print("Ensemble weights (inverse-RMSE):", self.ensemble_weights)
        
        def _raw_preds(model_key):
            # model dilatih pada target log -> konversi expm1 ke skala bulan
            val_log = self.models[model_key].predict(self.X_val)
            test_log = self.models[model_key].predict(self.X_test)
            return np.expm1(val_log), np.expm1(test_log)
        
        available_models = [k for k in self.models.keys()]  # mis: ['xgb','lgb','rf','gb','cat', 'stack'?]
        
        val_preds = {}
        test_preds = {}
        for mk in available_models:
            try:
                vp, tp = _raw_preds(mk)
                val_preds[mk] = vp
                test_preds[mk] = tp
            except Exception as e:
                print(f"Skip model {mk} for blending due to error: {e}")
        
        # Kandidat blend: fokuskan ke base learners yang kuat; exclude 'stack' dari blending
        blend_candidates = [m for m in val_preds.keys() if m != 'stack']
        if len(blend_candidates) == 0:
            blend_candidates = list(val_preds.keys())
        
        P = np.column_stack([val_preds[m] for m in blend_candidates])
        y = self.y_val_raw.astype(float)
        
        # Least-squares non-neg
        try:
            w_hat, _, _, _ = np.linalg.lstsq(P, y, rcond=None)
            w_hat = np.nan_to_num(w_hat, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            w_hat = np.ones(len(blend_candidates)) / len(blend_candidates)
        w_nonneg = np.clip(w_hat, 0.0, None)
        w = w_nonneg / w_nonneg.sum() if w_nonneg.sum() > 0 else np.ones_like(w_nonneg)/len(w_nonneg)
        
        blend_val = P @ w
        blend_rmse = mean_squared_error(y, blend_val, squared=False)
        
        use_stack = False
        if 'stack' in val_preds:
            stack_val = val_preds['stack']
            stack_rmse = mean_squared_error(y, stack_val, squared=False)
            if stack_rmse + 1e-6 < blend_rmse:
                use_stack = True
        
        if use_stack:
            chosen_val = val_preds['stack']
            chosen_test = test_preds['stack']
            self.ensemble_method = "stack_only"
            self.ensemble_weights = {"stack": 1.0}
        else:
            chosen_val = blend_val
            chosen_test = np.column_stack([test_preds[m] for m in blend_candidates]) @ w
            self.ensemble_method = "blended_ls_nonneg"
            self.ensemble_weights = {m: float(w[i]) for i, m in enumerate(blend_candidates)}
        
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(chosen_val, y)
        calibrated_test = iso.transform(chosen_test)
        
        calibrated_test = np.clip(calibrated_test, self.target_min, self.target_p99)
        self.final_predictions = calibrated_test
        
        print("Optimized ensemble method:", self.ensemble_method)
        print("Optimized ensemble weights:", self.ensemble_weights)
        if not use_stack:
            print(f"Validation RMSE (blend): {blend_rmse:.4f}")
        else:
            print(f"Validation RMSE (stack): {stack_rmse:.4f}")
        
    def _train_neural_network(self):
        """Train neural network using PyTorch"""
        class SentenceNN(nn.Module):
            def __init__(self, input_size):
                super(SentenceNN, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Prepare data for PyTorch
        X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train.values).to(self.device)
        X_val_tensor = torch.FloatTensor(self.X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(self.y_val.values).to(self.device)
        
        # Initialize model
        model = SentenceNN(self.X_train.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        return model

    def _predict_neural_network(self, X):
        """Make predictions using neural network"""
        self.models['nn'].eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.models['nn'](X_tensor).cpu().numpy().squeeze()
        return predictions

    def make_predictions(self):
        """Make final predictions using dynamically weighted ensemble"""
        print("Making final predictions...")
        
        test_predictions = {}
        
        # Get predictions from each model (transform back from log)
        for name, model in self.models.items():
            if name == 'nn':
                preds_raw = self._predict_neural_network(self.X_test)
            else:
                preds_log = model.predict(self.X_test)
                preds_raw = np.expm1(preds_log)
            test_predictions[name] = preds_raw
        
        # Use dynamic weights if available; otherwise fallback
        if hasattr(self, 'ensemble_weights') and self.ensemble_weights:
            weights = self.ensemble_weights
        else:
            weights = {'xgb': 0.4, 'lgb': 0.3, 'rf': 0.2, 'gb': 0.1}
        
        # Calculate ensemble prediction (ensure weights sum to 1 over available models)
        available = {k: v for k, v in weights.items() if k in test_predictions}
        s = sum(available.values())
        norm_weights = {k: v / s for k, v in available.items()} if s > 0 else {k: 1.0 / len(available) for k in available}
        
        ensemble_pred = np.zeros(len(self.X_test), dtype=float)
        for name, weight in norm_weights.items():
            ensemble_pred += weight * test_predictions[name]
        
        # Apply calibrated business logic adjustments (bounded)
        base_sentences = ensemble_pred
        adjusted_sentences = base_sentences * (1.0 + self.test_merged['sentence_adjustment'].values)
        
        # Clip to realistic range based on training distribution
        min_clip = max(1.0, self.target_min)
        max_clip = max(min_clip + 1.0, self.target_p99)
        adjusted_sentences = np.clip(adjusted_sentences, min_clip, max_clip)
        
        # Ensure positive integers
        final_predictions = np.maximum(1, np.round(adjusted_sentences)).astype(int)
        
        return final_predictions

    def save_predictions(self, predictions):
        """Save predictions in the required format"""
        print("Saving predictions...")
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'id': self.test_df['id'],
            'lama hukuman (bulan)': predictions
        })
        
        # Save to file
        output_path = os.path.join(self.base_dir, "submission9.csv")
        submission.to_csv(output_path, index=False)
        
        print(f"Predictions saved to: {output_path}")
        print(f"Prediction statistics:")
        print(f"  Mean: {predictions.mean():.2f} months")
        print(f"  Median: {np.median(predictions):.2f} months")
        print(f"  Min: {predictions.min()} months")
        print(f"  Max: {predictions.max()} months")
        
        return submission

    def run_full_pipeline(self):
        """Run the complete prediction pipeline"""
        print("Starting sentence prediction pipeline...")
        print("="*50)
        
        try:
            # Load and prepare data
            self.load_data()
            self.feature_engineering()
            self.prepare_model_data()
            
            # Train models
            self.train_ensemble_model()
            
            # Make predictions
            predictions = self.make_predictions()
            
            # Save results
            submission = self.save_predictions(predictions)
            
            print("\n" + "="*50)
            print("Pipeline completed successfully!")
            
            return submission
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            raise
        

if __name__ == "__main__":
    # Initialize and run the model
    model = SentencePredictionModel()
    submission = model.run_full_pipeline()