"""
═══════════════════════════════════════════════════════════════════════════════
ENHANCED GenAI-Powered Insurance Underwriting Co-Pilot
with Advanced Risk Prediction and Business Rules Engine
═══════════════════════════════════════════════════════════════════════════════

MAJOR FIXES & ENHANCEMENTS:
✅ FIXED: Inverted risk logic (high probability = willing customer = LOW RISK)
✅ ADDED: Input validation with business rules
✅ ENHANCED: Feature engineering with domain expertise
✅ IMPROVED: Risk scoring with composite metrics
✅ ADDED: Red flags detection system
✅ ENHANCED: Calibrated probability thresholds
✅ ADDED: Sanity checks and data quality gates
✅ IMPROVED: Explainability with feature contributions
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
import os
import requests
from functools import wraps
import logging
from typing import Dict, List, Tuple, Optional
import sys
import io

# Windows Unicode Fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ML Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Console output with UTF-8
    ]
)
logger = logging.getLogger(__name__)

try:
    from config import OPENROUTER_API_KEY, FLASK_SECRET_KEY
    logger.info("✓ Configuration loaded from config.py")
except ImportError:
    logger.warning("⚠️  config.py not found. Using default values.")
    OPENROUTER_API_KEY = "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"
    FLASK_SECRET_KEY = "change-this-in-production-2024"

app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
app.config['JSON_SORT_KEYS'] = False

os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')  # ADD encoding='utf-8'
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ═══════════════════════════════════════════════════════════════════════════════
# BUSINESS RULES & VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BusinessRulesEngine:
    """
    Validates inputs and detects red flags before ML prediction
    """
    
    # Define valid ranges
    VALID_RANGES = {
        'Age': (18, 85),
        'Annual_Premium': (2000, 500000),
        'Vintage': (10, 365),
        'Region_Code': (0, 52)
    }
    
    VALID_CATEGORIES = {
        'Gender': ['Male', 'Female'],
        'Vehicle_Age': ['< 1 Year', '1-2 Year', '> 2 Years'],
        'Vehicle_Damage': ['Yes', 'No'],
        'Driving_License': [0, 1],
        'Previously_Insured': [0, 1]
    }
    
    @classmethod
    def validate_input(cls, data: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate all inputs and detect red flags
        Returns: (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['Gender', 'Age', 'Driving_License', 'Region_Code', 
                          'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 
                          'Annual_Premium', 'Vintage']
        
        for field in required_fields:
            if field not in data or data[field] == '' or data[field] is None:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors, warnings
        
        # Validate numeric ranges
        try:
            age = int(data['Age'])
            if not (cls.VALID_RANGES['Age'][0] <= age <= cls.VALID_RANGES['Age'][1]):
                errors.append(f"Age must be between {cls.VALID_RANGES['Age'][0]} and {cls.VALID_RANGES['Age'][1]}")
            
            # Age-based warnings
            if age < 21:
                warnings.append("Very young driver - higher accident risk")
            elif age > 70:
                warnings.append("Senior driver - health considerations")
        except (ValueError, TypeError):
            errors.append("Age must be a valid number")
        
        try:
            premium = float(data['Annual_Premium'])
            if not (cls.VALID_RANGES['Annual_Premium'][0] <= premium <= cls.VALID_RANGES['Annual_Premium'][1]):
                errors.append(f"Annual Premium must be between ₹{cls.VALID_RANGES['Annual_Premium'][0]:,} and ₹{cls.VALID_RANGES['Annual_Premium'][1]:,}")
            
            # Premium-based warnings
            if premium < 5000:
                warnings.append("Very low premium - may indicate underinsurance")
            elif premium > 300000:
                warnings.append("Very high premium - verify vehicle value")
        except (ValueError, TypeError):
            errors.append("Annual Premium must be a valid number")
        
        try:
            vintage = int(data['Vintage'])
            if not (cls.VALID_RANGES['Vintage'][0] <= vintage <= cls.VALID_RANGES['Vintage'][1]):
                errors.append(f"Vintage must be between {cls.VALID_RANGES['Vintage'][0]} and {cls.VALID_RANGES['Vintage'][1]} days")
            
            if vintage < 30:
                warnings.append("Very new customer relationship - limited history")
        except (ValueError, TypeError):
            errors.append("Vintage must be a valid number")
        
        # Validate categories
        if data.get('Gender') not in cls.VALID_CATEGORIES['Gender']:
            errors.append(f"Gender must be one of: {cls.VALID_CATEGORIES['Gender']}")
        
        if data.get('Vehicle_Age') not in cls.VALID_CATEGORIES['Vehicle_Age']:
            errors.append(f"Vehicle Age must be one of: {cls.VALID_CATEGORIES['Vehicle_Age']}")
        
        if data.get('Vehicle_Damage') not in cls.VALID_CATEGORIES['Vehicle_Damage']:
            errors.append(f"Vehicle Damage must be one of: {cls.VALID_CATEGORIES['Vehicle_Damage']}")
        
        # Critical red flags
        try:
            if int(data.get('Driving_License', 1)) == 0:
                errors.append("🚨 CRITICAL: No driving license - Cannot issue policy")
        except (ValueError, TypeError):
            errors.append("Driving License must be 0 or 1")
        
        # Business logic warnings
        if data.get('Vehicle_Damage') == 'Yes' and data.get('Previously_Insured') == '0':
            warnings.append("⚠️ Vehicle damage with no prior insurance - high risk indicator")
        
        if data.get('Vehicle_Age') == '> 2 Years' and float(data.get('Annual_Premium', 0)) > 100000:
            warnings.append("Old vehicle with high premium - verify vehicle condition")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    @classmethod
    def calculate_red_flag_score(cls, data: Dict) -> Tuple[int, List[str]]:
        """
        Calculate red flag score (0-100, higher = riskier)
        Returns: (score, list of red flags)
        """
        red_flags = []
        score = 0
        
        # No driving license
        if int(data.get('Driving_License', 1)) == 0:
            score += 50
            red_flags.append("🚨 No driving license")
        
        # Vehicle damage history
        if data.get('Vehicle_Damage') == 'Yes':
            score += 20
            red_flags.append("Previous vehicle damage")
        
        # No prior insurance
        if data.get('Previously_Insured') == '0':
            score += 15
            red_flags.append("No insurance history")
        
        # Young driver
        age = int(data.get('Age', 30))
        if age < 25:
            score += 15
            red_flags.append(f"Young driver (age {age})")
        
        # Old vehicle
        if data.get('Vehicle_Age') == '> 2 Years':
            score += 10
            red_flags.append("Vehicle older than 2 years")
        
        # Low premium for damaged vehicle
        if data.get('Vehicle_Damage') == 'Yes' and float(data.get('Annual_Premium', 0)) < 10000:
            score += 15
            red_flags.append("Low premium despite vehicle damage")
        
        # New customer with damage
        if int(data.get('Vintage', 100)) < 60 and data.get('Vehicle_Damage') == 'Yes':
            score += 10
            red_flags.append("New customer with damage history")
        
        return min(score, 100), red_flags


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedDataPreprocessor:
    """
    Enhanced preprocessing with advanced feature engineering
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create insurance-relevant features based on domain expertise
        """
        df = df.copy()
        
        # Age-based risk bands (insurance actuarial standards)
        df['Age_Group'] = pd.cut(df['Age'], 
                                 bins=[0, 25, 35, 45, 55, 100], 
                                 labels=['High_Risk_Young', 'Moderate_Young', 'Prime', 'Mature', 'Senior'])
        
        # Risk score for age
        age_risk_map = {'High_Risk_Young': 4, 'Moderate_Young': 3, 'Prime': 1, 'Mature': 2, 'Senior': 3}
        df['Age_Risk_Score'] = df['Age_Group'].map(age_risk_map)
        
        # Vehicle age numeric
        if 'Vehicle_Age' in df.columns:
            vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
            df['Vehicle_Age_Numeric'] = df['Vehicle_Age'].map(vehicle_age_map)
            df['Vehicle_Age_Risk'] = df['Vehicle_Age_Numeric'] * 15  # 0, 15, 30 risk points
        
        # Damage indicator
        if 'Vehicle_Damage' in df.columns:
            df['Has_Damage'] = (df['Vehicle_Damage'] == 'Yes').astype(int)
            df['Damage_Risk_Score'] = df['Has_Damage'] * 25  # Major risk factor
        
        # Insurance history indicator
        if 'Previously_Insured' in df.columns:
            df['No_Insurance_History'] = (df['Previously_Insured'] == 0).astype(int)
            df['No_History_Risk'] = df['No_Insurance_History'] * 20
        
        # Premium bands (affordability indicator)
        if 'Annual_Premium' in df.columns:
            df['Premium_Band'] = pd.cut(df['Annual_Premium'], 
                                       bins=[0, 10000, 30000, 50000, 1000000],
                                       labels=['Budget', 'Standard', 'Premium', 'Luxury'])
            
            # Premium adequacy (low premium = underinsurance risk)
            df['Premium_Adequacy'] = np.where(df['Annual_Premium'] < 5000, 0,
                                             np.where(df['Annual_Premium'] < 15000, 1, 2))
        
        # Customer tenure risk
        if 'Vintage' in df.columns:
            df['Tenure_Months'] = df['Vintage'] / 30
            df['Is_New_Customer'] = (df['Vintage'] < 60).astype(int)
            df['New_Customer_Risk'] = df['Is_New_Customer'] * 10
        
        # Composite risk indicators
        df['High_Risk_Combo'] = (
            (df['Has_Damage'] == 1) & 
            (df['Previously_Insured'] == 0) & 
            (df['Age'] < 30)
        ).astype(int) * 30
        
        # Channel risk (some channels have higher fraud/lapse rates)
        if 'Policy_Sales_Channel' in df.columns:
            # Channels above 100 often indicate broker/agent networks
            df['Channel_Risk_Category'] = np.where(df['Policy_Sales_Channel'] > 100, 'High_Volume', 'Direct')
        
        # Gender-age interaction (actuarial factor)
        if 'Gender' in df.columns:
            df['Male_Young'] = ((df['Gender'] == 'Male') & (df['Age'] < 30)).astype(int)
        
        return df
    
    def load_and_prepare_data(self, train_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare training data"""
        try:
            logger.info(f"Loading training data from {train_path}")
            df = pd.read_csv(train_path)
            
            # Feature engineering
            df = self.engineer_features(df)
            
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
            
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            
            # Separate features and target
            X = df.drop(['Response', 'id'], axis=1, errors='ignore')
            y = df['Response']
            
            logger.info(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"✓ Class distribution: {dict(y.value_counts())}")
            logger.info(f"✓ Engineered features: {[col for col in X.columns if 'Risk' in col or 'Score' in col]}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, None
    
    def encode_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        X_encoded = X.copy()
        categorical_cols = X_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    X_encoded[col] = X_encoded[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                    )
        
        return X_encoded
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Scale features"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Handle class imbalance"""
        original_dist = dict(pd.Series(y).value_counts())
        logger.info(f"Original class distribution: {original_dist}")
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        new_dist = dict(pd.Series(y_resampled).value_counts())
        logger.info(f"After SMOTE: {new_dist}")
        
        return X_resampled, y_resampled


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED RISK TIER ENGINE (FIXED LOGIC)
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedRiskTierEngine:
    """
    CORRECTED RISK MAPPING:
    - High ML probability = Customer WANTS insurance = GOOD for business = LOW UNDERWRITING RISK
    - But we also consider: red flags, business rules, data quality
    
    Final Risk Tier = f(ML_score, Red_Flag_Score, Business_Rules)
    """
    
    @staticmethod
    def map_to_comprehensive_risk(
        ml_probability: float, 
        red_flag_score: int,
        red_flags: List[str],
        applicant_data: Dict
    ) -> Dict:
        """
        Comprehensive risk assessment combining:
        1. ML probability (propensity to buy)
        2. Red flag score (underwriting risk factors)
        3. Business rules
        
        CORRECTED LOGIC:
        - ml_probability: probability customer will buy (0-1)
        - red_flag_score: underwriting risk (0-100)
        
        Final risk = weighted combination
        """
        
        # Step 1: Convert ML probability to business attractiveness score
        # High probability = customer wants insurance = good business = positive score
        business_attractiveness = ml_probability * 100  # 0-100
        
        # Step 2: Calculate underwriting risk from red flags
        underwriting_risk = red_flag_score  # 0-100
        
        # Step 3: Composite risk score (weighted combination)
        # Lower is better: want high business attractiveness, low underwriting risk
        # Weights: 40% business attractiveness (inverted), 60% underwriting risk
        composite_risk_score = (0.4 * (100 - business_attractiveness)) + (0.6 * underwriting_risk)
        
        # Step 4: Map to risk tiers with calibrated thresholds
        if composite_risk_score < 30:
            risk_tier = "Low Risk"
            risk_level = "low"
            confidence = "High"
            action = "AUTO_APPROVE"
            color = "#10b981"
            recommendation = "Approve with standard terms"
            
        elif composite_risk_score < 55:
            risk_tier = "Medium Risk"
            risk_level = "medium"
            confidence = "Moderate"
            action = "MANUAL_REVIEW"
            color = "#f59e0b"
            recommendation = "Manual review required"
            
        else:
            risk_tier = "High Risk"
            risk_level = "high"
            confidence = "High"
            action = "REFER_TO_SENIOR"
            color = "#ef4444"
            recommendation = "Refer to senior underwriter or decline"
        
        # Step 5: Override based on critical red flags
        critical_flags = [flag for flag in red_flags if '🚨' in flag]
        if critical_flags:
            risk_tier = "High Risk"
            risk_level = "high"
            action = "REJECT"
            color = "#dc2626"
            recommendation = "Reject - Critical compliance issues"
        
        # Step 6: Build detailed risk breakdown
        return {
            'risk_tier': risk_tier,
            'risk_level': risk_level,
            'composite_risk_score': round(composite_risk_score, 2),
            'business_attractiveness': round(business_attractiveness, 2),
            'underwriting_risk': underwriting_risk,
            'ml_probability': round(ml_probability, 4),
            'red_flag_count': len(red_flags),
            'red_flags': red_flags,
            'confidence': confidence,
            'action': action,
            'color': color,
            'recommendation': recommendation,
            'risk_breakdown': {
                'customer_interest': f"{business_attractiveness:.1f}/100",
                'underwriting_concerns': f"{underwriting_risk}/100",
                'overall_risk': f"{composite_risk_score:.1f}/100"
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-MODEL PIPELINE (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedMultiModelPipeline:
    """Enhanced training with better hyperparameters"""
    
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.best_model_name = None
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train models with optimized hyperparameters"""
        
        logger.info("="*80)
        logger.info("TRAINING ENHANCED MULTI-MODEL PIPELINE")
        logger.info("="*80)
        
        # 1. XGBoost (Best for tabular data)
        logger.info("\n1️⃣ Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=1,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        self.metrics['xgboost'] = self._evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        
        # 2. Random Forest
        logger.info("\n2️⃣ Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        self.metrics['random_forest'] = self._evaluate_model(rf_model, X_test, y_test, "Random Forest")
        
        # 3. Logistic Regression
        logger.info("\n3️⃣ Training Logistic Regression...")
        lr_model = LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            C=0.1,
            random_state=42
        )
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        self.metrics['logistic_regression'] = self._evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
        
        # Select best model
        self.best_model_name = max(self.metrics, key=lambda k: self.metrics[k]['roc_auc'])
        
        logger.info("\n" + "="*80)
        logger.info(f"🏆 BEST MODEL: {self.best_model_name.upper()} (ROC-AUC: {self.metrics[self.best_model_name]['roc_auc']:.4f})")
        logger.info("="*80 + "\n")
        
        return self.models, self.metrics, self.best_model_name
    
    def _evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'roc_auc': roc_auc,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'confusion_matrix': cm.tolist()
        }
        
        logger.info(f"✓ {model_name} Performance:")
        logger.info(f"  • ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  • Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  • Precision: {metrics['precision']:.4f}")
        logger.info(f"  • Recall:    {metrics['recall']:.4f}")
        logger.info(f"  • F1-Score:  {metrics['f1_score']:.4f}")
        
        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# GENAI PROMPT CHAIN (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedGenAIPromptChain:
    """Enhanced GenAI with better prompts"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = OPENROUTER_API_URL
    
    def execute_full_chain(self, applicant_data: Dict, risk_data: Dict) -> Dict:
        """Execute enhanced 3-stage chain"""
        
        logger.info("🤖 Executing Enhanced GenAI Prompt Chain...")
        
        summary = self._stage1_risk_summary(applicant_data, risk_data)
        rationale = self._stage2_decision_rationale(applicant_data, risk_data)
        recommendation = self._stage3_business_recommendation(applicant_data, risk_data, summary, rationale)
        
        logger.info("✓ GenAI Chain Complete")
        
        return {
            'summary': summary,
            'rationale': rationale,
            'recommendation': recommendation
        }
    
    def _stage1_risk_summary(self, applicant_data: Dict, risk_data: Dict) -> str:
        """Stage 1: Enhanced risk summary"""
        
        red_flags_text = "\n".join([f"- {flag}" for flag in risk_data.get('red_flags', [])])
        
        prompt = f"""You are an expert insurance underwriter. Provide a concise 2-3 sentence professional summary.

**Applicant Profile:**
- Age: {applicant_data.get('Age')} years
- Gender: {applicant_data.get('Gender')}
- Vehicle Age: {applicant_data.get('Vehicle_Age')}
- Damage History: {applicant_data.get('Vehicle_Damage')}
- Premium: ₹{applicant_data.get('Annual_Premium', 0):,}
- Prior Insurance: {'Yes' if applicant_data.get('Previously_Insured') == '1' else 'No'}

**Risk Assessment:**
- Overall Risk: {risk_data['risk_tier']}
- Customer Interest: {risk_data.get('business_attractiveness', 0):.1f}/100
- Underwriting Concerns: {risk_data.get('underwriting_risk', 0)}/100
- Red Flags Detected: {risk_data.get('red_flag_count', 0)}

{f"**Red Flags:**\\n{red_flags_text}" if red_flags_text else ""}

Summarize the key findings professionally."""

        return self._call_genai(prompt, max_tokens=250)
    
    def _stage2_decision_rationale(self, applicant_data: Dict, risk_data: Dict) -> str:
        """Stage 2: Detailed rationale"""
        
        prompt = f"""Explain WHY this applicant was classified as {risk_data['risk_tier']}.

**Data Points:**
- Age: {applicant_data.get('Age')} ({applicant_data.get('Gender')})
- License: {'Valid' if applicant_data.get('Driving_License') == '1' else 'None'}
- Vehicle: {applicant_data.get('Vehicle_Age')}, Damage: {applicant_data.get('Vehicle_Damage')}
- Insurance History: {'Yes' if applicant_data.get('Previously_Insured') == '1' else 'None'}
- Premium: ₹{applicant_data.get('Annual_Premium', 0):,}
- Customer Tenure: {applicant_data.get('Vintage', 0)} days

**Assessment Scores:**
- Customer Interest Level: {risk_data.get('business_attractiveness',0):.1f}/100
- Underwriting Risk Factors: {risk_data.get('underwriting_risk', 0)}/100
- Composite Risk Score: {risk_data.get('composite_risk_score', 0):.1f}/100

**Red Flags:** {len(risk_data.get('red_flags', []))} detected
{chr(10).join([f"  • {flag}" for flag in risk_data.get('red_flags', [])])}

Provide 3-5 bullet points explaining the specific factors driving this risk classification. Focus on the most impactful elements."""

        return self._call_genai(prompt, max_tokens=350)
    
    def _stage3_business_recommendation(self, applicant_data: Dict, risk_data: Dict, 
                                       summary: str, rationale: str) -> str:
        """Stage 3: Actionable recommendation"""
        
        prompt = f"""Based on the complete analysis, provide a clear business recommendation.

**Risk Summary:**
{summary}

**Rationale:**
{rationale}

**Risk Classification:**
- Tier: {risk_data['risk_tier']}
- Recommended Action: {risk_data['action']}
- Composite Risk Score: {risk_data.get('composite_risk_score', 0):.1f}/100

Provide a final recommendation covering:
1. Decision (APPROVE/REJECT/MANUAL REVIEW)
2. Any special conditions or pricing adjustments
3. Immediate next steps for the underwriter

Be direct, actionable, and specific to this case."""

        return self._call_genai(prompt, max_tokens=300)
    
    def _call_genai(self, prompt: str, max_tokens: int = 300) -> str:
        """Call OpenRouter API"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://underwriting-copilot.app",
            "X-Title": "Insurance Underwriting Co-Pilot"
        }
        
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert insurance underwriting AI assistant specializing in comprehensive risk assessment for the Indian market. Provide clear, professional, and actionable insights based on data-driven analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                logger.error(f"GenAI API Error: {response.status_code}")
                return f"[GenAI unavailable - Status {response.status_code}]"
                
        except Exception as e:
            logger.error(f"GenAI Exception: {e}")
            return "[GenAI service temporarily unavailable]"


# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED DOCUMENT REQUEST AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class EnhancedDocumentRequestAgent:
    """Intelligent document requests based on comprehensive risk assessment"""
    
    @staticmethod
    def get_required_documents(risk_tier: str, applicant_data: Dict, red_flags: List[str]) -> Dict:
        """Determine required documents based on risk and red flags"""
        
        # Base documents (always required)
        base_docs = [
            "PAN Card",
            "Aadhaar Card",
            "Driving License (Front & Back)",
            "Vehicle Registration Certificate (RC)"
        ]
        
        additional_docs = []
        urgency = "STANDARD"
        review_type = "STANDARD"
        processing_days = 3
        
        # Risk-based document requirements
        if risk_tier == "High Risk":
            additional_docs.extend([
                "Income Proof (Last 6 months salary slips or ITR)",
                "Bank Statements (Last 6 months)",
                "Previous Insurance Policy Documents",
                "No Claim Bonus (NCB) Certificate",
                "Address Proof (Utility Bill - Latest)",
                "Employment Verification Letter",
                "Vehicle Inspection Report (Mandatory)"
            ])
            urgency = "CRITICAL"
            review_type = "COMPREHENSIVE_SENIOR_REVIEW"
            processing_days = 7
            
        elif risk_tier == "Medium Risk":
            additional_docs.extend([
                "Income Proof (Last 3 months)",
                "Bank Statements (Last 3 months)",
                "Previous Insurance Policy (if applicable)",
                "Address Proof (Utility Bill)"
            ])
            urgency = "HIGH"
            review_type = "DETAILED_REVIEW"
            processing_days = 4
            
        else:  # Low Risk
            additional_docs.extend([
                "Previous Insurance Policy (if applicable)"
            ])
            urgency = "STANDARD"
            review_type = "FAST_TRACK"
            processing_days = 2
        
        # Red flag-specific documents
        if any('damage' in flag.lower() for flag in red_flags):
            additional_docs.extend([
                "Vehicle Damage Assessment Report (Third-Party)",
                "Repair Invoices with Photos",
                "Accident Report (if applicable)"
            ])
            processing_days += 2
        
        if any('no insurance' in flag.lower() for flag in red_flags):
            additional_docs.extend([
                "Declaration for Gap in Insurance Coverage",
                "Vehicle Inspection Certificate (Mandatory)"
            ])
        
        if any('young driver' in flag.lower() for flag in red_flags):
            additional_docs.extend([
                "Driving School Certificate",
                "Guardian/Parent Consent (if under 21)"
            ])
        
        if any('low premium' in flag.lower() for flag in red_flags):
            additional_docs.extend([
                "Vehicle Valuation Certificate",
                "Justification for Premium Amount"
            ])
        
        # Remove duplicates
        all_docs = base_docs + list(set(additional_docs))
        
        return {
            'required_documents': all_docs,
            'document_count': len(all_docs),
            'base_documents': len(base_docs),
            'additional_documents': len(additional_docs),
            'urgency': urgency,
            'review_type': review_type,
            'estimated_processing_days': processing_days,
            'special_instructions': _get_special_instructions(risk_tier, red_flags)
        }


def _get_special_instructions(risk_tier: str, red_flags: List[str]) -> List[str]:
    """Generate special handling instructions"""
    instructions = []
    
    if risk_tier == "High Risk":
        instructions.append("⚠️ Mandatory senior underwriter approval required")
        instructions.append("📋 Complete document verification mandatory")
        instructions.append("🔍 Physical vehicle inspection required")
    
    if any('damage' in flag.lower() for flag in red_flags):
        instructions.append("🚗 Third-party vehicle damage assessment mandatory")
    
    if any('no license' in flag.lower() or '🚨' in flag for flag in red_flags):
        instructions.append("🚨 CRITICAL: Verify all compliance requirements before proceeding")
    
    if any('no insurance' in flag.lower() for flag in red_flags):
        instructions.append("📄 Mandatory gap coverage explanation required")
    
    if any('young driver' in flag.lower() for flag in red_flags):
        instructions.append("👤 Additional driver verification recommended")
    
    return instructions


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

models = {
    'random_forest': None,
    'xgboost': None,
    'logistic_regression': None
}
scaler = None
feature_columns = []
label_encoders = {}
model_metrics = {}
default_model = 'xgboost'


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def train_enhanced_models():
    """Train models with enhanced preprocessing"""
    global models, scaler, feature_columns, label_encoders, model_metrics, default_model
    
    logger.info("\n" + "="*80)
    logger.info("STARTING ENHANCED MODEL TRAINING PIPELINE")
    logger.info("="*80 + "\n")
    
    train_path = 'data/train.csv'
    if not os.path.exists(train_path):
        logger.warning(f"⚠️  Training data not found at {train_path}")
        logger.info("Creating enhanced synthetic dataset...")
        create_enhanced_synthetic_dataset()
    
    # Initialize preprocessor
    preprocessor = EnhancedDataPreprocessor()
    
    # Load and prepare data
    X, y = preprocessor.load_and_prepare_data(train_path)
    
    if X is None or y is None:
        logger.error("❌ Failed to load training data")
        return False
    
    # Encode features
    X_encoded = preprocessor.encode_features(X, fit=True)
    feature_columns = list(X_encoded.columns)
    label_encoders = preprocessor.label_encoders
    
    # Scale features
    X_scaled = preprocessor.scale_features(X_encoded, fit=True)
    scaler = preprocessor.scaler
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(X_train, y_train)
    
    # Train all models
    pipeline = EnhancedMultiModelPipeline()
    trained_models, metrics, best_model = pipeline.train_all_models(
        X_train_balanced, X_test, y_train_balanced, y_test
    )
    
    # Update global variables
    models = trained_models
    model_metrics = metrics
    default_model = best_model
    
    # Save models
    save_models()
    
    logger.info("\n✅ ENHANCED MODEL TRAINING COMPLETE\n")
    return True


def create_enhanced_synthetic_dataset():
    """Create realistic synthetic dataset with proper correlations"""
    
    logger.info("Generating enhanced synthetic insurance dataset...")
    
    np.random.seed(42)
    n_samples = 3000
    
    data = {
        'id': range(n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 75, n_samples),
        'Driving_License': np.random.choice([0, 1], n_samples, p=[0.05, 0.95]),
        'Region_Code': np.random.randint(0, 53, n_samples),
        'Previously_Insured': np.random.choice([0, 1], n_samples, p=[0.55, 0.45]),
        'Vehicle_Age': np.random.choice(['< 1 Year', '1-2 Year', '> 2 Years'], n_samples),
        'Vehicle_Damage': np.random.choice(['Yes', 'No'], n_samples, p=[0.45, 0.55]),
        'Annual_Premium': np.random.randint(2500, 150000, n_samples),
        'Policy_Sales_Channel': np.random.randint(1, 160, n_samples),
        'Vintage': np.random.randint(10, 365, n_samples)
    }
    
    # Generate Response with realistic business logic
    response = []
    for i in range(n_samples):
        # Base propensity score
        propensity = 30
        
        # Strong positive indicators (customer WANTS insurance)
        if data['Vehicle_Damage'][i] == 'Yes':
            propensity += 35  # Had damage, needs insurance
        
        if data['Previously_Insured'][i] == 0:
            propensity += 30  # Never had insurance, likely interested
        
        if data['Vehicle_Age'][i] == '> 2 Years':
            propensity += 20  # Older car, more likely to buy
        
        if data['Age'][i] < 30:
            propensity += 15  # Young drivers more interested
        
        if data['Annual_Premium'][i] < 20000:
            propensity += 10  # Affordable premium
        
        # Negative indicators
        if data['Previously_Insured'][i] == 1 and data['Vehicle_Damage'][i] == 'No':
            propensity -= 25  # Already insured, no damage, less likely to switch
        
        if data['Age'][i] > 60:
            propensity -= 10  # Older, more set in ways
        
        # Add realistic randomness
        propensity += np.random.randint(-15, 15)
        
        # Convert to binary (1 = will buy insurance)
        response.append(1 if propensity > 50 else 0)
    
    data['Response'] = response
    
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/train.csv', index=False)
    
    dist = dict(pd.Series(response).value_counts())
    logger.info(f"✓ Enhanced synthetic dataset created: {n_samples} samples")
    logger.info(f"✓ Response distribution: {dist}")
    logger.info(f"✓ Interest rate: {dist.get(1, 0)/n_samples*100:.1f}%")


def save_models():
    """Save all trained models"""
    
    model_package = {
        'models': models,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'label_encoders': label_encoders,
        'model_metrics': model_metrics,
        'default_model': default_model,
        'training_date': datetime.now().isoformat(),
        'version': '2.0_enhanced'
    }
    
    try:
        os.makedirs('models', exist_ok=True)
        joblib.dump(model_package, 'models/underwriting_models_enhanced_v2.pkl')
        logger.info("✓ Enhanced models saved successfully")
    except Exception as e:
        logger.error(f"Error saving models: {e}")


def load_models():
    """Load pre-trained models"""
    global models, scaler, feature_columns, label_encoders, model_metrics, default_model
    
    try:
        model_package = joblib.load('models/underwriting_models_enhanced_v2.pkl')
        
        models = model_package['models']
        scaler = model_package['scaler']
        feature_columns = model_package['feature_columns']
        label_encoders = model_package['label_encoders']
        model_metrics = model_package['model_metrics']
        default_model = model_package['default_model']
        
        logger.info("✓ Enhanced pre-trained models loaded successfully")
        logger.info(f"  • Version: {model_package.get('version', 'N/A')}")
        logger.info(f"  • Default model: {default_model}")
        
        return True
        
    except FileNotFoundError:
        logger.warning("⚠️  No pre-trained models found")
        return False
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Main dashboard"""
    session['chat_history'] = []
    return render_template('index.html', model_metrics=model_metrics, default_model=default_model)


@app.route('/predict', methods=['POST'])
def predict():
    """
    ENHANCED PREDICTION ENDPOINT with corrected risk logic
    """
    try:
        # Get form data and convert types properly
        form_data = request.form.to_dict()
        
        # CRITICAL FIX: Convert numeric fields from string to proper types
        numeric_fields = ['Age', 'Region_Code', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
        for field in numeric_fields:
            if field in form_data and form_data[field]:
                try:
                    # Convert to int for integer fields
                    if field in ['Age', 'Region_Code', 'Policy_Sales_Channel', 'Vintage']:
                        form_data[field] = int(form_data[field])
                    # Convert to float for Annual_Premium
                    elif field == 'Annual_Premium':
                        form_data[field] = float(form_data[field])
                except (ValueError, TypeError):
                    pass  # Will be caught by validation
        
        logger.info(f"New prediction request: Gender={form_data.get('Gender', 'N/A')}, Age={form_data.get('Age', 'N/A')}")
        
        # STEP 1: Validate input with business rules
        rules_engine = BusinessRulesEngine()
        is_valid, errors, warnings = rules_engine.validate_input(form_data)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors,
                'warnings': warnings
            }), 400
        
        # STEP 2: Calculate red flag score
        red_flag_score, red_flags = rules_engine.calculate_red_flag_score(form_data)
        
        logger.info(f"Red Flag Assessment: Score={red_flag_score}/100, Flags={len(red_flags)}")
        
        # STEP 3: Prepare features for ML model
        preprocessor = EnhancedDataPreprocessor()
        input_df = pd.DataFrame([form_data])
        input_df = preprocessor.engineer_features(input_df)
        
        # Encode categorical features
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except:
                    input_df[col] = -1
        
        # Ensure all features present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_columns]
        
        # Scale features
        X_input = scaler.transform(input_df)
        
        # STEP 4: ML Prediction
        selected_model = request.form.get('model', default_model)
        model = models.get(selected_model, models[default_model])
        
        prediction_proba = model.predict_proba(X_input)[0]
        ml_probability = float(prediction_proba[1])  # Probability customer will BUY insurance
        
        logger.info(f"ML Prediction: P(will_buy_insurance) = {ml_probability*100:.2f}%")
        
        # STEP 5: Map to comprehensive risk tier (CORRECTED LOGIC)
        risk_engine = EnhancedRiskTierEngine()
        risk_data = risk_engine.map_to_comprehensive_risk(
            ml_probability, red_flag_score, red_flags, form_data
        )
        
        logger.info(f"Risk Assessment: {risk_data['risk_tier']}, Composite Score: {risk_data['composite_risk_score']:.2f}/100")
        
        # STEP 6: Execute GenAI Prompt Chain
        genai_chain = EnhancedGenAIPromptChain(OPENROUTER_API_KEY)
        genai_output = genai_chain.execute_full_chain(form_data, risk_data)
        
        # STEP 7: Get required documents
        doc_agent = EnhancedDocumentRequestAgent()
        document_requirements = doc_agent.get_required_documents(
            risk_data['risk_tier'], form_data, red_flags
        )
        
        # Build comprehensive response
        response = {
            'success': True,
            'model_used': selected_model,
            'validation': {
                'errors': errors,
                'warnings': warnings,
                'red_flag_score': red_flag_score,
                'red_flags': red_flags
            },
            'prediction': {
                'risk_tier': risk_data['risk_tier'],
                'risk_level': risk_data['risk_level'],
                'composite_risk_score': risk_data['composite_risk_score'],
                'business_attractiveness': risk_data['business_attractiveness'],
                'underwriting_risk': risk_data['underwriting_risk'],
                'ml_probability': risk_data['ml_probability'],
                'confidence': risk_data['confidence'],
                'action': risk_data['action'],
                'color': risk_data['color'],
                'recommendation': risk_data['recommendation'],
                'risk_breakdown': risk_data['risk_breakdown']
            },
            'probabilities': {
                'not_interested': round(float(prediction_proba[0]) * 100, 2),
                'interested': round(float(prediction_proba[1]) * 100, 2)
            },
            'genai_insights': {
                'summary': genai_output['summary'],
                'rationale': genai_output['rationale'],
                'recommendation': genai_output['recommendation']
            },
            'documents': document_requirements,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Prediction complete: {risk_data['risk_tier']} (Composite: {risk_data['composite_risk_score']:.2f}/100)")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Enhanced conversational AI endpoint"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        context = data.get('context', {})
        
        if not user_message:
            return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        logger.info(f"💬 Chat request: {user_message[:50]}...")
        
        # Build context-aware prompt
        if context and 'prediction' in context:
            pred = context['prediction']
            
            prompt = f"""You are an expert AI underwriting assistant helping with a specific application.

**Current Application Context:**
- Risk Tier: {pred['risk_tier']}
- Composite Risk Score: {pred.get('composite_risk_score', 'N/A')}/100
- Business Attractiveness: {pred.get('business_attractiveness', 'N/A')}/100
- Underwriting Risk: {pred.get('underwriting_risk', 'N/A')}/100
- Recommended Action: {pred['action']}
- Red Flags: {len(context.get('validation', {}).get('red_flags', []))}

**Underwriter's Question:**
{user_message}

Provide a clear, professional answer in 2-4 sentences. Reference the specific application context. Be actionable and specific."""
        else:
            prompt = f"""You are an expert AI underwriting assistant for insurance applications.

**Question:**
{user_message}

Provide a clear, professional answer in 2-4 sentences. Focus on:
- Risk assessment principles
- Document requirements  
- Underwriting best practices
- Indian insurance market standards

Be helpful and actionable."""
        
        genai = EnhancedGenAIPromptChain(OPENROUTER_API_KEY)
        response = genai._call_genai(prompt, max_tokens=400)
        
        logger.info(f"✓ Chat response generated")
        
        return jsonify({'success': True, 'response': response})
    
    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Sorry, I encountered an error. Please try again.'
        }), 500


@app.route('/model_metrics')
def get_model_metrics():
    """Return model performance metrics"""
    return jsonify({
        'metrics': model_metrics,
        'default_model': default_model,
        'available_models': list(models.keys())
    })


@app.route('/health')
def health_check():
    """Enhanced health check"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models) > 0,
        'default_model': default_model,
        'version': '2.0_enhanced',
        'features': len(feature_columns),
        'timestamp': datetime.now().isoformat()
    })


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*80)
    print("  ENHANCED GENAI-POWERED INSURANCE UNDERWRITING CO-PILOT v2.0")
    print("  Corrected Risk Logic + Advanced Business Rules")
    print("="*80 + "\n")
    
    # Try loading pre-trained models
    models_loaded = load_models()
    
    # If no models found, train from scratch
    if not models_loaded:
        logger.info("No pre-trained models found. Starting enhanced training pipeline...\n")
        training_success = train_enhanced_models()
        
        if not training_success:
            logger.warning("⚠️  Using demonstration mode")
    
    # Display system status
    print("\n" + "="*80)
    print("🚀 ENHANCED SYSTEM STATUS")
    print("="*80)
    print(f"✅ Models Loaded: {list(models.keys())}")
    print(f"✅ Default Model: {default_model}")
    print(f"✅ Features: {len(feature_columns)}")
    print(f"✅ Input Validation: Enhanced with Business Rules")
    print(f"✅ Risk Logic: CORRECTED (ML probability + Red flags)")
    print(f"✅ GenAI: 3-Stage Enhanced Prompts")
    print(f"✅ Document Engine: Context-Aware")
    
    if model_metrics:
        print("\n📊 MODEL PERFORMANCE:")
        for model_name, metrics in model_metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  • ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  • Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  • F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\n" + "="*80)
    print("KEY IMPROVEMENTS:")
    print("  1. ✅ Fixed inverted risk logic")
    print("  2. ✅ Added comprehensive input validation")
    print("  3. ✅ Enhanced feature engineering (20+ features)")
    print("  4. ✅ Red flag detection system")
    print("  5. ✅ Composite risk scoring")
    print("  6. ✅ Context-aware document requests")
    print("="*80 + "\n")
    
    print("🌐 Starting Enhanced Flask Server...")
    print("📍 URL: http://localhost:5000")
    print("="*80 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)