import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class ABTestingBaseline:
    """Standard equal-split A/B testing baseline."""
    def __init__(self, arms, unit_cost):
        self.arms = arms
        self.unit_cost = unit_cost
        self.K = len(arms)
        self.current_idx = 0
        self.counts = np.zeros(self.K, dtype=int)
        self.values = np.zeros(self.K, dtype=float)

    def select_arm(self, context_vector=None):
        arm = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.K
        return arm

    def update(self, arm_idx, reward, converted, context_vector=None):
        self.counts[arm_idx] += 1
        self.values[arm_idx] += (reward - self.values[arm_idx]) / self.counts[arm_idx]


class RegretTracker:
    def __init__(self, opt_reward):
        self.opt_reward = opt_reward
        self.history = []
        self.cum_reward = 0.0
        self.cum_regret = 0.0

    def log_step(self, step, policy, price, reward, opt_exp_reward, context_dict=None, competitor_price=None):
        self.cum_reward += reward
        step_regret = max(0.0, opt_exp_reward - reward)
        self.cum_regret += step_regret
        
        record = {
            "step": step,
            "policy": policy,
            "price": price,
            "reward": reward,
            "cumulative_reward": self.cum_reward,
            "regret": step_regret,
            "cumulative_regret": self.cum_regret
        }
        if context_dict:
            record.update(context_dict)
        if competitor_price:
            record["competitor_price"] = competitor_price
            
        self.history.append(record)

    def get_dataframe(self):
        return pd.DataFrame(self.history)


class ElasticityModel:
    def __init__(self, model_type="random_forest"):
        self.model_type = model_type
        self.model = None
        self.feature_names = ["price", "is_mobile", "is_metro", "is_weekend"]

    def _determine_dynamic_estimators(self, n_samples):
        """Dynamically calculates optimal tree count based on dataset sample size."""
        if n_samples < 30:
            return 15
        elif n_samples < 100:
            return 30
        elif n_samples < 300:
            return 50
        else:
            return 100

    def train(self, df_logs):
        if df_logs.empty:
            return
            
        X = pd.DataFrame()
        X["price"] = df_logs["price"]
        X["is_mobile"] = df_logs.get("device", "Desktop").apply(lambda x: 1 if x == "Mobile" else 0)
        X["is_metro"] = df_logs.get("tier", "Metro").apply(lambda x: 1 if x == "Metro" else 0)
        X["is_weekend"] = df_logs.get("is_weekend", False).astype(int)
        
        y = df_logs["purchased"]

        # Dynamically scale n_estimators based on the dataset size
        n_samples = len(df_logs)
        dynamic_n = self._determine_dynamic_estimators(n_samples)

        self.model = RandomForestRegressor(
            n_estimators=dynamic_n, 
            random_state=42, 
            max_depth=6
        )
        self.model.fit(X, y)

    def get_feature_importances(self):
        if self.model and hasattr(self.model, "feature_importances_"):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}

    def predict_demand_curve(self, prices, is_mobile=0, is_metro=1, is_weekend=0):
        if self.model is None:
            return np.full(len(prices), 0.5)

        X_test = pd.DataFrame({
            "price": prices,
            "is_mobile": [is_mobile] * len(prices),
            "is_metro": [is_metro] * len(prices),
            "is_weekend": [is_weekend] * len(prices)
        })
        p_conv = self.model.predict(X_test)
        p_conv = np.clip(p_conv, 0.01, 0.99)
        return p_conv