# -*- coding: utf-8 -*-
"""
File Name: battery_ml_lib.py

Description: analysis of features.

Author: junghwan lee
Email: jhrrlee@gmail.com
Date Created: 2023.09.08
"""

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso
import xgboost as xgb
#import shap
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

class BatteryFeatureImportance:
    def __init__(self, new_cycle_sum, eol, feature_names):
        self.new_cycle_sum = new_cycle_sum
        self.eol = eol.ravel()
        self.feature_names = feature_names

        self.num_cells, self.num_features, self.num_cycles = new_cycle_sum.shape
        self.expanded_data = new_cycle_sum.reshape(self.num_cells, -1)

        self.model = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        # For storing permutation importance and aggregated importance
        self.perm_importance = None
        self.aggregated_importance = None

    def train_model(self, regressor_choice='random_forest', n_estimators=100):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.expanded_data, self.eol, test_size=0.3, random_state=42)

        if regressor_choice == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif regressor_choice == 'gradient_boosting':
            self.model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        elif regressor_choice == 'lasso':
            self.model = Lasso(alpha=0.1)
        elif regressor_choice == 'xgboost':
            self.model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=42)
        else:
            raise ValueError(f"Unknown regressor choice: {regressor_choice}")

        self.model.fit(self.X_train, self.Y_train)

    def feature_importance(self, method='permutation', n_repeats=30):
        if method == 'permutation':
            perm_result = permutation_importance(self.model, self.X_test, self.Y_test, n_repeats=n_repeats, random_state=42)
            self.perm_importance = perm_result.importances_mean
        elif method == 'coefficients':
            self.perm_importance = self.model.coef_
        elif method == 'tree_importance':
            self.perm_importance = self.model.feature_importances_
        elif method == 'shap':
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test)
            self.perm_importance = np.mean(np.abs(shap_values), axis=0)
        else:
            raise ValueError(f"Unknown feature importance method: {method}")

        self.aggregate_feature_importance()
        self.plot_aggregated_importance()

    def evaluate_model(self):
        train_preds = self.model.predict(self.X_train)
        test_preds = self.model.predict(self.X_test)

        metrics = {
            "Training MSE": mean_squared_error(self.Y_train, train_preds),
            "Validation MSE": mean_squared_error(self.Y_test, test_preds),
            "Training RMSE": np.sqrt(mean_squared_error(self.Y_train, train_preds)),
            "Validation RMSE": np.sqrt(mean_squared_error(self.Y_test, test_preds)),
            "Training MAE": mean_absolute_error(self.Y_train, train_preds),
            "Validation MAE": mean_absolute_error(self.Y_test, test_preds),
            "Training MAPE": np.mean(np.abs((self.Y_train - train_preds) / self.Y_train)) * 100,
            "Validation MAPE": np.mean(np.abs((self.Y_test - test_preds) / self.Y_test)) * 100
        }

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    def aggregate_feature_importance(self):
        expanded_importance = self.perm_importance
        aggregated_importance = [expanded_importance[i::self.num_cycles].sum() for i in range(self.num_features)]
        self.aggregated_importance = np.array(aggregated_importance)

    def plot_aggregated_importance(self):
        sorted_idx = self.aggregated_importance.argsort()
        feature_names_sorted = np.array(self.feature_names)[sorted_idx]
        feature_importances_sorted = self.aggregated_importance[sorted_idx]

        plt.figure(figsize=(10, 5))
        plt.barh(feature_names_sorted, feature_importances_sorted)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        plt.show()