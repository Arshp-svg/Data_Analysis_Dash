import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

class AdvancedAnalytics:
    def __init__(self, df):
        self.df = df
        
    def perform_time_series_analysis(self, date_column, value_column):
        # Convert date column to datetime
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        
        # Sort by date
        df_sorted = self.df.sort_values(date_column)
        
        # Create time series plot with trend
        fig = px.scatter(df_sorted, x=date_column, y=value_column, title=f'Time Series Analysis: {value_column}')
        
        # Add trend line
        z = np.polyfit(range(len(df_sorted)), df_sorted[value_column], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=df_sorted[date_column],
                y=p(range(len(df_sorted))),
                name="Trend",
                line=dict(color="red")
            )
        )
        
        return fig
    
    def perform_statistical_test(self, test_type):
        if test_type == "T-Test":
            numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) >= 2:
                t_stat, p_value = stats.ttest_ind(
                    self.df[numeric_columns[0]].dropna(),
                    self.df[numeric_columns[1]].dropna()
                )
                return {
                    "test": "Independent T-Test",
                    "t_statistic": t_stat,
                    "p_value": p_value
                }
        elif test_type == "Chi-Square Test":
            categorical_columns = self.df.select_dtypes(include=['object']).columns
            if len(categorical_columns) >= 2:
                contingency_table = pd.crosstab(
                    self.df[categorical_columns[0]],
                    self.df[categorical_columns[1]]
                )
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                return {
                    "test": "Chi-Square Test",
                    "chi2_statistic": chi2,
                    "p_value": p_value,
                    "degrees_of_freedom": dof
                }
        
        return "Not enough appropriate columns for selected test"
    
    def perform_regression_analysis(self, x_column, y_column):
        X = self.df[x_column].values.reshape(-1, 1)
        y = self.df[y_column].values
        
        # Add constant for statsmodels
        X = sm.add_constant(X)
        
        # Fit regression model
        model = sm.OLS(y, X).fit()
        
        # Create scatter plot with regression line
        fig = px.scatter(self.df, x=x_column, y=y_column, title=f'Regression Analysis: {y_column} vs {x_column}')
        
        # Add regression line
        x_range = np.linspace(self.df[x_column].min(), self.df[x_column].max(), 100)
        y_pred = model.predict([1] * len(x_range), x_range)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                name="Regression Line",
                line=dict(color="red")
            )
        )
        
        return fig, model.summary()
