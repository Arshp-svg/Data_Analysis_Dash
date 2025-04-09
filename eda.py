import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class EDA:
    def __init__(self, df):
        self.df = df
        
    def get_summary_statistics(self):
        return self.df.describe()
    
    def plot_distribution(self, column):
        fig = px.histogram(self.df, x=column, title=f'Distribution of {column}')
        fig.update_layout(showlegend=False)
        return fig
    
    def plot_correlation_matrix(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        return fig
