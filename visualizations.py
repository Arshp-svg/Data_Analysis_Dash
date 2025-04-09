import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class Visualizations:
    def __init__(self, df):
        self.df = df

    def create_scatter_plot(self, x_column, y_column):
        fig = px.scatter(
            self.df,
            x=x_column,
            y=y_column,
            title=f'Scatter Plot: {y_column} vs {x_column}'
        )
        return fig

    def create_box_plot(self, column):
        fig = px.box(
            self.df,
            y=column,
            title=f'Box Plot: {column}'
        )
        return fig

    def create_time_series_plot(self, date_column, value_column):
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        fig = px.line(
            self.df.sort_values(date_column),
            x=date_column,
            y=value_column,
            title=f'Time Series Plot: {value_column} over time'
        )
        return fig

    def create_correlation_heatmap(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        return fig

    def create_bar_chart(self, x_column, y_column, color_column=None):
        fig = px.bar(
            self.df,
            x=x_column,
            y=y_column,
            color=color_column,
            title=f'Bar Chart: {y_column} by {x_column}'
        )
        return fig

    def create_pie_chart(self, names_column, values_column):
        fig = px.pie(
            self.df,
            names=names_column,
            values=values_column,
            title=f'Pie Chart: {names_column} distribution'
        )
        return fig

    def create_histogram(self, column, nbins=30):
        fig = px.histogram(
            self.df,
            x=column,
            nbins=nbins,
            title=f'Histogram: {column} distribution'
        )
        return fig

    def create_area_chart(self, x_column, y_column):
        fig = px.area(
            self.df,
            x=x_column,
            y=y_column,
            title=f'Area Chart: {y_column} over {x_column}'
        )
        return fig

    def create_violin_plot(self, y_column, x_column=None, color_column=None):
        fig = px.violin(
            self.df,
            y=y_column,
            x=x_column,
            color=color_column,
            box=True,
            points="all",
            title=f'Violin Plot: {y_column}'
        )
        return fig
