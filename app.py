import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from eda import EDA
from advanced_analytics import AdvancedAnalytics
from visualizations import Visualizations

def main():
    st.title("Advanced Data Analysis Dashboard")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            # Initialize modules
            processor = DataProcessor(df)
            eda = EDA(df)
            analytics = AdvancedAnalytics(df)
            viz = Visualizations(df)
            
            # Sidebar navigation
            analysis_type = st.sidebar.selectbox(
                "Select Analysis Type",
                ["Data Preprocessing", "Exploratory Analysis", "Advanced Analytics", "Visualizations"]
            )
            
            if analysis_type == "Data Preprocessing":
                st.header("Data Preprocessing")
                
                # Display basic information
                st.subheader("Dataset Overview")
                st.write(f"Number of rows: {df.shape[0]}")
                st.write(f"Number of columns: {df.shape[1]}")
                
                # Data preprocessing options
                if st.checkbox("Handle Missing Values"):
                    df = processor.handle_missing_values()
                    st.success("Missing values handled successfully!")
                
                if st.checkbox("Convert Data Types"):
                    df = processor.convert_data_types()
                    st.success("Data types converted successfully!")
                
                if st.checkbox("Clean Data"):
                    df = processor.clean_data()
                    st.success("Data cleaned successfully!")
                
                # Display processed data
                st.subheader("Processed Data Preview")
                st.write(df.head())
                
            elif analysis_type == "Exploratory Analysis":
                st.header("Exploratory Data Analysis")
                
                # Summary statistics
                if st.checkbox("Show Summary Statistics"):
                    st.subheader("Summary Statistics")
                    st.write(eda.get_summary_statistics())
                
                # Distribution analysis
                if st.checkbox("Show Distribution Analysis"):
                    st.subheader("Distribution Analysis")
                    selected_column = st.selectbox("Select column for distribution analysis", df.columns)
                    fig = eda.plot_distribution(selected_column)
                    st.plotly_chart(fig)
                
                # Correlation analysis
                if st.checkbox("Show Correlation Analysis"):
                    st.subheader("Correlation Analysis")
                    fig = eda.plot_correlation_matrix()
                    st.plotly_chart(fig)
                
            elif analysis_type == "Advanced Analytics":
                st.header("Advanced Analytics")
                
                analysis_option = st.selectbox(
                    "Select Analysis Type",
                    ["Time Series Analysis", "Statistical Tests", "Regression Analysis"]
                )
                
                if analysis_option == "Time Series Analysis":
                    date_column = st.selectbox("Select date column", df.columns)
                    value_column = st.selectbox("Select value column", df.columns)
                    fig = analytics.perform_time_series_analysis(date_column, value_column)
                    st.plotly_chart(fig)
                
                elif analysis_option == "Statistical Tests":
                    test_type = st.selectbox("Select Statistical Test", ["T-Test", "Chi-Square Test"])
                    result = analytics.perform_statistical_test(test_type)
                    st.write(result)
                
                elif analysis_option == "Regression Analysis":
                    x_column = st.selectbox("Select independent variable", df.columns)
                    y_column = st.selectbox("Select dependent variable", df.columns)
                    fig, summary = analytics.perform_regression_analysis(x_column, y_column)
                    st.plotly_chart(fig)
                    st.write(summary)
                
            elif analysis_type == "Visualizations":
                st.header("Data Visualizations")

                plot_type = st.selectbox(
                    "Select Plot Type",
                    [
                        "Scatter Plot",
                        "Box Plot",
                        "Time Series Plot",
                        "Correlation Heatmap",
                        "Bar Chart",
                        "Pie Chart",
                        "Histogram",
                        "Area Chart",
                        "Violin Plot"
                    ]
                )

                if plot_type == "Scatter Plot":
                    x_column = st.selectbox("Select X-axis", df.columns)
                    y_column = st.selectbox("Select Y-axis", df.columns)
                    fig = viz.create_scatter_plot(x_column, y_column)
                    st.plotly_chart(fig)

                elif plot_type == "Box Plot":
                    column = st.selectbox("Select column", df.columns)
                    fig = viz.create_box_plot(column)
                    st.plotly_chart(fig)

                elif plot_type == "Time Series Plot":
                    date_column = st.selectbox("Select date column", df.columns)
                    value_column = st.selectbox("Select value column", df.columns)
                    fig = viz.create_time_series_plot(date_column, value_column)
                    st.plotly_chart(fig)

                elif plot_type == "Correlation Heatmap":
                    fig = viz.create_correlation_heatmap()
                    st.plotly_chart(fig)

                elif plot_type == "Bar Chart":
                    x_column = st.selectbox("Select X-axis", df.columns)
                    y_column = st.selectbox("Select Y-axis", df.columns)
                    color_column = st.selectbox("Optional: Select color grouping", [None] + list(df.columns))
                    fig = viz.create_bar_chart(x_column, y_column, color_column if color_column != "None" else None)
                    st.plotly_chart(fig)

                elif plot_type == "Pie Chart":
                    names_column = st.selectbox("Select category column", df.columns)
                    values_column = st.selectbox("Select values column", df.columns)
                    fig = viz.create_pie_chart(names_column, values_column)
                    st.plotly_chart(fig)

                elif plot_type == "Histogram":
                    column = st.selectbox("Select column for histogram", df.columns)
                    bins = st.slider("Select number of bins", 5, 100, 30)
                    fig = viz.create_histogram(column, nbins=bins)
                    st.plotly_chart(fig)

                elif plot_type == "Area Chart":
                    x_column = st.selectbox("Select X-axis", df.columns)
                    y_column = st.selectbox("Select Y-axis", df.columns)
                    fig = viz.create_area_chart(x_column, y_column)
                    st.plotly_chart(fig)

                elif plot_type == "Violin Plot":
                    y_column = st.selectbox("Select Y-axis", df.columns)
                    x_column = st.selectbox("Optional: Select category (X-axis)", [None] + list(df.columns))
                    color_column = st.selectbox("Optional: Select color grouping", [None] + list(df.columns))
                    fig = viz.create_violin_plot(
                        y_column,
                        x_column if x_column != "None" else None,
                        color_column if color_column != "None" else None
                    )
                    st.plotly_chart(fig)

                        
            # Data export
            if st.button("Export Processed Data"):
                processed_df = processor.get_processed_data()
                csv = processed_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
if __name__ == "__main__":
    main()
