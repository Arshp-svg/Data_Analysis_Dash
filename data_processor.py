import pandas as pd

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def handle_missing_values(self, strategy="mean"):
        """
        Handle missing values in the dataset using a simple strategy.
        Default strategy: fill numeric columns with mean, others with mode.
        """
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype in ['float64', 'int64']:
                    if strategy == "mean":
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == "median":
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        return self.df

    def convert_data_types(self):
        """
        Attempt to convert data types automatically where appropriate.
        """
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col])
            except:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    pass  # Leave as is if conversion fails
        return self.df

    def clean_data(self):
        """
        Basic data cleaning - strip strings, remove duplicates.
        """
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].str.strip()
        
        self.df.drop_duplicates(inplace=True)
        return self.df

    def get_processed_data(self):
        """
        Return the processed dataframe.
        """
        return self.df
