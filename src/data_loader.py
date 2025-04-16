import pandas as pd

def load_data(url=None):
    """ 
    Loads the Iris dataset from a specified online source.
    
    Parameters:
    url (str): Optional custom URL to load the dataset from.
               Defaults to seaborn's Iris dataset URL.
    
    Returns:
    pandas.DataFrame: The loaded dataset containing features and labels.
    """
    default_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    dataset_url = url if url else default_url
    
    try:
        data = pd.read_csv(dataset_url)
        return data
    except Exception as e:
        print(f"Failed to load dataset from {dataset_url}: {e}")
        return pd.DataFrame()
