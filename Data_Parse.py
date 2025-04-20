import numpy as np
import pandas as pd

def open_file():

    file = pd.read_csv("student_performance_dataset.csv", header=0, dtype=str)
    
    # Drop the first column (Student_ID) as it's not useful for regression
    file = file.drop(columns=["Student_ID"])
    
    # Define mappings
    gender = {"Male": 0, "Female": 1}
    pass_fail = {"Pass": 1, "Fail": 0}
    parent_education = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    home_internet = {"Yes": 1, "No": 0}
    extracurricular_activities = {"Yes": 1, "No": 0}
    pd.set_option('future.no_silent_downcasting', True)
    # Replace categorical values and explicitly cast to integers
    file["Gender"] = file["Gender"].replace(gender).astype(float)
    file["Pass_Fail"] = file["Pass_Fail"].replace(pass_fail).astype(float)
    file["Parental_Education_Level"] = file["Parental_Education_Level"].replace(parent_education).astype(float)
    file["Internet_Access_at_Home"] = file["Internet_Access_at_Home"].replace(home_internet).astype(float)
    file["Extracurricular_Activities"] = file["Extracurricular_Activities"].replace(extracurricular_activities).astype(float)
    

    file = file.dropna()  # Drop rows with missing values
    
    # Convert to NumPy array
    data_array = file.to_numpy()
    data_array = np.array(data_array, dtype=float)
    data_array = normalize(data_array)
    return data_array

def normalize(data_array):
    # Normalize the data array to the range [0, 1]
    min_val = np.min(data_array, axis=0)
    max_val = np.max(data_array, axis=0)
    data_array = (data_array - min_val) / (max_val - min_val)
    return data_array

if __name__ == "__main__":
    # Load the data from the CSV file
    data_array = open_file()

    # Print the shape of the data array
    print("Shape of the data array:", data_array.shape)

    # Print the first 5 elements of the data array
    print("First 5 elements of the data array:", data_array[:5])