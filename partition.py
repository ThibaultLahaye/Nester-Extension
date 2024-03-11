import os
import shutil
from sklearn.model_selection import train_test_split

def partition():
    # Define the symbols to be split into testing and training sets
    numerical_symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    math_symbols = ['+', '=', '-']
    symbols = numerical_symbols + math_symbols

    # Path to your data directory
    data_dir = './extracted_images/'

    # Get the list of all class directories for symbols of interest
    class_directories = [os.path.join(data_dir, directory) for directory in os.listdir(data_dir) if directory in symbols]

    # Initialize lists to store file paths and corresponding labels
    file_paths = []
    labels = []

    # Loop through each class directory
    for class_dir in class_directories:
        # Get the class label from the directory name
        label = os.path.basename(class_dir)
        
        # Get the list of file paths in the class directory
        class_files = [os.path.join(class_dir, file) for file in os.listdir(class_dir)]
        
        # Add file paths and corresponding labels to the lists
        file_paths.extend(class_files)
        labels.extend([label] * len(class_files))

    # Split the data into training and testing sets
    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42)  # You can adjust the test_size as needed

    # Print the number of samples in each set
    print("Number of samples in training set:", len(train_files))
    print("Number of samples in testing set:", len(test_files))

    # Define directories
    train_dir = './splited_dataset/train/'
    test_dir = './splited_dataset/test/'

    # Remove existing directories if they exist
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Copy training files
    for file_path, label in zip(train_files, train_labels):
        label_dir = os.path.join(train_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(file_path, os.path.join(label_dir, os.path.basename(file_path)))

    # Copy testing files
    for file_path, label in zip(test_files, test_labels):
        label_dir = os.path.join(test_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(file_path, os.path.join(label_dir, os.path.basename(file_path)))

    print("Data split and copied successfully.")

if __name__ == "__main__":
    partition()