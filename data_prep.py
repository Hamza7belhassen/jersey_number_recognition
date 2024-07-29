# Database downloaded from Roboflow
# https://universe.roboflow.com/flashxyz/jerseydetection/browse?queryText=&pageSize=50&startingIndex=100&browseQuery=true
# Format csv : Multi Label classification

import os
import pandas as pd
import csv
import random

def change_names(csv_path,image_dir):

    csv_path = csv_path
    df = pd.read_csv(csv_path)

    image_directory = image_dir

    backup_csv_path = csv_path.replace('.csv', '_backup.csv')
    df.to_csv(backup_csv_path, index=False)

    for index, row in df.iterrows():
        old_filename = row['filename']
        new_filename = f'valid_{index:04d}.jpg'  # New filename format

        # Construct full old and new file paths
        old_file_path = os.path.join(image_directory, old_filename)
        new_file_path = os.path.join(image_directory, new_filename)

        # Rename the image file
        if os.path.exists(old_file_path):
            os.rename(old_file_path, new_file_path)
            # Update the filename in the DataFrame
            df.at[index, 'filename'] = new_filename
        else:
            print(f"File {old_filename} not found in directory")

    df.to_csv(csv_path, index=False)

def classes_org(csv_path,new_csv_path):

    csv_path = csv_path

    df = pd.read_csv(csv_path)

    new_df = pd.DataFrame()
    new_df['filename'] = df['filename']

    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    sliced_df = df.iloc[:, 1:]

    classes = []
    for index, row in sliced_df.iterrows():
        class_column = row.idxmax()  # Find the name of the column with the max value in the row
        class_number = int(class_column)  # Convert the class name (which is a string) to an integer
        classes.append(class_number)

    # Add the 'class' column to the new DataFrame
    new_df['class'] = classes

    # Save the new DataFrame to a new CSV file
    new_csv_path = new_csv_path
    new_df.to_csv(new_csv_path, index=False)


# change_names('/train/_classes.csv','/train')
# change_names('/test/_classes.csv','/test')
# change_names('/valid/_classes.csv','/valid')
#
# classes_org('C:/Users/HassenBELHASSEN/Desktop/hamza/train/_classes.csv','C:/Users/HassenBELHASSEN/Desktop/hamza/train/_classes_f.csv')
# classes_org('/test/_classes.csv','/test/_classes_f.csv')
# classes_org('C:/Users/HassenBELHASSEN/Desktop/hamza/valid/_classes.csv','C:/Users/HassenBELHASSEN/Desktop/hamza/valid/_classes_f.csv')


#
# # Define the input and output file paths
# input_file = 'C:/Users/HassenBELHASSEN/Desktop/hamza/valid/_classes.csv'
# output_file = 'C:/Users/HassenBELHASSEN/Desktop/hamza/valid/_classes_m.csv'
#
# # Read the original CSV and modify it
# with open(input_file, 'r', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     rows = list(reader)  # Read all rows into a list
#
# # Modify the header
# header = rows[0]
# new_header = [header[0]] + ['-1'] + header[1:]  # Add 'class -1' to the beginning of the header
#
# # Modify each row
# modified_rows = [new_header]
# for row in rows[1:]:
#     if row:  # Ensure the row is not empty
#         new_row = [row[0]] + ['0'] + row[1:]  # Add '0' to the beginning of the one-hot encoded part
#         modified_rows.append(new_row)
#
# # Write the modified rows to a new CSV file
# with open(output_file, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(modified_rows)
#
# print(f'Modified CSV file saved as {output_file}')

#
#
# # Directory containing your images
# image_dir = 'C:/Users/HassenBELHASSEN/Desktop/class_1'
#
# # Path to your CSV file
# csv_file = 'C:/Users/HassenBELHASSEN/Desktop/hamza/train/_classes_m.csv'
#
# # Values to add for each image
# values = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(len(values))
# #Get a list of all image files in the directory
# image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
#
# # Append each image filename with values to the CSV file
# with open(csv_file, 'a', newline='') as file:
#     writer = csv.writer(file)
#     for image_file in image_files:
#         writer.writerow([image_file] + values)
#
# print(f"Added {len(image_files)} images to {csv_file}")
#
# # Read all data from the CSV file
# with open(csv_file, 'r', newline='') as file:
#     reader = csv.reader(file)
#     data = list(reader)
#
# # Shuffle the data rows (excluding the header if there is one)
# header = data[0] if data and len(data) > 0 else None
# if header:
#     shuffled_data = [header] + random.sample(data[1:], len(data) - 1)
# else:
#     shuffled_data = random.sample(data, len(data))
#
# # Write the shuffled data back to the CSV file
# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(shuffled_data)
#
# print("Shuffled all rows in the CSV file.")

input_file = 'C:/Users/HassenBELHASSEN/Desktop/hamza/train/_classes.csv'  # Replace with the actual path to your input CSV
df = pd.read_csv(input_file)

# Initialize a new DataFrame for the transformed data
columns = ['filename'] + [str(i) for i in range(10)]
transformed_data = []

# Process each row in the original DataFrame
for index, row in df.iterrows():
    filename = row['filename']
    new_row = {col: 0 for col in columns}
    new_row['filename'] = filename

    for col in df.columns[1:]:  # Skip the 'filename' column
        if row[col] == 1:
            col_int = int(col)
            if col_int < 10:
                new_row[str(col_int)] = 1
            else:
                digit_1 = col_int // 10
                digit_2 = col_int % 10
                new_row[str(digit_1)] = 1
                new_row[str(digit_2)] = 1

    transformed_data.append(new_row)

transformed_df = pd.DataFrame(transformed_data, columns=columns)
output_file = 'C:/Users/HassenBELHASSEN/Desktop/hamza/train/new_classes.csv'  # Replace with the actual path to your output CSV
transformed_df.to_csv(output_file, index=False)