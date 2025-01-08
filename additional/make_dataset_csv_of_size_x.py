import os
import pandas as pd

import os
import pandas as pd


def make_dataset_csv_of_size_x(csv_paths, output_path, number_of_lines):
    """
    Create a dataset CSV with the specified number of lines from multiple input CSVs.
    Ensures that no line is used more than once.

    Parameters:
    - csv_paths (list): List of paths to the input CSV files.
    - output_path (str): Path to save the resulting CSV file.
    - number_of_lines (int): Desired total number of lines in the output CSV.

    Returns:
    - None
    """

    lines_remain = number_of_lines
    lines_per_csv = number_of_lines // len(csv_paths)  # Integer division
    output_df = pd.DataFrame()  # Initialize the output DataFrame
    used_indices = {csv_path: set() for csv_path in csv_paths}  # Dictionary to track used indices

    # Ensure the output file does not already exist
    if os.path.exists(output_path):
        output_df = pd.read_csv(output_path)
        print(f"File '{output_path}' already exists.")

    # Process each CSV
    for csv_path in csv_paths:
        # Read the CSV
        df = pd.read_csv(csv_path)
        line_count = len(df)  # Total number of rows in the CSV
        print(f"Processing '{csv_path}'... lines: {line_count}")

        if line_count < lines_per_csv:
            print(f"CSV '{csv_path}' has only {line_count} lines. Adding all lines to output.")
            # Append all lines from this CSV
            output_df = pd.concat([output_df, df], ignore_index=True)
            used_indices[csv_path] = set(range(line_count))  # Mark all lines as used
            lines_remain -= line_count  # Update remaining required lines
        else:
            print(f"CSV '{csv_path}' has enough lines. Taking {lines_per_csv} lines.")
            # Select random rows, ensuring no duplicate indices
            available_indices = set(range(len(df))) - used_indices[csv_path]
            sampled_indices = list(available_indices)[:lines_per_csv]
            output_df = pd.concat([output_df, df.iloc[sampled_indices]], ignore_index=True)
            used_indices[csv_path].update(sampled_indices)  # Mark selected indices as used
            lines_remain -= lines_per_csv

        # Stop if we've collected enough lines
        if lines_remain <= 0:
            break

    # If there are still remaining lines, take from the remaining CSVs
    for csv_path in csv_paths:
        if lines_remain <= 0:
            break  # Stop if we've collected enough lines

        df = pd.read_csv(csv_path)
        available_indices = set(range(len(df))) - used_indices[csv_path]  # Exclude already used indices
        line_count = len(available_indices)

        if line_count >= lines_remain:
            print(f"Taking the remaining {lines_remain} lines from '{csv_path}'.")
            sampled_indices = list(available_indices)[:lines_remain]
            output_df = pd.concat([output_df, df.iloc[sampled_indices]], ignore_index=True)
            used_indices[csv_path].update(sampled_indices)  # Mark selected indices as used
            lines_remain = 0
        else:
            print(f"Taking all {line_count} remaining lines from '{csv_path}'.")
            output_df = pd.concat([output_df, df.iloc[list(available_indices)]], ignore_index=True)
            used_indices[csv_path].update(available_indices)  # Mark selected indices as used
            lines_remain -= line_count

    # Check if we have enough lines
    if len(output_df) < number_of_lines:
        print("The total number of lines available is insufficient to meet the required number of lines!")
        return

    # Save the resulting DataFrame to the output file
    output_df.to_csv(output_path, index=False)
    print(f"Dataset created with {len(output_df)} lines and saved to '{output_path}'.")



fake_folders_paths = [
    r'D:\Database\Audio/DeepFakeProject/fake/OUTETTS Fake audio 4s processed',
    r'D:\Database\Audio/DeepFakeProject/fake/Tortoise Fake audio 4s processed',
    r'D:\Database\Audio/DeepFakeProject/fake/XTTS Fake audio 4s processed'
]

real_folders_paths = [
    r'D:\Database\Audio\DeepFakeProject\Real\cv corpus 20 4s processed',
    r'D:\Database\Audio\DeepFakeProject\Real\Peoples speech 4s processed'
    ]

total_examples = 2700
ratio_fake, ratio_real = 0.5, 0.5
ratio_train, ratio_test, ratio_validation = 0.8, 0.1, 0.1

number_train_fake = int(total_examples * ratio_fake * ratio_train)
number_train_real = int(total_examples * ratio_real * ratio_train)
number_test_fake = int(total_examples * ratio_fake * ratio_test)
number_test_real = int(total_examples * ratio_real * ratio_test)
number_validation_fake = int(total_examples * ratio_fake * ratio_validation)
number_validation_real = int(total_examples * ratio_real * ratio_validation)

print(f"The number of fake train : {number_train_fake}")
print(f"The number of real train : {number_train_real}")
print(f"The number of fake test : {number_test_fake}")
print(f"The number of real test : {number_test_real}")
print(f"The number of validation fake : {number_validation_fake}")
print(f"The number of validation test : {number_validation_real}")

print(f"Total : {number_train_fake + number_train_real + number_test_fake + number_test_real + number_validation_fake + number_validation_real}")


validation_paths = [os.path.join(path, "Validation.csv") for path in fake_folders_paths]
train_paths = [os.path.join(path, "Train.csv") for path in fake_folders_paths]
test_paths = [os.path.join(path, "Test.csv") for path in fake_folders_paths]

#fake data
make_dataset_csv_of_size_x(train_paths, 'train_30h.csv', number_train_fake)
make_dataset_csv_of_size_x(validation_paths, 'validation_30h.csv', number_test_fake)
make_dataset_csv_of_size_x(test_paths, 'test_30h.csv', number_validation_fake)

validation_paths = [os.path.join(path, "Validation.csv") for path in real_folders_paths]
train_paths = [os.path.join(path, "Train.csv") for path in real_folders_paths]
test_paths = [os.path.join(path, "Test.csv") for path in real_folders_paths]

#real data
make_dataset_csv_of_size_x(train_paths, 'train_30h.csv', number_train_real)
make_dataset_csv_of_size_x(validation_paths, 'validation_30h.csv', number_test_real)
make_dataset_csv_of_size_x(test_paths, 'test_30h.csv', number_validation_real)

def analyze_label_column(csv_file_path):
    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file_path)

        # Check if the "label" column exists
        if "label" not in df.columns:
            print("Error: The CSV file does not contain a 'label' column.")
            return

        # Count the occurrences of 0s and 1s in the "label" column
        label_counts = df['label'].value_counts()

        # Extract the counts for 0s and 1s (default to 0 if not present)
        count_0 = label_counts.get(0, 0)
        count_1 = label_counts.get(1, 0)

        # Calculate the ratio of 0s to 1s
        total = count_0 + count_1
        if total == 0:
            print("Error: The 'label' column is empty or contains no 0s or 1s.")
            return

        ratio_0 = count_0 / total
        ratio_1 = count_1 / total

        # Print the results
        print(f"Count of 0s: {count_0}")
        print(f"Count of 1s: {count_1}")
        print(f"Ratio of 0s: {ratio_0:.2f}")
        print(f"Ratio of 1s: {ratio_1:.2f}")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
for name in "train_30h.csv", "validation_30h.csv", "test_30h.csv":
    analyze_label_column(name)


# _____________________________________ private area dont check _____________________________#

def make_dataset_csv_of_size_x1(csv_paths, output_path, number_of_lines):
    # make sure the outout csv exist and make one if not
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        print("Directory {} already exists".format(output_path))
        return

    make_dataset_csv_of_size_x_rec(csv_paths, output_path, number_of_lines)


def make_dataset_csv_of_size_x_rec(csv_paths, output_path, number_of_lines):
    lines_remain = number_of_lines
    sum_of_lines = 0
    lines_per_csv = number_of_lines / len(csv_paths)
    csv_line_counts = {}  # Dictionary to store line counts for each CSV
    next_csv_paths = csv_paths.copy()

    # Check how many lines all the csvs have, and for reach csv if it has enough lines
    for csv_path in csv_paths:
        line_count = len(pd.read_csv(csv_path)) - 1  # Exclude header

        # If the CSV has fewer lines than needed
        if line_count < lines_per_csv:
            print(f"CSV '{csv_path}' has only {line_count} lines. Adding all lines to output.")

            # Load all lines (excluding the header)
            df = pd.read_csv(csv_path)

            # Append to the output DataFrame
            output_df = pd.concat([output_df, df], ignore_index=True)

            # Remove this CSV from the list of files to process further
            next_csv_paths.remove(csv_path)

            # Subtruct lines from needed lines
            lines_remain -= line_count


    if sum_of_lines < number_of_lines:
        print("The number of lines is not sufficient!!!!!!!!!!!!!! (please delete the file)")
        return

    #if all the csv have enough lines put them in
    if len(next_csv_paths) == len(csv_paths):
        for csv_path in next_csv_paths:
            df = pd.read_csv(csv_path)
            output_df = pd.concat([output_df, df.sample(n=lines_per_csv, random_state=42)], ignore_index=True)
    else:
        make_dataset_csv_of_size_x(next_csv_paths, output_path, lines_remain)

