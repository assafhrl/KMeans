import pandas as pd
import numpy as np
import os
import sys
import mykmeanssp

ACCEPTED_INPUT_FILES_EXTENSIONS = ["txt", "csv"]
DEFAULT_MAX_ITER = 300
MIN_K = 2
MIN_MAX_ITER = 1
MIN_EPS = 0

# Get extension of file (if exists)
def get_filepath_extension(filepath):
    return filepath.split(".")[-1]

# Checks if the file type is valid
def is_valid_file(filepath):
    if os.path.isfile(filepath):
        return get_filepath_extension(filepath) in ACCEPTED_INPUT_FILES_EXTENSIONS
    return False

# Reads the input of a file
def read_single_input(filepath):
    df = pd.read_csv(filepath, header=None, index_col=0)
    df.index = df.index.astype(int, copy=False)
    return df

# Read the inputs from the files given to the program
def read_input(filepath_1, filepath_2):
    df1 = read_single_input(filepath_1)
    df2 = read_single_input(filepath_2)
    df = df1.join(df2, how='inner', rsuffix='_', sort=True)
    column_names = {df.columns[i]: i for i in range(len(df.columns))}
    df = df.rename(columns=column_names)
    return df

# Calculate the distance of a centroid from all other data points
def calculate_centroid_distance(centroid, data_points):
    centroid_diff = data_points.sub(centroid, axis=1)
    centroid_diff_square = centroid_diff*centroid_diff
    diff = centroid_diff_square.sum(axis=1)
    return diff

# Generate the first initial centroid with uniform probability
def generate_first_initial_centroid(data_points):
    np.random.seed(0)
    centroid_index = np.random.choice(data_points.index)
    return data_points.loc[[centroid_index]]

# Generate the next centroid based on the distances from other centroids
def generate_next_initial_centroids(data_points, dist):
    d = dist.min(axis=1)
    p = d / np.sum(d)
    centroid_index = np.random.choice(data_points.index, p=p)
    return data_points.loc[[centroid_index]]

# Update the DataFrame which contains the distances of each data point from all the centroids
def update_dist(centroids, data_points, dist):
    dist[centroids.index[-1]] = calculate_centroid_distance(centroids.iloc[-1], data_points)
    return dist
    
# Generate a DataFrame of centroids from the data points using a random process
def generate_initial_centroids(data_points, k):
    centroids = generate_first_initial_centroid(data_points)
    dist = pd.DataFrame(index=data_points.index)
    for i in range(k-1):
        dist = update_dist(centroids, data_points, dist)
        centroids = pd.concat([centroids, generate_next_initial_centroids(data_points, dist)])
    return centroids

# Exit the program in case of an invalid input
def exit_invalid_input():
    print("Invalid Input!")
    sys.exit(1)

# Check validity of arguments and parse them
def get_args():
    args = sys.argv.copy()
    if len(args) == 5:
        args.insert(2, str(DEFAULT_MAX_ITER))
    elif len(args) == 6:
        pass
    else:
        exit_invalid_input()

    try:
        k = int(args[1])
        if k < MIN_K:
            exit_invalid_input()
        max_iter = int(args[2])
        if max_iter < MIN_MAX_ITER:
            exit_invalid_input()
        eps = float(args[3])
        if eps < MIN_EPS:
            exit_invalid_input()
    except ValueError:
        exit_invalid_input()

    filepath_1 = args[4]
    filepath_2 = args[5]
    if not (is_valid_file(filepath_1) and is_valid_file(filepath_2)):
        exit_invalid_input()

    return k, max_iter, eps, filepath_1, filepath_2

# Convert a DataFrame to a list of lists
def df_to_list_of_lists(df):
    l = []
    for i in range(len(df)):
        sl = [float(df.iloc[i][j]) for j in df.columns]
        l.append(sl)
    return l

# Create string representation of a list
def list_to_str(l, template):
    l_str = [template.format(s) for s in l]
    return ",".join(l_str)

# Print the indices of the chosen centroids
def print_initial_centroids_indices(initial_centroids):
    indices = list(initial_centroids.index)
    indices_str = list_to_str(indices, "{:n}")
    print(indices_str)

# Print the result of the final centroids
def print_centroids(centroids_c):
    for centroid_c in centroids_c:
        centroid_str = list_to_str(centroid_c, "{:.4f}")
        print(centroid_str)

# Print the result of the program
def print_res(initial_centroids, centroids_c):
    print_initial_centroids_indices(initial_centroids)
    print_centroids(centroids_c)

# Exit the program in case of an error
def exit_error():
    print("An Error Has Occurred")
    sys.exit(1)

# Main Function
def main():
    k, max_iter, eps, filepath_1, filepath_2 = get_args()
    data_points = read_input(filepath_1, filepath_2)
    initial_centroids = generate_initial_centroids(data_points, k)
    data_points_c = df_to_list_of_lists(data_points)
    initial_centroids_c = df_to_list_of_lists(initial_centroids)
    centroids_c = mykmeanssp.fit(data_points_c, initial_centroids_c, eps, max_iter)
    if centroids_c is None:
        exit_error()
    print_res(initial_centroids, centroids_c)
    sys.exit(0)

if __name__ == "__main__":
    main()
