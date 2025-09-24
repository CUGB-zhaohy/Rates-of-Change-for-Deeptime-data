import pandas as pd
import matplotlib.pyplot as plt

# Read the first Excel file (first column is Agepoint, others are measurement columns)
file1_path = input("Enter the path to the data file: ").strip()
df1 = pd.read_excel(file1_path)

# Read the second Excel file (contains geological event time ranges)
file2_path = input("Enter the path to the events file: ").strip()
df2 = pd.read_excel(file2_path)

# Strip whitespace from column names in the second file
df2.columns = df2.columns.str.strip()

# Structure of the first file: the first column is Agepoint, the rest are measurement columns
age_col = df1.columns[0]
experiment_columns = df1.columns[1:]  # measurement columns start from the second column

# Create a dictionary to store the final results
result = {'Event Name': []}
for exp_col in experiment_columns:
    result[exp_col] = []

# Iterate over each event
for _, event in df2.iterrows():
    event_name = event['Event Name']
    start_time = event['Start Time']
    end_time = event['End Time']

    # Filter rows within the event's time range
    selected_data = df1[(df1[age_col] >= end_time) & (df1[age_col] <= start_time)]

    # Compute the maximum for each measurement column within the time range
    max_values = {}
    if not selected_data.empty:
        for exp_col in experiment_columns:
            max_values[exp_col] = selected_data[exp_col].max()
    else:
        # If no rows match, set the maximum as None
        for exp_col in experiment_columns:
            max_values[exp_col] = None

    # Store results
    result['Event Name'].append(event_name)
    for exp_col in experiment_columns:
        result[exp_col].append(max_values[exp_col])

# Convert results to DataFrame
result_df = pd.DataFrame(result)

# Output file path
output_file = input("Enter the path for the output file: ").strip()
result_df.to_excel(output_file, index=False)
print(f"Done. Results saved to: {output_file}")

# Ask user for chart title
chart_title = input("Enter chart title: ").strip()

# Plot line chart using result_df
# X-axis: experiment_columns
# Y-axis: corresponding values

# Prepare a color list (should be at least as many as the number of events)
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#9e9e9e'
]
# If there are more events than colors, extend the list as needed
while len(colors) < len(result_df):
    colors += colors  # simple repetition; consider optimizing in real use

# Use indices to replace experiment_columns as X-axis data
x = range(len(experiment_columns))

plt.figure(figsize=(20, 6))

for index, row in result_df.iterrows():
    event_name = row['Event Name']
    values = row[experiment_columns].values
    plt.plot(experiment_columns, values, marker='.', label=event_name, color=colors[index])

# Extract the numeric part as new tick labels
new_labels = [col.split('_')[-1] for col in experiment_columns]

plt.xticks(x, new_labels)  # replace X-axis tick labels with numeric parts
plt.title(chart_title)
plt.xlabel('Resolution')
plt.ylabel('Value')
plt.legend(title='PaleoClimate Events', loc='upper left')
plt.tight_layout()
plt.show()
