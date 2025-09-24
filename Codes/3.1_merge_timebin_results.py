import os
import pandas as pd

def main():
    # Define the range of time-bin values to read in batch
    time_bin_values = list(range(50, 1001, 50))

    # Store each file as a DataFrame indexed by Age_node for alignment
    data_frames = []

    for time_bin_value in time_bin_values:
        # Use a raw f-string to avoid backslash escape issues (e.g., \t)
        file_path = fr"C:\Users\22457\Desktop\test\Mean-Rate\O_timebinrate_{time_bin_value}.xlsx"
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)

            # Check required columns exist
            if "Age-point" in df.columns and "rate" in df.columns:
                # Set 'Age_node' as the index so merges align by it
                df = df[["Age-point", "rate"]].set_index("Age-point")
                # Rename the 'IQR' column to distinguish different time scales
                df.columns = [f"Timescale_{time_bin_value}"]
                data_frames.append(df)
            else:
                print(f"Required columns 'Age_node' or 'IQR' not found in {file_path}.")
        else:
            print(f"File does not exist: {file_path}")

    # Horizontally merge all dataframes by Age_node
    if data_frames:
        # Use an outer join so non-overlapping ages are preserved
        merged_df = pd.concat(data_frames, axis=1, join='outer')
        # Reset index to turn Age_node back into a column
        merged_df.reset_index(inplace=True)

        # Ask user for the output file path
        output_path = input("Please enter the output path for the new Excel file (including the filename): ")

        # Write merged data to a new Excel file
        merged_df.to_excel(output_path, index=False)
        print(f"Merged Excel has been saved to: {output_path}")
    else:
        print("No data was read. Please check the file paths or column names.")

if __name__ == "__main__":
    main()
