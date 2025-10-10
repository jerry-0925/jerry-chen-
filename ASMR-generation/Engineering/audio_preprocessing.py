import os
import csv

def list_and_clean_files(directory, output_csv, recursive=False):
    file_list = []

    for root, _, files in os.walk(directory):
        for file in files:
            original_path = os.path.join(root, file)

            # Remove leading '=', '-', or '_' characters
            cleaned_name = file.lstrip('=_-')
            cleaned_path = os.path.join(root, cleaned_name)

            # Rename the file if it's different and target doesn't exist
            if file != cleaned_name:
                if not os.path.exists(cleaned_path):
                    os.rename(original_path, cleaned_path)
                    print(f"Renamed: {file} → {cleaned_name}")
                else:
                    print(f"Skipped rename (target exists): {file}")
            else:
                cleaned_path = original_path

            # Add cleaned filename to list (just name, not full path)
            file_list.append([cleaned_name, ""])

        if not recursive:
            break

    # Sort the list alphabetically (case-insensitive)
    file_list.sort(key=lambda x: x[0].lower())

    # Write to CSV
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "Label"])
        writer.writerows(file_list)

    print(f"\nProcessed {len(file_list)} files. CSV saved to: {output_csv}")

# === Customize these ===
directory_path = "/Users/jerrychen/PycharmProjects/Engineering/audio/all"      # Folder with your files
output_csv_path = "file_list.csv"    # CSV file to create
recursive_search = False             # Set to True for subfolder support

list_and_clean_files(directory_path, output_csv_path, recursive_search)
