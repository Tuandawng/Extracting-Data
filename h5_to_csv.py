import h5py
import numpy as np
import os
import pandas as pd

# --- Configuration ---
HDF5_FILE_PATH = 'extracted_dataset_structured.h5' # Đường dẫn đến tệp HDF5 đã trích xuất
OUTPUT_VIBRATION_CSV_DIR = 'extracted_csv_data/vibration' # Thư mục cho dữ liệu vibration
OUTPUT_ACOUSTIC_CSV_DIR = 'extracted_csv_data/acoustic' # Thư mục cho dữ liệu acoustic

# --- Conversion Script ---

print(f"Attempting to convert data from HDF5 file: {HDF5_FILE_PATH}")
print(f"Output CSV files for vibration will be saved in: {OUTPUT_VIBRATION_CSV_DIR}")
print(f"Output CSV files for acoustic will be saved in: {OUTPUT_ACOUSTIC_CSV_DIR}")

# Tạo thư mục nếu chưa có
if not os.path.exists(OUTPUT_VIBRATION_CSV_DIR):
    os.makedirs(OUTPUT_VIBRATION_CSV_DIR)
    print(f"Created output directory for vibration: {OUTPUT_VIBRATION_CSV_DIR}")

if not os.path.exists(OUTPUT_ACOUSTIC_CSV_DIR):
    os.makedirs(OUTPUT_ACOUSTIC_CSV_DIR)
    print(f"Created output directory for acoustic: {OUTPUT_ACOUSTIC_CSV_DIR}")

processed_count = 0
skipped_no_data_count = 0
error_count = 0

try:
    # Mở tệp HDF5 ở chế độ chỉ đọc
    with h5py.File(HDF5_FILE_PATH, 'r') as f:
        # Duyệt qua tất cả các Group ở cấp cao nhất (mỗi Group là một tệp gốc)
        for group_name in f.keys():
            group = f[group_name]
            relative_filepath = group.attrs.get('relative_filepath_str', group_name)

            print(f"\nProcessing HDF5 Group: {group_name} (Original: {relative_filepath})")

            # Kiểm tra xem Group này có chứa Dataset dữ liệu cảm biến không (MAT files with data)
            if 'sensor_values' in group and isinstance(group['sensor_values'], h5py.Group):
                sensor_values_group = group['sensor_values']

                # Kiểm tra xem trong Group sensor_values có Dataset nào không
                sensor_datasets = {
                    name: sensor_values_group[name]
                    for name in sensor_values_group.keys()
                    if isinstance(sensor_values_group[name], h5py.Dataset) and sensor_values_group[name].size > 0
                }

                # Kiểm tra Dataset timestamps
                timestamps_dataset = None
                if 'timestamps' in group and isinstance(group['timestamps'], h5py.Dataset) and group['timestamps'].size > 0:
                    timestamps_dataset = group['timestamps']

                if sensor_datasets and timestamps_dataset:
                    try:
                        # Đọc dữ liệu
                        timestamps_array = timestamps_dataset[:]
                        first_sensor_name = list(sensor_datasets.keys())[0]
                        first_sensor_array = sensor_datasets[first_sensor_name][:]

                        # Kiểm tra số lượng mẫu khớp nhau giữa timestamps và sensor data
                        if len(timestamps_array) != first_sensor_array.shape[0]:
                            print(f"  Warning: Timestamps length ({len(timestamps_array)}) does not match sensor data length.")
                            skipped_no_data_count += 1
                            continue

                        # Chuẩn bị dữ liệu cho DataFrame
                        data_for_df = {'Timestamp': timestamps_array}

                        # Thêm dữ liệu cảm biến vào DataFrame
                        for sensor_name, dataset in sensor_datasets.items():
                            sensor_data_array = dataset[:]
                            if sensor_data_array.ndim == 1:
                                data_for_df[sensor_name] = sensor_data_array
                            elif sensor_data_array.ndim == 2:
                                for i in range(sensor_data_array.shape[1]):
                                    data_for_df[f"{sensor_name}_Ch{i+1}"] = sensor_data_array[:, i]

                        # Tạo DataFrame
                        df = pd.DataFrame(data_for_df)

                        # Xác định thư mục lưu trữ theo loại cảm biến
                        if 'vibration' in group_name.lower():
                            csv_filepath = os.path.join(OUTPUT_VIBRATION_CSV_DIR, f"{group_name}.csv")
                        elif 'acoustic' in group_name.lower():
                            csv_filepath = os.path.join(OUTPUT_ACOUSTIC_CSV_DIR, f"{group_name}.csv")
                        else:
                            print(f"  Warning: Group {group_name} does not match vibration or acoustic. Skipping.")
                            skipped_no_data_count += 1
                            continue

                        # Lưu DataFrame vào tệp CSV
                        df.to_csv(csv_filepath, index=False)

                        print(f"  Successfully saved to {csv_filepath} (Shape: {df.shape}).")
                        processed_count += 1

                    except Exception as e:
                        print(f"  Error converting or saving CSV for {relative_filepath} (Group: {group_name}): {e}")
                        error_count += 1

                else:
                    print(f"  Skipping {relative_filepath} (Group: {group_name}): No valid sensor data or timestamps.")
                    skipped_no_data_count += 1

            else:
                print(f"  Skipping {relative_filepath} (Group: {group_name}): No sensor_values found.")
                skipped_no_data_count += 1

except FileNotFoundError:
    print(f"\nError: HDF5 file not found at {HDF5_FILE_PATH}. Please run the extraction script first.")
except Exception as e:
    print(f"\nAn error occurred during HDF5 to CSV conversion: {e}")

print(f"\n--- CSV Conversion Summary ---")
print(f"Successfully converted: {processed_count}")
print(f"Skipped (no data or no valid timestamps): {skipped_no_data_count}")
print(f"Errors during conversion: {error_count}")
