import h5py
import numpy as np
import os
import pandas as pd # Thư viện mạnh mẽ cho làm việc với dữ liệu dạng bảng và CSV

# --- Configuration ---
# Đường dẫn đến tệp HDF5 đã trích xuất
HDF5_FILE_PATH = 'extracted_dataset_structured.h5' # <<< UPDATE THIS PATH if needed

# Thư mục để lưu các tệp CSV
# Sẽ tạo thư mục này nếu nó chưa tồn tại
OUTPUT_CSV_DIR = 'extracted_csv_data' # <<< UPDATE THIS PATH if needed

# --- Conversion Script ---

print(f"Attempting to convert data from HDF5 file: {HDF5_FILE_PATH}")
print(f"Output CSV files will be saved in: {OUTPUT_CSV_DIR}")

# Tạo thư mục output nếu chưa có
if not os.path.exists(OUTPUT_CSV_DIR):
    os.makedirs(OUTPUT_CSV_DIR)
    print(f"Created output directory: {OUTPUT_CSV_DIR}")

processed_count = 0
skipped_no_data_count = 0
error_count = 0

try:
    # Mở tệp HDF5 ở chế độ chỉ đọc ('r')
    with h5py.File(HDF5_FILE_PATH, 'r') as f:

        # Duyệt qua tất cả các Group ở cấp cao nhất (mỗi Group là một tệp gốc)
        for group_name in f.keys():
            group = f[group_name]
            relative_filepath = group.attrs.get('relative_filepath_str', group_name) # Get original path if available

            print(f"\nProcessing HDF5 Group: {group_name} (Original: {relative_filepath})")

            # Kiểm tra xem Group này có chứa Dataset dữ liệu cảm biến không (MAT files with data)
            # Dữ liệu cảm biến nằm trong Group con 'sensor_values'
            if 'sensor_values' in group and isinstance(group['sensor_values'], h5py.Group):
                sensor_values_group = group['sensor_values']

                # Kiểm tra xem trong Group sensor_values có Dataset nào không
                # và ít nhất một Dataset có size > 0
                sensor_datasets = {
                    name: sensor_values_group[name]
                    for name in sensor_values_group.keys()
                    if isinstance(sensor_values_group[name], h5py.Dataset) and sensor_values_group[name].size > 0
                }

                # Kiểm tra Dataset timestamps
                timestamps_dataset = None
                if 'timestamps' in group and isinstance(group['timestamps'], h5py.Dataset) and group['timestamps'].size > 0:
                     timestamps_dataset = group['timestamps']
                # else:
                     # print(f"  Warning: No timestamps dataset found for {group_name}. Skipping CSV conversion for this group.")
                     # skipped_no_data_count += 1
                     # continue # Skip if no timestamps

                # Nếu có ít nhất một Dataset cảm biến không rỗng VÀ có timestamps
                if sensor_datasets and timestamps_dataset:
                    try:
                        # Đọc dữ liệu
                        timestamps_array = timestamps_dataset[:] # Read the whole dataset
                        # Lấy tên Dataset cảm biến đầu tiên có dữ liệu để xác định số mẫu
                        first_sensor_name = list(sensor_datasets.keys())[0]
                        first_sensor_array = sensor_datasets[first_sensor_name][:] # Read the whole dataset

                        # Kiểm tra số lượng mẫu khớp nhau giữa timestamps và sensor data
                        if len(timestamps_array) != first_sensor_array.shape[0]:
                             print(f"  Warning: Timestamps length ({len(timestamps_array)}) does not match sensor data length ({first_sensor_array.shape[0]}) for {group_name}. Skipping CSV conversion.")
                             skipped_no_data_count += 1
                             continue

                        # Chuẩn bị dữ liệu cho DataFrame
                        # DataFrame cần các cột: Timestamp, Sensor1_Ch1, Sensor1_Ch2, ..., Sensor2_Ch1, ...
                        data_for_df = {
                            'Timestamp': timestamps_array
                        }

                        # Thêm dữ liệu cảm biến. Xử lý cả 1D và 2D sensor data
                        for sensor_name, dataset in sensor_datasets.items():
                             sensor_data_array = dataset[:] # Read the whole dataset
                             if sensor_data_array.ndim == 1:
                                  # Nếu 1D, tạo 1 cột
                                  data_for_df[sensor_name] = sensor_data_array
                             elif sensor_data_array.ndim == 2:
                                  # Nếu 2D (N, M), tạo M cột
                                  for i in range(sensor_data_array.shape[1]):
                                       data_for_df[f"{sensor_name}_Ch{i+1}"] = sensor_data_array[:, i]
                             else:
                                  print(f"  Warning: Sensor '{sensor_name}' in {group_name} has unexpected dimension {sensor_data_array.ndim}. Skipping for CSV.")
                                  # Remove this sensor from data_for_df if already added partially
                                  for key in list(data_for_df.keys()):
                                       if key.startswith(sensor_name): del data_for_df[key]
                                  continue # Skip this sensor, continue with others

                        # Tạo DataFrame
                        df = pd.DataFrame(data_for_df)

                        # Xác định tên tệp CSV đầu ra
                        # Sử dụng tên Group và thay thế __ và _ cuối cùng thành .mat/.tdms
                        csv_filename = group_name.replace('__', os.sep).replace('_mat', '.mat').replace('_tdms', '.tdms') + '.csv'
                        # Đảm bảo tên tệp cuối cùng an toàn và chỉ lấy tên tệp, không có đường dẫn thư mục gốc
                        csv_filename = csv_filename.split(os.sep)[-1] # Lấy chỉ tên tệp cuối cùng

                        csv_filepath = os.path.join(OUTPUT_CSV_DIR, csv_filename)

                        # Lưu DataFrame vào tệp CSV
                        df.to_csv(csv_filepath, index=False) # index=False để không ghi chỉ mục DataFrame vào CSV

                        print(f"  Successfully saved {relative_filepath} to {csv_filepath} (Shape: {df.shape}).")
                        processed_count += 1

                    except Exception as e:
                        print(f"  Error converting or saving CSV for {relative_filepath} (Group: {group_name}): {e}")
                        # import traceback; traceback.print_exc() # Uncomment for detailed error
                        error_count += 1

                else:
                     # Group exists but no valid sensor datasets or no timestamps
                     print(f"  Skipping CSV conversion for {relative_filepath} (Group: {group_name}): No non-empty sensor data datasets found or no timestamps.")
                     skipped_no_data_count += 1

            else:
                # Group does not contain 'sensor_values' group, likely a TDMS file with metadata only
                 print(f"  Skipping CSV conversion for {relative_filepath} (Group: {group_name}): Does not contain sensor_values group (Metadata only file).")
                 skipped_no_data_count += 1


except FileNotFoundError:
    print(f"\nError: HDF5 file not found at {HDF5_FILE_PATH}. Please run the extraction script first.")
except Exception as e:
    print(f"\nAn unhandled error occurred during HDF5 to CSV conversion: {e}")
    # import traceback; traceback.print_exc() # Uncomment for detailed error


print(f"\n--- CSV Conversion Summary ---")
print(f"Attempted to convert data from {len(f.keys()) if 'f' in locals() else 0} HDF5 Groups.")
print(f"Successfully converted to CSV files: {processed_count}")
print(f"Skipped (no non-empty data or timestamps): {skipped_no_data_count}")
print(f"Errors during conversion/saving: {error_count}")
print(f"------------------------------")