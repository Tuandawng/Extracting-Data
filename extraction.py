import os
import scipy.io as sio
from nptdms import TdmsFile # Import for TDMS file handling
import numpy as np
import re # To parse filenames
import sys
# import traceback # Uncomment for detailed error traces if needed

# --- Helper Function to Parse Filename and Extract Metadata ---
def parse_filename(filename, filepath):
    """
    Parses the filename to extract load, condition, severity, and sensor type hint.
    Expected format: aaaaNm_bbbb_cccc.extension. Handles optional severity and variants.
    Also filters out Zone.Identifier files.
    """
    # Filter out Windows Zone.Identifier files - already handled before calling this
    # if filename.endswith(':Zone.Identifier'):
    #     return None

    # Corrected regex based on observed file names:
    # ^(\d+Nm)       - Group 1: Load (digits followed by Nm)
    # _([^_]+)       - Group 2: Condition (one or more characters that are NOT an underscore)
    # (_)?           - Optional Group 3: An underscore (makes the severity part optional)
    # ([^.]*)        - Group 4: Severity (zero or more characters that are NOT a dot, before the extension)
    # \.(mat|tdms)$  - Match dot followed by extension (mat or tdms) and end of string (Group 5 is extension)
    match = re.match(r'^(\d+Nm)_([^_]+)(_)?([^.]*)\.(mat|tdms)$', filename, re.IGNORECASE)
    if match:
        load = match.group(1)      # e.g., '0Nm'
        condition = match.group(2) # e.g., 'BPFI', 'Normal', 'Unbalance'
        # The severity is in group 4 because group 3 was the optional underscore
        severity_match = match.group(4) # e.g., '03', '10', '0583mg', '' (for Normal)
        # If severity_match is empty, set severity to None or a placeholder
        severity = severity_match if severity_match else 'No_Severity' # Assign a placeholder like 'No_Severity' if missing


        file_extension = match.group(5).lower() # Extension is now group 5

        sensor_type = None
        # Infer sensor type based on the directory path (most reliable for this dataset structure)
        filepath_lower = filepath.lower()
        # Use os.sep to make path check work correctly on different OS
        if os.sep + 'acoustic' + os.sep in filepath_lower: # Check for the specific folder name
             sensor_type = 'Acoustic'
        elif os.sep + 'vibration' + os.sep in filepath_lower: # Check for the specific folder name
             sensor_type = 'Vibration'
        elif os.sep + 'current,temp' + os.sep in filepath_lower: # Check for the specific folder name
             sensor_type = 'Temp_Current'
        # Fallback if the file is not in one of these standard folders (less likely for this dataset)
        elif file_extension == 'mat':
             sensor_type = 'Mat_Unknown_Path'
        elif file_extension == 'tdms':
             sensor_type = 'TDMS_Unknown_Path'


        return {
            'filename': filename,
            'load': load,
            'condition': condition,
            'severity': severity,
            'extension': file_extension,
            'sensor_type': sensor_type, # This is the initial hint based on path/name
            'filepath': filepath       # Full path for debugging/access (absolute path)
            # 'relative_filepath': os.path.relpath(filepath, BASE_DATASET_DIRECTORY) # Optional: add relative path to metadata
        }
    else:
        # This line will print filenames that didn't match the expected data file pattern
        # print(f"Skipping: Filename format not recognized: {filename}") # Optional: uncomment for debugging
        return None

# --- Helper Function to print the structure of a MATLAB object/struct ---
# (This function is kept for debugging/inspection, uncomment calls as needed)
# Keep this function UNCOMMENTED if you need to use print_mat_structure for debugging other MAT files
def print_mat_structure(obj, indent=0, name="mat_object"):
    """Recursively prints the structure of a MATLAB object loaded by scipy.io.loadmat."""
    prefix = "  " * indent

    # Check if it's a MATLAB struct - more robust check
    if isinstance(obj, sio.matlab.mio5_params.mat_struct) and hasattr(obj, '_fieldnames'):
        print(f"{prefix}Struct '{name}':")
        if not obj._fieldnames:
            print(f"{prefix}  (No fields)")
        else:
            for field_name in obj._fieldnames:
                try:
                    print_mat_structure(obj[field_name], indent + 1, field_name)
                except Exception as e:
                    print(f"{prefix}  Error accessing field '{field_name}': {e}")

    elif isinstance(obj, np.ndarray):
        print(f"{prefix}NumPy Array '{name}': shape={obj.shape}, dtype={obj.dtype}")
        if obj.dtype.hasobject:
            # print(f"{prefix}  Contains Objects. Exploring first few elements:") # Keep output shorter
            elements_to_show = []
            try:
                if obj.size > 0:
                    # Try accessing first few elements robustly
                    flat_obj = obj.flatten()
                    elements_to_show = [flat_obj[i] for i in range(min(1, flat_obj.size))] # Only show 1 element for brevity
            except Exception as e:
                # print(f"{prefix}    Could not access elements to show: {e}") # Too noisy maybe?
                elements_to_show = []

            for i, elem in enumerate(elements_to_show):
                print(f"{prefix}  Element [{i}] type: {type(elem)}")
                # Recursively print structure for complex types, value for simple ones
                if isinstance(elem, (sio.matlab.mio5_params.mat_struct, np.ndarray, list, tuple, dict, np.void)): # Added np.void
                     print_mat_structure(elem, indent + 2, f"element_[{i}]_content")
                else: # Print value representation
                     value_repr = repr(elem)
                     if len(value_repr) > 100: value_repr = value_repr[:97] + "..."
                     print(f"{prefix}  Element [{i}] value: {value_repr}")
             # If obj.size > len(elements_to_show): print(f"{prefix}  ...") # Keep output shorter
             # if obj.size == 0: print(f"{prefix}  (Empty object array)") # Keep output shorter

    elif isinstance(obj, np.void): # Handle numpy.void (structured array element) directly
         print(f"{prefix}NumPy Void (Struct Record) '{name}': dtype={obj.dtype}")
         if obj.dtype.names:
             for field_name in obj.dtype.names:
                 try:
                     # Access element by field name. Result might be a scalar or array
                     field_value = obj[field_name]
                     print_mat_structure(field_value, indent + 1, field_name)
                 except Exception as e:
                     print(f"{prefix}  Error accessing field '{field_name}': {e}")
         else:
              print(f"{prefix}  (No fields)")


    else:
        # Print basic types and their values
        value_repr = repr(obj)
        if len(value_repr) > 100: value_repr = value_repr[:97] + "..."
        print(f"{prefix}Basic Type '{name}': type={type(obj)}, value={value_repr}")


# --- Function to Extract Data from .mat files (Vibration, Acoustic) ---
def extract_data_from_mat(filepath, metadata):
    """
    Extracts data, timestamp/sample rate from .mat files based on observed structured array nesting.
    Handles a deeper nested format found in vibration/acoustic files.
    """
    extracted_info = {
        'metadata': metadata, # Store the metadata dictionary including filepath
        'sensor_values': {},          # {descriptive_name: np.array}
        'timestamps': None,           # numpy array of timestamps
        'sample_rate': None,          # float
        'inferred_sensor_type': metadata.get('sensor_type', 'Mat_Unknown_Structure') # Start with sensor_type hint from path
    }

    print(f"  Attempting to extract data from .mat file...")

    try:
        mat_data = sio.loadmat(filepath)

        # --- Try to find the main signal container ('Signal' or 'signal') ---
        main_data_key = None
        if 'Signal' in mat_data:
             main_data_key = 'Signal'
        elif 'signal' in mat_data:
             main_data_key = 'signal'
        else:
             print(f"  Error: Neither 'Signal' nor 'signal' key found at root. Keys found: {list(mat_data.keys())}")
             return None # Fail if main key isn't found

        signal_container = mat_data[main_data_key]

        # --- Debugging Mat Structure ---
        # UNCOMMENT THE LINE BELOW TEMPORARILY TO PRINT DETAILED STRUCTURE FOR DEBUGGING MAT FILES
        # print("\n  --- Debugging Mat Structure ---")
        # print_mat_structure(signal_container)
        # print("  --- End Debugging ---\n")
        # --- End Debugging ---


        # --- Handle the observed (1,1) structured array format ---
        main_record = None
        if (isinstance(signal_container, np.ndarray) and
            signal_container.shape == (1, 1) and
            signal_container.dtype.hasobject): # Should be object dtype for structured array fields

             # Access the single record within the (1,1) array
             if signal_container.size > 0:
                 record_candidate = signal_container[0, 0]

                 # Check if this candidate is a structured array record (numpy.void) and has expected fields
                 # Checking for 'x_values' and 'y_values' is a good indicator of this specific structure
                 if isinstance(record_candidate, np.void) and 'x_values' in record_candidate.dtype.names and 'y_values' in record_candidate.dtype.names:
                      main_record = record_candidate
                      # inferred_type already set based on metadata path hint, refine if needed
                      # extracted_info['inferred_sensor_type'] = metadata.get('sensor_type', 'Mat_Structured') # Update type based on format + path

                 else:
                      print(f"  Error: Found (1,1) object array, but element [0,0] is not a numpy.void with 'x_values'/'y_values' fields (type: {type(record_candidate)}, dtype: {getattr(record_candidate, 'dtype', 'N/A')}).")
             else:
                  print(f"  Error: Found (1,1) object array, but it is empty.")


        if main_record is None:
             print(f"  Error: Could not find the expected main data in the structured array format within key '{main_data_key}'.")
             return None # Indicate failure if we can't get the main data record


        # --- Extract Time Information (start, increment, num_values) from 'x_values' ---
        start_value = None
        increment = None
        number_of_values = None

        try:
            # Access the content of 'x_values'. Debug output shows it's likely a (1,1) object array holding another record.
            x_data_container = main_record['x_values']

            if isinstance(x_data_container, np.ndarray) and x_data_container.shape == (1, 1) and x_data_container.dtype.hasobject and x_data_container.size > 0:
                 time_info_struct_candidate = x_data_container[0, 0]

                 # Check if this inner candidate is a numpy.void with time fields
                 if isinstance(time_info_struct_candidate, np.void) and 'start_value' in time_info_struct_candidate.dtype.names and 'increment' in time_info_struct_candidate.dtype.names and 'number_of_values' in time_info_struct_candidate.dtype.names:

                      # Extract the scalar values, handling potential (1,1) array wrapping
                      # Use .flatten()[0] to reliably get the scalar value from potentially nested arrays
                      try:
                          start_value_raw = time_info_struct_candidate['start_value']
                          if isinstance(start_value_raw, np.ndarray) and start_value_raw.size > 0:
                              start_value = float(start_value_raw.flatten()[0])
                          elif np.isscalar(start_value_raw): # Handle direct scalar if present
                              start_value = float(start_value_raw)
                          else: raise ValueError("start_value not found or not scalar/scalar array")


                          increment_raw = time_info_struct_candidate['increment']
                          if isinstance(increment_raw, np.ndarray) and increment_raw.size > 0:
                              increment = float(increment_raw.flatten()[0])
                          elif np.isscalar(increment_raw):
                               increment = float(increment_raw)
                          else: raise ValueError("increment not found or not scalar/scalar array")


                          num_values_raw = time_info_struct_candidate['number_of_values']
                          if isinstance(num_values_raw, np.ndarray) and num_values_raw.size > 0:
                              # number_of_values might be an integer, but int() is safe if it's a number
                              num_values_flat = num_values_raw.flatten()[0]
                              if np.isscalar(num_values_flat):
                                   number_of_values = int(num_values_flat)
                              else: raise ValueError("number_of_values flatten result is not scalar")
                          elif np.isscalar(num_values_raw):
                               number_of_values = int(num_values_raw)
                          else: raise ValueError("number_of_values not found or not scalar/scalar array")


                          print(f"  Extracted time parameters: start={start_value}, increment={increment}, num_values={number_of_values}")

                          # Reconstruct timestamps and calculate sample rate if parameters are valid
                          if increment is not None and increment > 0 and number_of_values is not None and number_of_values > 0:
                              extracted_info['sample_rate'] = 1.0 / increment
                              extracted_info['timestamps'] = np.arange(number_of_values) * increment + start_value
                              print(f"  Reconstructed timestamps array of shape {extracted_info['timestamps'].shape} with sample rate {extracted_info['sample_rate']:.2f} Hz.")
                          else:
                               print("  Warning: Invalid time parameters found (increment <= 0 or num_values <= 0). Cannot reconstruct time info.")


                      except (KeyError, ValueError, TypeError) as e:
                           print(f"  Warning: Error extracting scalar values from time info struct fields: {e}. Cannot reconstruct time info.")
                      except Exception as e:
                           print(f"  Warning: An unexpected error occurred during time parameter extraction: {e}. Cannot reconstruct time info.")


                 else:
                      print(f"  Warning: 'x_values' content at [0,0] is not a numpy.void with time fields (type: {type(time_info_struct_candidate)}, dtype: {getattr(time_info_struct_candidate, 'dtype', 'N/A')}). Cannot reconstruct time info.")

            else:
                 print(f"  Warning: 'x_values' field had unexpected type/shape {type(x_data_container)} {getattr(x_data_container, 'shape', 'N/A')}. Expected (1,1) object array. Cannot reconstruct time info.")
            # Note: We don't return failure here yet, as we might still get sensor data without time/rate

        except KeyError:
             print(f"  Warning: Could not find 'x_values' field in the structured record. Time info missing.")
        except Exception as e:
             print(f"  Warning: Error accessing 'x_values' field: {e}. Time info may be incomplete.")


        # --- Extract Sensor Values from 'y_values' ---
        try:
            # Access the content of 'y_values'. Debug output suggests it's a (1,1) object array holding another record.
            y_data_container = main_record['y_values']

            if isinstance(y_data_container, np.ndarray) and y_data_container.shape == (1, 1) and y_data_container.dtype.hasobject and y_data_container.size > 0:
                 sensor_data_struct_candidate = y_data_container[0, 0]

                 # Check if this inner candidate is a numpy.void with a 'values' field
                 if isinstance(sensor_data_struct_candidate, np.void) and 'values' in sensor_data_struct_candidate.dtype.names:

                      # Access the actual data array, handling potential (1,1) array wrapping
                      values_array_candidate = sensor_data_struct_candidate['values']

                      # Check if the final content is a numeric numpy array
                      if isinstance(values_array_candidate, np.ndarray) and values_array_candidate.dtype.kind in 'fiu' and values_array_candidate.size > 0:
                           # Determine a descriptive name based on the sensor type hint
                           sensor_name_suffix = "_Signal"
                           # Keep original sensor type hint from path if available
                           sensor_type_hint = metadata.get('sensor_type', 'Mat')
                           if sensor_type_hint == 'Vibration':
                                extracted_info['inferred_sensor_type'] = 'Vibration (Structured Mat)'
                                sensor_name = 'Vibration' + sensor_name_suffix
                           elif sensor_type_hint == 'Acoustic':
                                extracted_info['inferred_sensor_type'] = 'Acoustic (Structured Mat)'
                                sensor_name = 'Acoustic' + sensor_name_suffix
                           else:
                                extracted_info['inferred_sensor_type'] = 'Mat_Structured_Unknown'
                                sensor_name = sensor_type_hint + sensor_name_suffix


                           # Ensure data is in (N, M) shape where N is samples, M is channels.
                           # If it's 1D (N,), make it (N, 1)
                           # If it's >2D, flatten to (N, 1)
                           sensor_values_array = values_array_candidate
                           if values_array_candidate.ndim == 1:
                                sensor_values_array = values_values_array_candidate.reshape(-1, 1)
                           elif values_array_candidate.ndim > 2:
                                print(f"  Warning: 'y_values' content 'values' had unexpected dimension {values_array_candidate.ndim}. Flattening to (N, 1).")
                                sensor_values_array = values_array_candidate.flatten().reshape(-1, 1)


                           extracted_info['sensor_values'][sensor_name] = sensor_values_array
                           print(f"  Extracted sensor data '{sensor_name}' from 'y_values' ('values' field) (shape {sensor_values_array.shape}).")

                           # Cross-check number of samples with number_of_values from x_values
                           if number_of_values is not None and sensor_values_array.shape[0] != number_of_values:
                               # This warning is useful, but doesn't necessarily mean failure if sensor data was read
                               print(f"  Warning: Number of samples from 'y_values' ({sensor_values_array.shape[0]}) does not match 'number_of_values' from 'x_values' ({number_of_values}). Using sensor data length for timestamps if needed.")

                           # Check if the sensor data array is actually empty (size 0) - already checked with values_array_candidate.size > 0


                      else:
                           print(f"  Error: 'y_values' content 'values' had unexpected type/shape {type(values_array_candidate)} {getattr(values_array_candidate, 'shape', 'N/A')} {getattr(values_array_candidate, 'dtype', 'N/A')}. Cannot extract sensor values.")
                           extracted_info['sensor_values'] = {} # Ensure empty on failure

                 else:
                      print(f"  Error: 'y_values' content at [0,0] is not a numpy.void with 'values' field (type: {type(sensor_data_struct_candidate)}, dtype: {getattr(sensor_data_struct_candidate, 'dtype', 'N/A')}). Cannot extract sensor values.")
                      extracted_info['sensor_values'] = {} # Ensure empty on failure

            else:
                 print(f"  Error: 'y_values' field had unexpected type/shape {type(y_data_container)} {getattr(y_data_container, 'shape', 'N/A')}. Expected (1,1) object array. Cannot extract sensor values.")
                 extracted_info['sensor_values'] = {} # Ensure empty on failure


        except KeyError:
             print(f"  Error: Could not find 'y_values' field in the structured record or 'values' field in the inner struct. Cannot extract sensor values.")
             extracted_info['sensor_values'] = {} # Ensure empty on failure
        except Exception as e:
             print(f"  Error extracting sensor data from 'y_values': {e}.")
             # import traceback; traceback.print_exc()
             extracted_info['sensor_values'] = {} # Ensure empty on failure


        # --- Final Checks and Cleanups ---

        # If timestamps were not reconstructed but sample rate is known AND we have sensor data, regenerate a placeholder time vector
        # This is a fallback if 'x_values' structure is different or extraction fails for time info
        if extracted_info['timestamps'] is None and extracted_info['sample_rate'] is not None and extracted_info['sensor_values'] and any(arr.size > 0 for arr in extracted_info['sensor_values'].values()):
             print("  Timestamps not found from 'x_values', but sample rate exists and non-empty sensor data was extracted. Generating time vector from sensor data length.")
             # Get number of samples from the first extracted NON-EMPTY sensor array
             first_non_empty_sensor_name = next((name for name, arr in extracted_info['sensor_values'].items() if arr.size > 0), None)
             if first_non_empty_sensor_name:
                 num_samples = extracted_info['sensor_values'][first_non_empty_sensor_name].shape[0]
                 # Assume start time is 0 if not found/extracted
                 extracted_info['timestamps'] = np.arange(num_samples) / extracted_info['sample_rate']
                 print(f"  Generated timestamps array of shape {extracted_info['timestamps'].shape}.")
             # else: Warning already printed if no non-empty data found

        # If sample rate wasn't calculated from increment but timestamps were generated, calculate from timestamps as a fallback
        # Ensure timestamps exist AND we have sensor data
        elif extracted_info['sample_rate'] is None and extracted_info['timestamps'] is not None and len(extracted_info['timestamps']) > 1 and extracted_info['sensor_values'] and any(arr.size > 0 for arr in extracted_info['sensor_values'].values()):
             print("  Sample rate not found from 'x_values', but timestamps exist. Calculating sample rate.")
             time_diffs = np.diff(extracted_info['timestamps'])
             positive_diffs = time_diffs[time_diffs > 0]
             if len(positive_diffs) > 0:
                  extracted_info['sample_rate'] = 1.0 / np.mean(positive_diffs)
                  print(f"  Calculated sample rate from timestamps: {extracted_info['sample_rate']:.2f} Hz")
             elif len(np.unique(time_diffs)) == 1 and time_diffs[0] > 0:
                  extracted_info['sample_rate'] = 1.0 / time_diffs[0]
                  print(f"  Calculated sample rate from uniform timestamps: {extracted_info['sample_rate']:.2f} Hz")
             else:
                  print("  Warning: Could not calculate sample rate from generated timestamps (non-uniform/zero differences).")


        # Add sensor names list to info (using the descriptive names) - only for non-empty data
        extracted_info['sensor_names'] = [name for name, arr in extracted_info['sensor_values'].items() if arr.size > 0] # List only sensors with non-empty data

        # Final check for critical missing info *after* fallbacks
        if not extracted_info['sensor_values'] or not extracted_info['sensor_names']:
             print(f"  Final Extraction Result: Failed. No non-empty sensor values successfully extracted from {metadata['filename']}.")
             # Add specific error message if not already present
             if 'extraction_error' not in extracted_info:
                 extracted_info['extraction_error'] = "No non-empty sensor values extracted."
             # Do NOT return None here. Return the info dict with empty sensor_values.
             # The main loop will handle classifying this as metadata_only or failed based on sensor_values content.
             return extracted_info # Return info dict even if empty data

        # Warnings if time info is completely missing but data exists
        if extracted_info['timestamps'] is None:
             print(f"  Warning: Could not find or generate timestamps for {metadata['filename']}, but sensor data was extracted.")

        if extracted_info['sample_rate'] is None:
             print(f"  Warning: Could not find or derive sample rate for {metadata['filename']}, but sensor data was extracted.")


    except FileNotFoundError:
        print(f"  Error: .mat file not found at {filepath}")
        extracted_info['extraction_error'] = "File not found."
        return extracted_info # Return info dict even if file not found
    except Exception as e:
        print(f"  An unexpected error occurred while processing {filepath}: {e}")
        # import traceback; traceback.print_exc() # Uncomment for detailed error
        extracted_info['extraction_error'] = f"Unhandled error: {e}"
        return extracted_info # Return info dict even on unhandled error

    print(f"  Final Extraction Result: Success. Successfully extracted data from .mat file.")
    return extracted_info


# --- Function to Extract Data from .tdms files (Temp_Current) ---
# Keep this function AS IS from the previous version.
# It will return extracted_info even if sensor_values is empty.
def extract_data_from_tdms(filepath, metadata, tdms_group_name, tdms_channel_names):
     """
     Extracts data, timestamp/sample rate, and channel properties from .tdms files.
     Uses properties to generate more descriptive sensor names.
     Explicitly handles read errors for individual channels.
     Returns info even if no sensor values could be read.
     """
     extracted_info = {
          'metadata': metadata, # Store the metadata dictionary including filepath
          'sensor_values': {},          # {descriptive_name: np.array} - Only non-empty data stored here
          'raw_channel_names': [],      # List of original DAQ channel names found and successfully read data from (non-empty)
          'channel_properties': {},     # {descriptive_name: {property_key: value}} - Properties for channels with non-empty data
          'all_channel_properties_raw': {}, # {raw_channel_name: {property_key: value}} - Properties for ALL configured channels found
          'timestamps': None,           # numpy array of timestamps
          'sample_rate': None,          # float
          'inferred_sensor_type': 'Temp_Current (TDMS)' # TDMS files have a consistent type here
     }

     print(f"  Attempting to extract data from .tdms file...")

     try:
          with TdmsFile.open(filepath) as tdms_file:

              # --- TDMS DEBUGGING: Print all groups ---
              # print(f"  TDMS Groups found: {[g.name for g in tdms_file.groups()]}") # <-- UNCOMMENT THIS LINE FOR DEBUGGING TDMS FILES
              # --- End TDMS Debugging ---

              found_group = None
              # all_group_names_in_file = [g.name for g in tdms_file.groups()] # Avoid re-creating this list if not debugging
              for g in tdms_file.groups():
                  if g.name == tdms_group_name:
                       found_group = g
                       break

              if found_group is None:
                   print(f"  Error: TDMS group '{tdms_group_name}' not found in this file.")
                   # Return partial info with error indication
                   extracted_info['extraction_error'] = f"Group '{tdms_group_name}' not found."
                   return extracted_info # Return info even if group not found

              print(f"  Successfully found TDMS group: '{found_group.name}'")

              # --- TDMS DEBUGGING: Print all channels in the found group ---
              # UNCOMMENT THE LINE BELOW FOR DEBUGGING TDMS FILES
              # print(f"  Channels found in group '{found_group.name}': {[c.name for c in found_group.channels()]}")
              # --- End TDMS Debugging ---


              temp_sample_rate = None      # Sample rate found from properties

              # First pass: Collect properties for *all* configured channels found in the group
              configured_channels_found_objs = [] # Store channel objects for configured names found
              for channel_obj in found_group.channels(): # Iterate all channels in the group
                   if channel_obj.name in tdms_channel_names: # Check if this channel is one we care about
                        configured_channels_found_objs.append(channel_obj)
                        # Store properties using raw name
                        extracted_info['all_channel_properties_raw'][channel_obj.name] = dict(channel_obj.properties)

                        # Try to get sample rate hint from this channel's properties
                        if temp_sample_rate is None and 'wf_increment' in channel_obj.properties and channel_obj.properties['wf_increment'] is not None and channel_obj.properties['wf_increment'] > 0:
                             try:
                                 temp_sample_rate = 1.0 / float(channel_obj.properties['wf_increment'])
                                 # print(f"  Found sample rate hint from channel '{channel_obj.name}' properties: {temp_sample_rate:.2f} Hz") # Optional debug
                             except (ValueError, TypeError) as e:
                                  print(f"  Warning: Could not convert 'wf_increment' for channel '{channel_obj.name}' to float: {e}")
                                  temp_sample_rate = None # Reset if conversion failed


              extracted_info['sample_rate'] = temp_sample_rate # Set sample rate if found


              if not configured_channels_found_objs:
                   print(f"  Warning: No configured TDMS channels {tdms_channel_names} were found in group '{tdms_group_name}'. Cannot read data.")
                   # Return info with empty sensor_values and properties, but potentially sample_rate/metadata
                   extracted_info['extraction_warning'] = "Configured channels not found in group."
                   return extracted_info # Return info even if no configured channels found


              # Second pass: Attempt to read DATA for configured channels found
              channel_datas = {} # {raw_channel_name: data_array} - Only non-empty data stored here

              print(f"  Attempting to read data for {len(configured_channels_found_objs)} configured channels found...")

              # Iterate through the channel objects that match our configured names
              for channel_obj in configured_channels_found_objs:
                   try:
                        # Attempt to read the data for THIS channel
                        channel_data = channel_obj.data # This is where the "Channel data has not been read" error occurs

                        # Check if the data is not None and has size > 0
                        if isinstance(channel_data, np.ndarray) and channel_data.size > 0:
                            channel_datas[channel_obj.name] = channel_data # Store only if data is non-empty
                            extracted_info['raw_channel_names'].append(channel_obj.name) # Track raw names with non-empty data
                            print(f"  Successfully read NON-EMPTY data for channel: '{channel_obj.name}' (shape {channel_data.shape})")
                        elif isinstance(channel_data, np.ndarray) and channel_data.size == 0:
                            # print(f"  Successfully read EMPTY data for channel: '{channel_obj.name}' (shape {channel_data.shape}). Skipping for sensor_values.") # Optional debug
                            pass # Do NOT add to channel_datas if empty

                        else:
                             print(f"  Warning: Read data for channel '{channel_obj.name}' is not a numpy array or has unexpected format ({type(channel_data)}). Skipping for sensor_values.")
                             # extracted_info['extraction_warning'] = f"Data for channel {channel_obj.name} has unexpected format." # Add warning? Could get too noisy.


                   except Exception as e:
                       # Catch the specific "Channel data has not been read" or any other read error
                       print(f"  Error reading data for TDMS channel '{channel_obj.name}': {e}. Skipping this channel for sensor_values.")
                       extracted_info.setdefault('channel_read_errors', {})[channel_obj.name] = str(e) # Store the specific read error
                       # import traceback; traceback.print_exc() # Uncomment for debugging read errors


              # --- Process Collected NON-EMPTY Channel Data and Assign Descriptive Names ---
              # We only assign descriptive names and move properties for channels where we got NON-EMPTY data
              print("  Processing collected NON-EMPTY channel data and assigning names...")
              temp_count = 0 # Reset counters for naming based on *collected* data types
              current_count = 0
              processed_channel_names = [] # Keep track of descriptive names added to extracted_info['sensor_values']

              # Iterate through channels whose data was successfully read AND is non-empty
              for raw_channel_name in channel_datas: # Iterate over keys in channel_datas (only non-empty ones)
                   channel_data = channel_datas[raw_channel_name]
                   channel_properties_raw = extracted_info['all_channel_properties_raw'].get(raw_channel_name, {}) # Get properties by raw name

                   descriptive_name = raw_channel_name # Default name if type unknown
                   channel_type = channel_properties_raw.get('DAC~Channel~Type') # Use raw properties for type check

                   if channel_type == 'Temperature':
                       temp_count += 1
                       descriptive_name = f'Temperature_{temp_count}'
                   elif channel_type == 'Current':
                       current_count += 1
                       descriptive_name = f'Current_{current_count}'
                   # Else, use the raw name as descriptive_name

                   # Avoid duplicate descriptive names
                   if descriptive_name in processed_channel_names:
                       descriptive_name = f"{descriptive_name}_{raw_channel_name.replace('/', '_').replace('~', '_')}"

                   # Store the non-empty data under the descriptive name
                   if channel_data.ndim == 1:
                       channel_data = channel_data.reshape(-1, 1)
                   elif channel_data.ndim > 2:
                        print(f"  Warning: Processed channel '{raw_channel_name}' had unexpected dimension {channel_data.ndim}. Using flattened data.")
                        channel_data = channel_data.flatten().reshape(-1, 1)

                   extracted_info['sensor_values'][descriptive_name] = channel_data
                   # Store properties under descriptive name for successfully extracted DATA channels
                   extracted_info['channel_properties'][descriptive_name] = channel_properties_raw # Use the raw properties dictionary
                   processed_channel_names.append(descriptive_name)
                   # print(f"  Stored data for '{descriptive_name}' (shape {channel_data.shape}).") # Uncomment for more detail


              # --- End of Processing Collected NON-EMPTY Channel Data ---

              # --- Finalize Time Information (using sample rate if available) ---
              # If timestamps were not available directly from time_track (which nptdms might provide automatically sometimes)
              # AND sample rate was found (from wf_increment) AND we have non-empty sensor data, generate timestamps
              # Note: nptdms might populate channel_obj.properties['wf_start_time'] etc. which time_track uses.
              # If time_track() worked, extracted_info['timestamps'] would already be set by the default nptdms behavior.
              # Since time_track failed in previous debug, we rely on explicit reconstruction from properties.

              if extracted_info['sample_rate'] is not None and extracted_info['timestamps'] is None:
                  # Generate timestamps ONLY if we have successfully extracted NON-EMPTY sensor data
                  if extracted_info['sensor_values'] and any(arr.size > 0 for arr in extracted_info['sensor_values'].values()):
                      print("  Timestamps not found directly, but sample rate exists and non-empty sensor data was extracted. Attempting to generate time vector from sample rate and sensor data length.")
                      # Get number of samples from the first extracted NON-EMPTY sensor array
                      first_non_empty_sensor_name = next((name for name, arr in extracted_info['sensor_values'].items() if arr.size > 0), None)
                      if first_non_empty_sensor_name:
                          num_samples = extracted_info['sensor_values'][first_non_empty_sensor_name].shape[0]

                          # Try to get start_offset from the properties of one of the channels used for data (using its raw name)
                          start_offset = 0.0 # Default start time
                          # Find properties for the first channel whose data was successfully extracted (using raw name)
                          if extracted_info['raw_channel_names']:
                               first_raw_name_with_data = extracted_info['raw_channel_names'][0]
                               raw_props_for_offset = extracted_info['all_channel_properties_raw'].get(first_raw_name_with_data, {})
                               # Use .get() with default 0.0 in case property is missing or None
                               start_offset = raw_props_for_offset.get('wf_start_offset', 0.0)
                               # Ensure start_offset is a number
                               try: start_offset = float(start_offset)
                               except (ValueError, TypeError): start_offset = 0.0


                          if num_samples > 0:
                               # Generate timestamps: start_offset + index * (1/sample_rate)
                               extracted_info['timestamps'] = np.arange(num_samples) * (1.0 / extracted_info['sample_rate']) + start_offset
                               print(f"  Generated timestamps from sample rate and properties (shape {extracted_info['timestamps'].shape}).")
                          else:
                               # This case should ideally not happen if extracted_info['sensor_values'] is not empty but size is 0
                               print("  Warning: No non-empty sensor data length found to generate timestamps based on sample rate.")
                  # Else: no non-empty sensor values were extracted, so can't generate timestamps even with sample rate


              # Add sensor names list (using descriptive names) - only for non-empty data
              extracted_info['sensor_names'] = [name for name, arr in extracted_info['sensor_values'].items() if arr.size > 0]


          # The 'with TdmsFile.open(...) as tdms_file:' block ensures the file is closed
          print(f"  Finished processing TDMS file. Non-empty sensor values extracted: {bool(extracted_info['sensor_values'])}.")
          return extracted_info # Always return extracted_info if file was opened successfully

     except FileNotFoundError:
         print(f"  Error: TDMS file not found at {filepath}")
         extracted_info['extraction_error'] = "File not found."
         return extracted_info # Return partial info even if file not found
     except Exception as e:
         print(f"  An unexpected unhandled error occurred while processing {filepath}: {e}")
         # import traceback; traceback.print_exc() # Uncomment for detailed error
         extracted_info['extraction_error'] = f"Unhandled error: {e}"
         return extracted_info # Return partial info even on unhandled error


# --- Configuration ---
# Replace with the path to the root folder of your dataset.
# Example: BASE_DATASET_DIRECTORY = '/path/to/your/downloaded/ztmf3m7h5x/'
BASE_DATASET_DIRECTORY = '/home/dangtuan/projects/multiagent-maintenance/project_dataset' # <<< --- UPDATE THIS PATH ---

# --- IMPORTANT for TDMS files (Temperature, Motor Current) ---
# Based on your inspection output, the group is 'Log' and channels are cDAQ names.
TDMS_GROUP_NAME = 'Log'
TDMS_CHANNEL_NAMES = [
    'cDAQ9185-1F486B5Mod1/ai0', # Based on inspection, likely Temperature 1
    'cDAQ9185-1F486B5Mod1/ai1', # Based on inspection, likely Temperature 2
    'cDAQ9185-1F486B5Mod2/ai0', # Based on inspection, likely Current 1
    'cDAQ9185-1F486B5Mod2/ai2', # Based on inspection, likely Current 2
    'cDAQ9185-1F486B5Mod2/ai3'  # Based on inspection, likely Current 3
    # There might be other channels like 'cDAQ9185-1F486B5Mod2/ai1', etc.
    # If you need more, inspect a TDMS file and add them here if they are under 'Log'.
    # NOTE: You might need to add more channel names here if other TDMS files use different cDAQ modules or channels!
    # Based on your TDMS output, these seem to be the only relevant channels used across the dataset for temp/current.
]

# Dictionary to store extracted data for all files that returned an extracted_info dictionary (Success with Data or Metadata Only)
all_extracted_info = {}

# Lists to store relative paths based on final processing outcome categories
processed_with_data_files = []
processed_metadata_only_files = []
skipped_initial_files = [] # Files that didn't match the parse_filename regex or were hidden/system

print(f"Starting data extraction from base directory: {BASE_DATASET_DIRECTORY}")

total_files_in_directory = 0
relevant_files_attempted = 0 # Files matching parse_filename

# --- Main Extraction Loop - Iterate through subdirectories ---
for root, _, files in os.walk(BASE_DATASET_DIRECTORY):
    for filename in files:
        total_files_in_directory += 1
        filepath = os.path.join(root, filename)
        relative_filepath = os.path.relpath(filepath, BASE_DATASET_DIRECTORY)

        # Skip hidden files or system files
        if filename.startswith('.') or filename.endswith(':Zone.Identifier'):
            # print(f"Skipping hidden/system file: {relative_filepath}") # Optional debug
            skipped_initial_files.append(relative_filepath)
            continue

        # --- TEMPORARY DEBUG FILTER: Process only one specific file at a time for debugging ---
        # UNCOMMENT the following lines and REPLACE the path with the SPECIFIC file
        # you want to debug (e.g., one failed .mat or one failed .tdms)
        # After debugging, COMMENT these lines out again to process all files.

        # target_debug_file = 'acoustic/0Nm_BPFI_03.mat' # <-- REPLACE WITH YOUR DEBUG FILE PATH
        #
        # if relative_filepath != target_debug_file:
        #      # Optional: print that we are skipping this file
        #      # print(f"   --> Skipping {relative_filepath} (not target debug file)")
        #      continue # Skip this file if it's not the one we want to debug
        #
        # # If we reach here, it means relative_filepath == target_debug_file
        # print(f"\n--- DEBUG MODE: Found target file, processing: {relative_filepath} ---")
        # --- END TEMPORARY DEBUG FILTER ---


        metadata = parse_filename(filename, filepath)

        # Check if file matches the pattern and is one of the relevant extensions
        if metadata is None or metadata.get('extension') not in ['mat', 'tdms']:
             # print(f"Skipping {relative_filepath}: Does not match expected filename pattern or extension.") # Optional: uncomment to see files skipped by parse_filename
             skipped_initial_files.append(relative_filepath) # Add to skipped list
             continue

        # --- File is relevant, attempt extraction ---
        relevant_files_attempted += 1
        print(f"\nProcessing relevant file {relevant_files_attempted}: {relative_filepath}")

        extracted_info = None
        extraction_successful = False # Flag if the extraction function returned a non-None dictionary

        if metadata['extension'] == 'mat':
             try:
                 extracted_info = extract_data_from_mat(filepath, metadata)
                 if extracted_info is not None: # Function returned a dictionary
                     extraction_successful = True
             except Exception as e:
                  # This should ideally be caught within the function, but as a safety net
                  print(f"  An unhandled exception occurred during MAT processing for {filename}: {e}")
                  # import traceback; traceback.print_exc()
                  extracted_info = {'metadata': metadata, 'extraction_error': f'Unhandled exception during MAT processing: {e}'}
                  extraction_successful = True # We have an info dict, even if it signals an error


        elif metadata['extension'] == 'tdms':
             # Ensure TDMS group name is configured before processing TDMS files
             if TDMS_GROUP_NAME == 'REPLACE_WITH_ACTUAL_TDMS_GROUP_NAME':
                 print(f"  TDMS GROUP NAME NOT UPDATED IN CONFIGURATION! Skipping extraction for {relative_filepath}.")
                 extracted_info = {'metadata': metadata, 'extraction_error': 'TDMS GROUP NAME NOT CONFIGURED'}
                 extraction_successful = True # We have an info dict for reporting
             else:
                try:
                     # Ensure nptdms is installed if processing TDMS
                     if 'nptdms' not in sys.modules: # Check if the module was successfully imported
                          print("  Skipping TDMS file: nptdms library not installed or failed to import.")
                          extracted_info = {'metadata': metadata, 'extraction_error': 'nptdms library not available'}
                          extraction_successful = True # We have an info dict for reporting
                     else: # Only attempt extraction if nptdms is available
                         extracted_info = extract_data_from_tdms(filepath, metadata, TDMS_GROUP_NAME, TDMS_CHANNEL_NAMES)
                         if extracted_info is not None: # Function returned a dictionary
                            extraction_successful = True


                except Exception as e:
                     # This should ideally be caught within the function, but as a safety net
                     print(f"  An unhandled exception occurred during TDMS processing for {filename}: {e}")
                     # import traceback; traceback.print_exc() # Uncomment for detailed error
                     extracted_info = {'metadata': metadata, 'extraction_error': f'Unhandled exception during TDMS processing: {e}'}
                     extraction_successful = True # We have an info dict for reporting


        # --- Process the result from extraction function ---
        if extracted_info is not None:
            # The extraction function returned a dictionary (success or contained error info)
            all_extracted_info[relative_filepath] = extracted_info # Store the info dictionary

            # Check if the info dict contains non-empty sensor data
            if extracted_info.get('sensor_values') and any(isinstance(arr, np.ndarray) and arr.size > 0 for arr in extracted_info['sensor_values'].values()):
                 # Contains at least one non-empty sensor data array
                 processed_with_data_files.append(relative_filepath)
                 print(f"    Result: Processed with Data.")
                 print(f"    Inferred Sensor Type: {extracted_info.get('inferred_sensor_type', 'N/A')}")
                 # Use the sensor_names list which only includes names with non-empty data now
                 print(f"    Extracted Sensors: {extracted_info.get('sensor_names', 'N/A')} (Non-empty data)")
                 # Get number of samples from the first extracted NON-EMPTY sensor array
                 first_non_empty_sensor_name = next((name for name, arr in extracted_info['sensor_values'].items() if isinstance(arr, np.ndarray) and arr.size > 0), None)
                 if first_non_empty_sensor_name:
                     num_samples = extracted_info['sensor_values'][first_non_empty_sensor_name].shape[0]
                     print(f"    Number of Samples (first non-empty): {num_samples}")
                 else:
                     print("    Number of Samples: N/A (Logic error, should have non-empty data)") # Should not happen here

                 print(f"    Sample Rate: {extracted_info.get('sample_rate', 'Not found')} Hz")
                 print(f"    Timestamps extracted/generated: {'timestamps' in extracted_info and extracted_info['timestamps'] is not None}")

            else:
                 # Returned info, but no non-empty sensor data was found/extracted (includes cases where function reported an error/warning)
                 processed_metadata_only_files.append(relative_filepath)
                 print(f"    Result: Processed (Metadata Only).")
                 print(f"    Inferred Sensor Type: {extracted_info.get('inferred_sensor_type', 'N/A')}")
                 # List sensors that were found, even if empty data
                 all_found_sensor_names = list(extracted_info.get('sensor_values', {}).keys())
                 print(f"    Extracted Sensors (found, but data was empty or unreadable): {all_found_sensor_names}")
                 print(f"    Sample Rate (hint): {extracted_info.get('sample_rate', 'Not found')} Hz")
                 # Optional: print specific extraction error/warning from the info dict
                 if 'extraction_error' in extracted_info:
                     print(f"    Extraction Error/Reason: {extracted_info['extraction_error']}")
                 elif 'extraction_warning' in extracted_info:
                     print(f"    Extraction Warning: {extracted_info['extraction_warning']}")

        else:
            # This case should be rare with the updated functions returning dictionaries,
            # but it handles scenarios where the extraction function itself returned None.
            # This would typically indicate a critical failure within the function itself.
            print(f"    Result: Extraction Function Returned None (Critical Failure).")
            # These files are not added to all_extracted_info, so they aren't included
            # in processed_with_data_files or processed_metadata_only_files.
            # They are implicitly accounted for by `relevant_files_attempted` minus
            # the sum of the other two lists.

# --- Final Summary ---
print(f"\n--- Processing Summary ---")
print(f"Total files found in base directory: {total_files_in_directory}")
print(f"Files skipped initially (hidden/system or pattern mismatch): {len(skipped_initial_files)}")
print(f"Files matching expected pattern (.mat/.tdms) and attempted: {relevant_files_attempted}")
print(f"  Processed successfully (with non-empty data): {len(processed_with_data_files)}")
print(f"  Processed (metadata/properties only, no non-empty data): {len(processed_metadata_only_files)}")
# Calculate files where extraction function returned None
files_returning_none = relevant_files_attempted - len(processed_with_data_files) - len(processed_metadata_only_files)
print(f"  Extraction function returned None (critical failure): {files_returning_none}") # This count should ideally be 0

print(f"--------------------------")

# Verification checks
total_categorized_at_start = len(skipped_initial_files) + relevant_files_attempted
print(f"Check: Total files found ({total_files_in_directory}) == Sum of initial parse outcomes ({total_categorized_at_start})")
if total_files_in_directory == total_categorized_at_start:
     print("Check successful.")
else:
     print("Check failed: Sum of initial outcomes ({total_categorized_at_start}) does not match total files found ({total_files_in_directory}).")

total_relevant_outcomes = len(processed_with_data_files) + len(processed_metadata_only_files) + files_returning_none
print(f"Check: Relevant files attempted ({relevant_files_attempted}) == Sum of processing outcomes ({total_relevant_outcomes})")
if relevant_files_attempted == total_relevant_outcomes:
    print("Check successful.")
else:
    print("Check failed: Sum of outcomes ({total_relevant_outcomes}) does not match attempted files ({relevant_files_attempted}).")


# --- Print lists of file outcomes ---
print("\n--- Danh sách các tệp đã trích xuất ĐẦY ĐỦ DỮ LIỆU ---")
if processed_with_data_files:
    for i, filepath in enumerate(sorted(processed_with_data_files)):
        print(f"{i + 1}. {filepath}")
else:
    print("Không có tệp nào được trích xuất đầy đủ dữ liệu cảm biến.")

print("\n--- Danh sách các tệp đã trích xuất CHỈ METADATA/PROPERTIES (Không có dữ liệu tín hiệu non-empty) ---")
if processed_metadata_only_files:
    for i, filepath in enumerate(sorted(processed_metadata_only_files)):
        print(f"{i + 1}. {filepath}")
else:
    print("Không có tệp nào được trích xuất chỉ metadata/properties.")

print("\n--- Danh sách các tệp KHÔNG THỂ XỬ LÝ HOẶC BỊ BỎ QUA BAN ĐẦU ---")
# This list combines files skipped initially and files where the extraction function returned None.
# The summary counters give the breakdown. This list just provides the paths for initially skipped files.
print("\n--- Files skipped initially (hidden/system or pattern mismatch) ---")
if skipped_initial_files:
     for i, filepath in enumerate(sorted(skipped_initial_files)):
          print(f"{i+1}. {filepath}")
else:
     print("Không có tệp nào bị bỏ qua ban đầu.")

if files_returning_none > 0:
    print(f"\n--- Files where extraction function returned NONE (critical failure) ---")
    print(f"({files_returning_none} files - details were printed above during processing)")
    # We don't have a specific list of paths for these easily here without storing them earlier.
    # The printed output during processing should give details for these.


# --- How to access the extracted data (Example - Uncomment to use) ---
# print("\n--- Sample Access ---")
# # Use all_extracted_info dictionary now
# # for relative_filepath, data in all_extracted_info.items():
# #     print(f"\nAccessing data for: {relative_filepath}")
# #     print("  Metadata:", data['metadata'])
# #     print("  Inferred Type:", data.get('inferred_sensor_type'))
# #
# #     if data['timestamps'] is not None:
# #         print(f"  Timestamps shape: {data['timestamps'].shape}")
# #         # print("  First 5 timestamps:", data['timestamps'][:5])
# #
# #     print(f"  Sample Rate: {data.get('sample_rate', 'Not found')} Hz")
# #
# #     print("  Sensor Data:")
# #     if 'sensor_values' in data and data['sensor_values']:
# #         for sensor_name, values in data['sensor_values'].items():
# #             print(f"    '{sensor_name}': shape={values.shape}, size={values.size}") # Added size check
# #             # Check if data is non-empty before trying to print values
# #             # if values.size > 0:
# #             #     print(f"      First 5 values: {values.flatten()[:min(5, values.size)]}")
# #
# #             # You can look up properties for TDMS files by the descriptive sensor_name
# #             unit = '?'
# #             if data.get('inferred_sensor_type', '').startswith('Vibration'):
# #                  unit = 'g' # Based on description
# #             elif data.get('inferred_sensor_type', '').startswith('Acoustic'):
# #                  unit = 'Pa' # Based on description
# #             elif data.get('inferred_sensor_type') == 'Temp_Current (TDMS)':
# #                  # Try to get unit from stored channel properties using the descriptive name
# #                  if sensor_name in data.get('channel_properties', {}): # channel_properties only has data for non-empty sensors now
# #                       unit = data['channel_properties'][sensor_name].get('unit_string', '?')
# #                  else:
# #                       unit = 'C/A' # Fallback unit hint
# #             print(f"      Unit (estimated or from properties): {unit}")
# #     else:
# #         print("    No sensor values extracted.")
# #
# #     # If you want raw TDMS channel names and properties (stored by raw name):
# #     # if data.get('inferred_sensor_type', '').endswith('(TDMS)'): # Check if it's any TDMS type
# #     #     print("  Raw TDMS Channel Names (Channels attempted to read non-empty data from):", data.get('raw_channel_names'))
# #     #     # Properties for ALL configured channels found in the group (whether data was readable or not)
# #     #     print("  Properties for ALL configured channels found (by raw name):", data.get('all_channel_properties_raw')) # May be very verbose
# #     #     # Properties for channels with non-empty data (by descriptive name)
# #     #     print("  Properties for channels with NON-EMPTY data (by descriptive name):", data.get('channel_properties')) # May be very verbose
# #
# #     # Print extraction error/warning if present
# #     # if 'extraction_error' in data: print("  Extraction Error:", data['extraction_error'])
# #     # if 'extraction_warning' in data: print("  Extraction Warning:", data['extraction_warning'])
# #     # if 'channel_read_errors' in data: print("  Channel Read Errors (TDMS):", data['channel_read_errors'])
# # print("\n--- End Sample Access ---")


# --- Saving the Extracted Data (Optional - Uncomment to use) ---
# # Saving to HDF5 is highly recommended for large datasets.
# # Make sure you have h5py installed (`pip install h5py`).
#
import h5py # Uncomment if saving to HDF5

hdf5_output_path = 'extracted_dataset_structured.h5'

try:
    print(f"\nSaving extracted data structure to {hdf5_output_path}...")
    # Use 'all_extracted_info' dictionary now
    with h5py.File(hdf5_output_path, 'w') as f:

        for relative_filepath, data in all_extracted_info.items():
            # We store all_extracted_info, so this loop includes files with data and metadata only.
            # The HDF5 saving logic should handle what gets saved based on data content.

            # Create a group for each file. Use relative path, but make it HDF5-safe.
            # Replace / with __, . with _, and other potentially problematic characters
            group_name = relative_filepath.replace(os.sep, '__').replace('.', '_').replace(' ', '_').replace('-', '_').replace(':', '_').replace('[', '').replace(']', '').replace('<', '_').replace('>', '_').replace('\'', '').replace('\"', '').replace('~', '_') # Added ~
            # Ensure it doesn't start with a reserved character (like _) or is empty
            if not group_name or not group_name[0].isalnum(): group_name = 'file_' + group_name.lstrip('_')
            # Ensure group name is not empty after cleaning
            if not group_name:
                print(f"Warning: Generated empty HDF5 group name for {relative_filepath}. Skipping save for this file.")
                continue

            try:
                file_group = f.create_group(group_name)
                # print(f"  Created group: {group_name}") # Optional debug
            except Exception as e:
                 print(f"Error creating HDF5 group for {relative_filepath} (name: {group_name}): {e}. Skipping save for this file.")
                 continue

            # Store metadata as attributes on the group
            if 'metadata' in data:
                meta_dict = data['metadata'].copy()
                # Remove 'filepath' from metadata before saving as attribute if you prefer relative paths
                if 'filepath' in meta_dict: del meta_dict['filepath']
                # Add relative path as an attribute if it wasn't in metadata originally
                if 'relative_filepath' not in meta_dict:
                     meta_dict['relative_filepath_str'] = relative_filepath # Save as string attribute

                for key, value in meta_dict.items():
                    # Clean keys to be HDF5-safe attributes (should be strings)
                    attr_key = key.replace('.', '_').replace('~', '_').replace(' ', '_').replace('-', '_').replace(':', '_').replace('[', '').replace(']', '')
                    if not attr_key or not attr_key[0].isalnum(): attr_key = 'attr_' + attr_key.lstrip('_') # Safe attribute key
                    # Convert complex types or numpy object arrays to something savable as HDF5 attribute
                    value_to_save = None
                    try:
                        if isinstance(value, np.ndarray):
                             if value.dtype.hasobject:
                                  value_to_save = str(value) # Convert object arrays to string representation
                             elif value.ndim == 0: # Handle numpy scalar arrays
                                  value_to_save = value.item() # Extract Python scalar
                             else: # Attempt to save small numeric arrays directly
                                 # Check if array is small enough and contains only simple types
                                 if value.size < 10 and np.issubdtype(value.dtype, np.number):
                                       value_to_save = value
                                 else:
                                      value_to_save = str(value) # Too large or complex array to save as attribute

                        elif isinstance(value, bytes):
                             value_to_save = value.decode('utf-8', errors='ignore')
                        elif isinstance(value, (list, tuple)):
                             # Attempt to convert list/tuple of simple types to numpy array for attribute
                             try:
                                if all(isinstance(i, (str, int, float, np.number, np.bool_)) for i in value):
                                     # For lists of strings, use h5py's string dtype
                                     if all(isinstance(i, str) for i in value):
                                          value_to_save = np.array(value, dtype=h5py.string_dtype(encoding='utf-8'))
                                     else: # For lists of numbers/bools
                                          value_to_save = np.array(value)
                                else:
                                     value_to_save = str(value) # Fallback for mixed/complex lists
                             except Exception:
                                  value_to_save = str(value) # Fallback if conversion fails

                        elif isinstance(value, (str, int, float, bool, np.bool_, np.number)):
                             value_to_save = value # Simple types are fine
                        else:
                             value_to_save = str(value) # Fallback for other complex types

                        # Save the attribute if the value_to_save is not None
                        if value_to_save is not None:
                             try:
                                  # Check if the resulting value_to_save type is supported by HDF5 attributes
                                  # Simple check: h5py attributes generally support scalars, numpy arrays of basic types, strings.
                                  if isinstance(value_to_save, (str, int, float, bool, np.bool_, np.number)):
                                       file_group.attrs[attr_key] = value_to_save
                                  elif isinstance(value_to_save, np.ndarray) and not value_to_save.dtype.hasobject:
                                        file_group.attrs[attr_key] = value_to_save
                                  else:
                                       # Convert to string if complex type is not directly supported
                                       file_group.attrs[attr_key] = str(value_to_save)

                             except Exception as e:
                                  print(f"Warning: Could not save attribute '{key}' ({attr_key}) for {relative_filepath} (value: {value_to_save}, type: {type(value_to_save)}): {e}. Skipping attribute.")

                    except Exception as e:
                         print(f"Warning: Error preparing attribute '{key}' for {relative_filepath} (value: {value}, type: {type(value)}): {e}. Skipping attribute.")


            # Store inferred type as attribute
            if 'inferred_sensor_type' in data and data['inferred_sensor_type'] is not None:
                try:
                    file_group.attrs['inferred_sensor_type'] = data['inferred_sensor_type']
                except Exception as e:
                     print(f"Warning: Could not save 'inferred_sensor_type' attribute for {relative_filepath}: {e}")

            # Store extraction error/warning as attribute if present
            if 'extraction_error' in data:
                try:
                    file_group.attrs['extraction_error'] = data['extraction_error']
                except Exception as e: print(f"Warning: Could not save extraction_error attribute for {relative_filepath}: {e}")
            if 'extraction_warning' in data:
                 try:
                     file_group.attrs['extraction_warning'] = data['extraction_warning']
                 except Exception as e: print(f"Warning: Could not save extraction_warning attribute for {relative_filepath}: {e}")
            if 'channel_read_errors' in data: # Save TDMS channel read errors
                 try:
                      # Convert dict of errors to a list of strings or save as nested attributes/dataset?
                      # Saving as attributes on a sub-group seems best.
                      read_errors_group = file_group.create_group('channel_read_errors')
                      for chan_name, error_msg in data['channel_read_errors'].items():
                           # Save each error as an attribute named after the channel
                           chan_name_safe = chan_name.replace('.', '_').replace('/', '_').replace('\\', '_').replace(' ', '_').replace('-', '_').replace(':', '_').replace('[', '').replace(']', '').replace('<', '_').replace('>', '_').replace('\'', '').replace('\"', '').replace('~', '_')
                           if not chan_name_safe or not chan_name_safe[0].isalnum(): chan_name_safe = 'chan_' + chan_name_safe.lstrip('_')
                           try:
                               read_errors_group.attrs[chan_name_safe] = str(error_msg)
                           except Exception as e:
                               print(f"Warning: Could not save channel read error for '{chan_name}' ({chan_name_safe}) in {relative_filepath}: {e}")
                 except Exception as e:
                      print(f"Warning: Could not create 'channel_read_errors' group for {relative_filepath}: {e}")


            # Store properties for ALL configured channels found (TDMS only)
            if 'all_channel_properties_raw' in data and data['all_channel_properties_raw']:
                 props_group_raw = None
                 try:
                     # Create a group for all raw properties
                     props_group_raw = file_group.create_group('all_channel_properties_raw')
                 except Exception as e:
                      print(f"Warning: Could not create 'all_channel_properties_raw' group for {relative_filepath}: {e}")

                 if props_group_raw:
                     # Iterate through properties stored using the raw names
                     for raw_channel_name, props in data['all_channel_properties_raw'].items():
                          # Create a sub-group for properties of each channel (using raw name as group name)
                          channel_prop_group_name_raw = raw_channel_name.replace('.', '_').replace('/', '_').replace('\\', '_').replace(' ', '_').replace('-', '_').replace(':', '_').replace('[', '').replace(']', '').replace('<', '_').replace('>', '_').replace('\'', '').replace('\"', '').replace('~', '_')
                          if not channel_prop_group_name_raw or not channel_prop_group_name_raw[0].isalnum(): channel_prop_group_name_raw = 'raw_chan_' + channel_prop_group_name_raw.lstrip('_')
                          try:
                              channel_prop_group_raw = props_group_raw.create_group(channel_prop_group_name_raw)
                          except Exception as e:
                               print(f"Warning: Could not create raw channel property group '{channel_prop_group_name_raw}' for channel '{raw_channel_name}' in {relative_filepath}: {e}. Skipping raw channel properties.")
                               continue

                          # Save each property as an attribute in the channel's sub-group
                          for p_key, p_value in props.items():
                               p_key_safe = p_key.replace('.', '_').replace('/', '_').replace('\\', '_').replace(' ', '_').replace('-', '_').replace(':', '_').replace('[', '').replace(']', '').replace('<', '_').replace('>', '_').replace('\'', '').replace('\"', '').replace('~', '_')
                               if not p_key_safe or not p_key_safe[0].isalnum(): p_key_safe = 'p_' + p_key_safe.lstrip('_')

                               # Attempt to save as attribute if simple type
                               # Check if the value is directly supported by HDF5 attributes
                               p_value_to_save = p_value
                               try:
                                    if isinstance(p_value, (str, int, float, bool, np.bool_, np.number)):
                                         pass # Simple types are fine
                                    elif isinstance(p_value, bytes):
                                         p_value_to_save = p_value.decode('utf-8', errors='ignore')
                                    elif isinstance(p_value, np.ndarray) and p_value.ndim == 0 and np.isscalar(p_value.item()): # Handle numpy scalar arrays
                                         p_value_to_save = p_value.item() # Extract Python scalar
                                    else: # Fallback for complex types or arrays
                                         p_value_to_save = str(p_value)

                                    # Save the attribute
                                    if p_value_to_save is not None:
                                         channel_prop_group_raw.attrs[p_key_safe] = p_value_to_save

                               except Exception as e:
                                    # Fallback to string if attribute saving failed
                                    try:
                                         channel_prop_group_raw.attrs[p_key_safe] = str(p_value)
                                    except Exception as e_str:
                                         print(f"Warning: Could not save raw property '{p_key}' for channel '{raw_channel_name}' in {relative_filepath} (value: {p_value}, type: {type(p_value)}): {e} / {e_str}. Skipping property.")


            # Store sensor values (each sensor with NON-EMPTY data as a dataset within a 'sensor_values' group)
            # FIX INDENTATION HERE
            if 'sensor_values' in data and data['sensor_values']:
                 # Only create the group if there is at least one non-empty sensor array
                 if any(isinstance(arr, np.ndarray) and arr.size > 0 for arr in data['sensor_values'].values()):
                     sensors_group = None
                     try:
                          sensors_group = file_group.create_group('sensor_values')
                          # print(f"  Created 'sensor_values' group.") # Optional debug
                     except Exception as e:
                          print(f"Error creating sensor values group for {relative_filepath}: {e}")

                     if sensors_group:
                          for sensor_name, values_array in data['sensor_values'].items():
                               if isinstance(values_array, np.ndarray) and values_array.size > 0: # Only save non-empty data
                                    # Ensure sensor name is a valid HDF5 dataset name
                                    dataset_name = sensor_name.replace('.', '_').replace('/', '_').replace('\\', '_').replace(' ', '_').replace('-', '_').replace(':', '_').replace('[', '').replace(']', '').replace('<', '_').replace('>', '_').replace('\'', '').replace('\"', '').replace('~', '_') # Add more replacements for safety
                                    if not dataset_name or not dataset_name[0].isalnum(): dataset_name = 'sensor_' + dataset_name.lstrip('_') # Ensure not empty and doesn't start with invalid chars

                                    # Convert object arrays or non-numeric arrays to something savable (float or string)
                                    values_array_savable = None
                                    if values_array.dtype.hasobject:
                                         print(f"Warning: Sensor '{sensor_name}' in {relative_filepath} has object dtype. Attempting conversion to string for saving.")
                                         try:
                                             # Flatten object array and convert each element to string
                                             values_array_savable = np.array([str(x) for x in values_array.flatten()], dtype=h5py.string_dtype(encoding='utf-8'))
                                         except Exception as conv_e:
                                              print(f"Error converting object array for '{sensor_name}' to string: {conv_e}. Skipping dataset.")
                                              continue # Skip this sensor dataset
                                    elif not np.issubdtype(values_array.dtype, np.number):
                                         print(f"Warning: Sensor '{sensor_name}' in {relative_filepath} has non-numeric dtype {values_array.dtype}. Attempting conversion to float.")
                                         try:
                                              values_array_savable = values_array.astype(float)
                                         except Exception as conv_e:
                                              print(f"Error converting non-numeric array for '{sensor_name}' to float: {conv_e}. Skipping dataset.")
                                              continue
                                    else:
                                        values_array_savable = values_array # Data is already numeric and standard

                                    # Ensure data is 1D or 2D for consistency in HDF5 datasets
                                    if values_array_savable is not None:
                                        if values_array_savable.ndim == 0: # Handle scalar? Make it 1D
                                             values_array_savable = np.array([values_array_savable])
                                        elif values_array_savable.ndim > 2: # Flatten >2D
                                             print(f"Warning: Dataset '{dataset_name}' from '{sensor_name}' in {relative_filepath} has >2D shape {values_array_savable.shape}. Flattening to 1D.")
                                             values_array_savable = values_array_savable.flatten()

                                    if values_array_savable is not None:
                                         try:
                                              # h5py dataset names cannot start with '.'
                                              if dataset_name.startswith('.'): dataset_name = '_' + dataset_name
                                              sensors_group.create_dataset(dataset_name, data=values_array_savable, compression="gzip") # Add compression
                                              # print(f"  Saved dataset '{dataset_name}' (shape {values_array_savable.shape}).") # Optional debug
                                         except Exception as e:
                                              print(f"Error saving dataset '{dataset_name}' for sensor '{sensor_name}' in {relative_filepath}: {e}. Skipping dataset.")
                               else:
                                    # print(f"  Skipping saving dataset '{sensor_name}' - data is empty or None.") # Optional debug
                                    pass # Skip saving empty data

                 # else:
                 #      print(f"  No non-empty sensor values to save for {relative_filepath}.") # Optional debug message


            # Store timestamps if available (only if there was at least one non-empty sensor data array saved)
            if 'timestamps' in data and data['timestamps'] is not None and (sensors_group is not None or (data['sensor_values'] and any(isinstance(arr, np.ndarray) and arr.size > 0 for arr in data['sensor_values'].values()))):
            # Check if timestamps exist AND either the sensors_group was created (meaning non-empty data exists) OR the sensor_values dict *contains* non-empty data (handles case where sensors_group creation failed)
                 try:
                      # Ensure timestamps are numeric and 1D before saving
                      if isinstance(data['timestamps'], np.ndarray) and data['timestamps'].dtype.kind in 'fiu':
                          timestamps_to_save = data['timestamps'].flatten()
                          # Save timestamps dataset at the file group level
                          file_group.create_dataset('timestamps', data=timestamps_to_save, compression="gzip")
                          # print(f"  Saved timestamps.") # Optional debug
                      else:
                           print(f"Warning: Timestamps for {relative_filepath} are not a numeric numpy array ({type(data['timestamps'])} {getattr(data['timestamps'], 'dtype', 'N/A')}). Skipping save.")

                 except Exception as e:
                      print(f"Error saving timestamps for {relative_filepath}: {e}")
                      # import traceback; traceback.print_exc()

            # Store sample rate as attribute if available (regardless of data presence, if extracted)
            if 'sample_rate' in data and data['sample_rate'] is not None:
                 try:
                      file_group.attrs['sample_rate'] = float(data['sample_rate']) # Ensure it's a float
                      # print(f"  Saved sample rate ({data['sample_rate']:.2f} Hz).") # Optional debug
                 except Exception as e:
                      print(f"Error saving sample rate attribute for {relative_filepath}: {e}")
                      # import traceback; traceback.print_exc()


    print(f"\nSuccessfully saved extracted data structure to {hdf5_output_path}")

except ImportError:
     print("\nError: h5py library not found. Cannot save to HDF5. Install with: pip install h5py")
except Exception as e:
    print(f"\nError saving data to HDF5: {e}")
    import traceback
    traceback.print_exc()