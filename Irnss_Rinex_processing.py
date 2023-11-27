import pandas as pd
import plotly.graph_objects as go
from tkinter import filedialog, Tk
from datetime import datetime
import math

class RinexProcessor:

       
    def __init__(self):
        self.file_paths_dataset_01 = self.select_files("Select Files for Data Set 01")
        self.file_paths_dataset_02 = self.select_files("Select Files for Data Set 02")
        self.dataset_01 = [self.read_file(path) for path in self.file_paths_dataset_01]
        self.dataset_02 = [self.read_file(path) for path in self.file_paths_dataset_02]
    
    def select_files(self, title):
        root = Tk()
        root.withdraw()
        file_paths = filedialog.askopenfilenames(title=title)
        return file_paths
        
    def read_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Locate the end of the header section
        header_end_index = None
        for i, line in enumerate(lines):
            if "END OF HEADER" in line:
                header_end_index = i
                break

        if header_end_index is None:
            raise ValueError("Header end marker not found in file.")

        # Extract header information for satellite data
        sys_and_obs_types = self.extract_satellite_data(lines[:header_end_index + 1])
        if not sys_and_obs_types:
            raise ValueError("File does not contain observation systems and types.")

        # Process only the lines after the header
        data_lines = lines[header_end_index + 1:]

        # Process data into DataFrame
        data = self.process_data(data_lines, sys_and_obs_types)
        return data
    
    def extract_satellite_data(self, lines):
        satellite_data = {}
        for line in lines:
            if line.strip().endswith("SYS / # / OBS TYPES"):
                parts = line.split()
                system = parts[0]
                if system in ['G', 'R', 'S', 'E', 'I', 'C', 'Q']:
                    try:
                        num_obs = int(parts[1])  # Extracting number of observations
                    except ValueError:
                        # Handle case where the number of observations is not correctly extracted
                        continue  # or raise an error
                    # Filter out only valid observation types
                    obs_types = [part for part in parts[2:] if part not in ['SYS', '/', '#', 'OBS', 'TYPES']]
                    satellite_data[system] = {'num_obs': num_obs, 'obs_types': obs_types}
                    print(f"Observation types in NavIC data : \n {obs_types}")
                    # satellite_data[system] = {'num_obs': num_obs, 'obs_types': obs_types}
        return satellite_data
    
    def process_data(self, lines, sys_and_obs_types):
        data_entries = []
        current_epoch = None
        current_rx_clk_offset = None

            # Combine all observation types from different systems
        all_obs_types = []
        for system in sys_and_obs_types.values():
            all_obs_types.extend(system['obs_types'])
        all_obs_types = sorted(set(all_obs_types))  # Remove duplicates and sort

        for line in lines:
            if line.startswith('>'):
                current_epoch, current_rx_clk_offset = self.extract_epoch_and_clk_offset(line)
                continue

            satellite_system = line[0]
            if satellite_system in sys_and_obs_types:
                measurements = self.extract_measurements(line, sys_and_obs_types[satellite_system])
                data_entry = [current_epoch] + measurements + [current_rx_clk_offset]
                data_entries.append(data_entry)

        # Assuming the longest line has maximum number of columns
        if data_entries:
            # max_cols = max(len(entry) for entry in data_entries)
            # columns = ['Epoch', 'SAT_PRN'] + [f'Obs_{i}' for i in range(1, max_cols - 2)] + ['Rx_clk_offset']
            columns = ['Epoch', 'SAT_PRN'] + all_obs_types + ['Rx_clk_offset']
            return pd.DataFrame(data_entries, columns=columns)
        else:
            return pd.DataFrame()  # Return an empty DataFrame if there are no data entries

    # def process_data(self, lines, sys_and_obs_types):
    #     # Initialize data structure to hold DataFrames for each system
    #     # data_frames = {system: [] for system in sys_and_obs_types.keys()}
    #     data_entries = []
    #     current_epoch = None
    #     current_rx_clk_offset = None
    #     for line in lines:
    #         if line.startswith('>'):
    #             # Extract epoch information and Rx_clk_offset
    #             current_epoch, current_rx_clk_offset = self.extract_epoch_and_clk_offset(line)
    #             continue

    #         # Extract satellite system from the line
    #         satellite_system = line[0]
    #         if satellite_system in sys_and_obs_types:
    #             # Extract measurements and append to the corresponding list
    #             measurements = self.extract_measurements(line, sys_and_obs_types[satellite_system])
    #             data_entry = [current_epoch] + measurements + [current_rx_clk_offset]
    #             data_entries.append(data_entry)

    #     # Convert lists to pandas DataFrames with dynamic column names
    #     for system, data in data_entries.items():
    #         if len(data) > 0:
    #             # Determine the number of columns dynamically from the data
    #             max_cols = max(len(row) for row in data)
    #             obs_types = sys_and_obs_types[system]['obs_types']
    #             expected_cols = ['Epoch', 'SAT_PRN'] + obs_types + ['Rx_clk_offset']
    #             if max_cols != len(expected_cols):
    #                 # Adjust the number of observation columns if necessary
    #                 additional_obs_cols = [f'Obs_{i}' for i in range(len(obs_types), max_cols - 3)]
    #                 columns = ['Epoch', 'SAT_PRN'] + obs_types + additional_obs_cols + ['Rx_clk_offset']
    #             else:
    #                 columns = expected_cols
    #             # print(f"Warning: Mismatch in number of columns for system {system}. Data is skipped.")
    #             data_entries[system] = pd.DataFrame(data, columns=columns)

    #     return data_entries

    def extract_epoch_and_clk_offset(self, line):
        # Extract epoch information and Rx_clk_offset from the line
        # Format: > yyyy mm dd hh mm ss.sssssss  status_flag  num_obs  Rx_clk_offset
        parts = line.split()
        year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
        hours, minutes, seconds = int(parts[4]), int(parts[5]), float(parts[6])
        
        # Convert to Modified Julian Date
        mjd = self.date_to_mjd(year, month, day)

        # Calculate the fraction of the day
        day_fraction = (hours * 3600 + minutes * 60 + seconds) / 86400

        # Combine MJD and day fraction
        epoch = mjd + day_fraction
        Rx_clk_offset = parts[-1]
        return epoch, Rx_clk_offset

    def date_to_mjd(self, year, month, day):
        # Convert a calendar date to Modified Julian Date
        if month <= 2:
            year -= 1
            month += 12
        a = math.floor(year / 100)
        b = 2 - a + math.floor(a / 4)
        mjd = math.floor(365.25 * (year + 4716)) + math.floor(30.6001 * (month + 1)) + day + b - 2401525.5
        return mjd

    def extract_measurements(self, line, sys_obs_info):
        elements = line.split()
        sat_prn = elements[0]  # Satellite PRN
        obs_types = sys_obs_info['obs_types']
        measurements = []

        # Index to track position in elements
        elem_index = 1

        for obs_type in obs_types:
            if elem_index < len(elements):
                # Add observation value
                measurements.append(elements[elem_index])
                elem_index += 1

                # Skip signal strength indicators after pseudo range and Doppler measurements
                if obs_type.startswith('C') or obs_type.startswith('D'):
                    if elem_index < len(elements):  # Check to avoid index out of range
                        elem_index += 1  # Skip the signal strength indicator

        return [sat_prn] + measurements
        
    # def calculate_differences(self):
    #     if len(self.dataset_01) < 1 or len(self.dataset_02) < 1:
    #         raise ValueError("Two files are required for comparison/ No data for processing.")

    #     diff_dataframes = {}
    #     for system in self.dataset_01.keys():
    #         if system in self.dataset_02:
    #             # Merging datasets based on 'Epoch' and 'SAT_PRN'
    #             merged_df = pd.merge(self.dataset_01[system], self.dataset_02[system], on=['Epoch', 'SAT_PRN'], suffixes=('_1', '_2'))
                
    #             # Initialize a DataFrame to store the differences
    #             diff_df = pd.DataFrame()
    #             diff_df['Epoch'] = merged_df['Epoch']
    #             diff_df['SAT_PRN'] = merged_df['SAT_PRN']

    #             # Calculating differences for observation types
    #             for obs_type in self.dataset_01[system].columns[2:-1]:  # Excluding 'Epoch', 'SAT_PRN', 'Rx_clk_offset'
    #                 if obs_type in self.dataset_02[system].columns:
    #                     diff_df[obs_type + '_diff'] = merged_df[obs_type + '_1'] - merged_df[obs_type + '_2']

    #             # Calculating difference for Rx_clk_offset
    #             diff_df['Rx_clk_offset_diff'] = merged_df['Rx_clk_offset_1'] - merged_df['Rx_clk_offset_2']

    #             diff_dataframes[system] = diff_df
    #     print(f"Data frames: \n{diff_dataframes}")


    def calculate_differences(self):
        # print(f" Data set 01 : \n{self.dataset_01}")
        # print(f" Data set 02 : \n{self.dataset_02}")

                # Write dataset_01 to a CSV file
        for i, df in enumerate(self.dataset_01):
            df.to_csv(f'dataset_01_{i}.csv', sep='\t',index=False)
            print(f"Dataset 01 - File {i} saved as 'dataset_01_{i}.csv'")

        # Write dataset_02 to a CSV file
        for i, df in enumerate(self.dataset_02):
            df.to_csv(f'dataset_02_{i}.csv', sep='\t',index=False)
            print(f"Dataset 02 - File {i} saved as 'dataset_02_{i}.csv'")

        if len(self.dataset_01) < 1 or len(self.dataset_02) < 1:
            raise ValueError("Two files are required for comparison/ No data for processing.")

        diff_dataframes = []
        for df1, df2 in zip(self.dataset_01, self.dataset_02):

            # Check if df1 and df2 are DataFrames
            if not isinstance(df1, pd.DataFrame) or not isinstance(df2, pd.DataFrame):
                continue  # Skip if either is not a DataFrame

            # Merging datasets based on 'Epoch' and 'SAT_PRN'
            merged_df = pd.merge(df1, df2, on=['Epoch', 'SAT_PRN'], suffixes=('_1', '_2'))
            
            # Initialize a DataFrame to store the differences
            diff_df = pd.DataFrame()
            diff_df['Epoch'] = merged_df['Epoch']
            diff_df['SAT_PRN'] = merged_df['SAT_PRN']

            # Calculate differences for each observation type
            for col in df1.columns:
                if col not in ['Epoch', 'SAT_PRN', 'Rx_clk_offset']:
                    # Convert columns to numeric type before subtracting
                    merged_df[col + '_1'] = pd.to_numeric(merged_df[col + '_1'], errors='coerce')
                    merged_df[col + '_2'] = pd.to_numeric(merged_df[col + '_2'], errors='coerce')

                    # Calculate differences
                    diff_df[col + '_diff'] = merged_df[col + '_1'] - merged_df[col + '_2']

            # Calculate difference for Rx_clk_offset
            diff_df['Rx_clk_offset_diff'] = pd.to_numeric(merged_df['Rx_clk_offset_1'], errors='coerce') - pd.to_numeric(merged_df['Rx_clk_offset_2'], errors='coerce')
            diff_dataframes.append(diff_df)

        print(f" Difference Data frames: \n{diff_dataframes}")

        return diff_dataframes


    def plot_differences(self, differences):
        # Define the list of specific signals to plot
        specific_signals = ['C5C_diff', 'L5C_diff', 'C9C_diff', 'L9C_diff']
        geo_satellites = ['I03', 'I06', 'I10', 'I07']

        for signal in specific_signals:
            fig = go.Figure()
            for diff_df in differences:
                # Filter the DataFrame for only the specified GEO STATIONARY SATELLITES
                geo_df = diff_df[diff_df['SAT_PRN'].isin(geo_satellites)]

                if signal in geo_df.columns:
                    for sat_prn in geo_satellites:
                        # Filter for a specific satellite
                        sat_df = geo_df[geo_df['SAT_PRN'] == sat_prn]
                        fig.add_trace(go.Scatter(x=sat_df['Epoch'], y=sat_df[signal], mode='lines', name=f'{sat_prn} - {signal}'))

            fig.update_layout(title=f"Differences in {signal} for GEO STATIONARY SATELLITES", xaxis_title='Epoch', yaxis_title='Difference')
            fig.show()

                    
        
def main():
    processor = RinexProcessor()
    differences = processor.calculate_differences()
    processor.plot_differences(differences)

if __name__ == "__main__":
    main()
