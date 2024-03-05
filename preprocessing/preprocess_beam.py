# Anders Knudby, April 2021
# This code converts ICESat-2 .h5 files to csv files

# Imports
import glob
import pandas as pd
import os
import numpy as np
import argparse
import json


# Functions
def deleteFileIfExists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def findSurface(input, minElev, maxElev):
    df = input

    hist = np.histogram(df["elev"], int(maxElev - minElev))
    largest_bin = np.argmax(hist[0])

    # Get a subset of the histogram around the largest bin.
    # If possible get bin edges for four bins:
    #       The two bins before the largest bin, the largest bin, 
    #       and the bin after the largest bin.
    #This checks if there are two bins before the largest bin and 
    #   at least one bin after it as well. 
    if largest_bin >= 2 and largest_bin <= len(hist[0]) - 2:
        # numbers = hist[0][largest_bin - 2:largest_bin + 3]
        elevations = hist[1][largest_bin - 2:largest_bin + 3]

    #Else, if the largest bin is the last histogram bin, only get bin edges
    #   for the two bins before the largest bin, and the largest histogram bin
    elif largest_bin == len(hist[0]) - 1:
        # numbers = hist[0][largest_bin - 2:largest_bin + 3]
        elevations = hist[1][largest_bin - 2:largest_bin + 2]

    #Else, if the largest bin is the second histogram bin, only get bin edges
    #   for the bin before the largest bin, the largest histogram bin, 
    #   and the bin after the largest bin.        
    elif largest_bin == 1: 
        # numbers = hist[0][largest_bin - 1:largest_bin + 3]
        elevations = hist[1][largest_bin - 1:largest_bin + 3]

    #Else, if the largest bin is the first histogram bin, only get bin edges
    #   for the largest histogram bin, and the bin after the largest bin.   
    elif largest_bin == 0:  # This can happen when there is no land in the dataset, and no atmospheric noise
        # numbers = hist[0][largest_bin:largest_bin + 3]
        elevations = hist[1][largest_bin:largest_bin + 3]

    # OTherwise, catch any exceptions where there are not two bins before and
    #   one bin after the largest bin.
    else:
        print("Weird depth distribution of points, check visually")
        empty_df = pd.DataFrame()
        return empty_df

    df_subset = df[(df["elev"] > elevations[0]) & (df["elev"] < elevations[min(len(elevations) - 1, 4)])]

    mean = np.mean(df_subset["elev"])
    sd = np.std(df_subset["elev"])

    # set water surface as 5
    df.loc[(df['elev'] > mean - 2 * sd) & (df['elev'] < mean + 2 * sd), 'class'] = 5
    # only keep the points that are lower than average water surface level to decrease data volume
    df = df[df["elev"] < mean]

    # print("Done finding water surface")
    return df


# setting
parser = argparse.ArgumentParser(description='Convert ATL03 to CSV file')
parser.add_argument('--data_dir', type=str, required=True, help='Input directory')
parser.add_argument('--maxElev', type=int, default=10, help='Maximum elevation for filter')
parser.add_argument('--minElev', type=int, default=-50, help='Minimum elevation for filter')
parser.add_argument('--removeLand', action='store_true')
parser.add_argument('--removeIrrelevant', action='store_true')
parser.add_argument('--utm', action='store_true')
parser.add_argument('--interval', default=100000)


def convert(dataDir, utm=True, removeLand=True, removeIrrelevant=True, interval=100000, maxElev=10, minElev=-50):
    """
        Takes in a data directory that contains six beam csv files produced from a IceSat-2 granule.
        Runs part of the preprocessing on the beam data to prepare it for classification predictions.
        Write six csv files of preprocessed beam data, stored in a csv_data ouput directory.
        Preprocessing is continued by running split_data_bulk.py with the beam data in the csv_data directory.

        dataDir: String - directory path to the original beam data
        utm: Bool - Always true, remove from command line and remove non-true utm portions of the algorithm that are never accessed
        removeLand - Always true, remove from command line and remove non-true removeLand portions of the algorithm
        removeIrrelevant - Always true, remove from command line and remove non-true portions of the algorithm
        interval: Int - always 100,000. Remove from command line, make a variable at top of function or close to where it's called near findSurface
        maxElev: Int - always 10. Remove from command line, make a variable.
        minElev: Int - always -50. Remove from command line, make a variable. 

        TODO: Use CoastNet surface classification instead of findSurface function
        TODO: Are interval, maxElev, minElev irrelevant because of CoastNet sea surface classification?
        TODO: Is UTM irrelevant due to easting and northing being provided in original beam data. 
    """
    # Get all files in this directory that end in _input.csv
    #   This will be the six beam files for the granule.
    #   IceSAT-2 input filename pattern = gtxx_granule_name_input.csv 
    beams = glob.glob(dataDir + "/*_input.csv")

    # Create a directory path for "csv_data" to store the output files of preprocess_beam
    output_dir = os.path.join(dataDir, 'csv_data')
    
    #If the csv_data folder doesn't exist, create it. 
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #Get the JSON file that holds the per granule variables. 
    json_name = glob.glob(dataDir + "/*.json")

    #Create a dictionary of the per granule variables from the JSON file
    granule_vars = json.load(json_name[0])

    #Loop through the six beams
    for filename in beams:

        print(f'Preprocessing {filename}...')

        # Read in the csv file with pandas and store it in a beam dataframe
        beam_df = pd.read_csv(filename)

        # Add a class column
        beam_df['class'] = np.full((len(df)), 3)

        # Remove sea surface photons, this section replaces the  findSurface()

        # Get the sea surface filename for this beam
        # IceSAT-2 input filename pattern = gtxx_granule_name_input.csv
        # Sea Surface filename pattern = gtxx_granule_name_sea_surface.csv
        sea_surface_filename = filename.replace('input', 'sea_surface')
        # Use pandas to read in a dataframe for the sea surface csv with the same beam name
        sea_surface_df = pd.read_csv(sea_surface_filename)

        # Assuming beam_df and sea_surface_df have the same number of rows/photons (they should)
        # If sea_surface_df.class_ph == 41 (sea surface), then set the class for beam_df at that same photon index to 5 (sea surface for PointNet)
        beam_df.loc[sea_surface_df.class_ph == 41, 'class'] = 5
        
        # Get the subset of the sea_surface_df where class_ph == 41 (sea surface)
        df_subset = sea_surface_df[sea_surface_df.class_ph == 41]
        # Take the average of the geoid corrected height of the sea surface photons.
        mean = np.mean(df_subset['surface_ph'])
        # Only keep the photons that are lower than average water surface level to decrease data volume.
        beam_df = beam_df[beam_df['geoid_corrected_h'] < mean]

        #End of findSurface() replacement

        # Keep photons where the max_signal_conf is greater than or equal to 3.
        # Keep photons where lat_ph is less than 9000.
        beam_df = beam_df[(beam_df['max_signal_conf'] >= 3)
                            & (beam_df['lat_ph'] < 9000)]

        # Remove data outside reasonable boundaries
        # Keep photons where geoid_corrected_h is greater than -12,000
        beam_df = beam_df[beam_df['geoid_corrected_h'] > -12000]
        
        # TODO: Always true, so we should remove "removeIrrelevant" from the command line and if check, and just evalute the statement every time. 
        # Remove irrelevant photons (deeper than 50m, higher than 10m)
        if removeIrrelevant:
            beam_df = beam_df[(beam_df["geoid_corrected_h"] > minElev) & (beam_df["geoid_corrected_h"] < maxElev)]
            
        # Remove photons where the NDWI value is less than 0.3
        # TODO: Always true, so we should remove "removeLand" from the command line and if check, and just evalute the statement every time. 
        if removeLand:  
            beam_df = beam_df[(beam_df["ndwi"] >= 0.3)]

        # TESTING CODE: Log the along track distance of each beam
        # log_along_track_path = dataDir + "/along_track_distance.txt"
        # with open(log_along_track_path, "a+") as output_file:
        #     output_file.write(f"{filename} Along-track distance: {beam_df['x_atc'].max() - beam_df['x_atc'].min()}\n")

        # Drop unused columns
        beam_df = beam_df.drop(['x_atc', 'y_atc', 'sigma_along', 'sigma_across', 'sigma_h', 'delta_time', 'yapc', 'quality_ph', 'ndwi'], axis=1)



        # Rename columns to match expected names in split_data_bulk.py and generate_training_data.py
        # TODO: Change downstream names to match original column names instead of renaming. Check split_data_bulk.py and generate_training_data.py and prediction files.
        # TODO: Potentailly drop the lat and lon columns? I don't think they're used later. Check split_data_bulk.py and generate_training_data.py and prediction files.
        beam_df.rename(columns={'index_ph': 'ph_index', 'geoid_corrected_h': 'elev', 'x_ph': 'x', 'y_ph': 'y', 'lon_ph': 'lon', 'lat_ph': 'lat', 'max_signal_conf': 'signal_conf_ph'})
        # Change the order of the columns
        beam_df = beam_df[['ph_index', 'x', 'y', 'lon', 'lat', 'elev', 'signal_conf_ph', 'class']]  

        # Do normalization so all values are between 0 and 1
        # df = (df - df.min()) / (df.max() - df.min())

        # Round to three decimals for all variables
        # TODO: Test this, this line was originally uncommented but a recent code update by the original author has it now deactivated.
        # df = df.round(decimals=3)

        # Start of old findSurface() code: 
        # TODO: Keep until we can test how CoastNet's surface identification effects the results of PointNet. 

        # # Empty dataframe with the same column names as the beam_df.
        # # Holds the segments of data as they're sent through the findSurface() function. 
        # # TODO: findSurface() is obsolete, this is also no longer needed?
        # df_segment_all = pd.DataFrame(columns=df.columns)
        
        # #Original method
        # # num = math.ceil((df['y'].max() - df['y'].min()) / interval)
        # # y1 = df['y'].min()

        # #Segmenting by ph_index
        # num_rows = df.index
        # num = math.ceil(num_rows.max() / interval)
        # y1 = 0


        # # print(f"num: {num}")
        # # print(f"y1: {y1}")

        # for i in range(num):
        #     y2 = y1 + interval

        #     #Segmenting by ph_index
        #     # If this is the last segment, make sure to copy the last photon
        #     if i == num-1: 
        #         df_segment = df.iloc[y1:].copy()
        #     else:
        #         df_segment = df.iloc[y1:y2].copy()

        #     #Original method
        #     # df_segment = df[(df['y'] >= y1) & (df['y'] < y2)].copy()

        #     if not df_segment.empty:
        #         df_segment = findSurface(df_segment, minElev, maxElev)
        #         # If there is a weird distribution of points, skip this segment.
        #         if df_segment.empty:
        #             continue

        #         df_segment_all = pd.concat([df_segment_all, df_segment], ignore_index=True)
        #     y1 = y2

        # df = df_segment_all

        # End of old findSurface() code

        # Add "N" or "S" to utm_zone, this will be added to the name of the intermediate output files.
        # TODO: Remove this by changing how split_data_bulk reads in the intermediate files, or move the
        #   functionality of split_data_bulk.py into preprocess_beam.py.  
        if beam_df["lat"].mean() > 0:
            zone = granule_vars["utm_zone"] + "N"
        else:
            zone = granule_vars["utm_zone"] + "S"

        # Write data to csv file
        output_filename = output_dir + "/" + os.path.splitext(os.path.basename(filename))[0] + "_raw_" + zone + ".csv" 
        
        beam_df.to_csv(output_filename, index=False)

        # Move the original files to new folder
        # new_filename = os.path.join(dir, os.path.basename(filename))
        # shutil.move(filename, new_filename)

        # print("H5 to CSV done!")

def main(args):
    dataDir = args.data_dir
    utm = args.utm
    removeLand = args.removeLand
    removeIrrelevant = args.removeIrrelevant
    interval = args.interval
    maxElev = args.maxElev
    minElev = args.minElev
    convert(dataDir, utm=utm, removeLand=removeLand, removeIrrelevant=removeIrrelevant, interval=interval, maxElev=maxElev, minElev=minElev)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
