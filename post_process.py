'''
Created by Yiwen Lin
Date: Jul 2023
'''
import os, argparse
import re
import pandas as pd


def refraction_correction_approx(b_z, w_z):
    b_z = b_z + 0.25416 * (w_z - b_z)
    return b_z


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='experiment root')
    parser.add_argument('--data_dir', type=str, help='input data directory')
    parser.add_argument('--file_list', type=str, default='file_list.txt', help='a list of original files in txt format')
    parser.add_argument('--output_dir', type=str, help='output directory')

    return parser.parse_args()


def main(args):
    log_dir = args.log_dir
    input_dir = os.path.join(log_dir, args.data_dir)
    output_dir = os.path.join(log_dir, args.output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    file_list = []
    pattern = r'^(.*[NS])'
    for sub_file in os.listdir(input_dir):
        match = re.search(pattern, sub_file)
        if match:
            file_list.append(match.group(0))

    file_list = set(file_list)

    columns = pd.read_csv(os.path.join(input_dir, os.listdir(input_dir)[0])).columns

    for file in file_list:
        sub_file_list = []
        for sub_file in os.listdir(input_dir):
            if file in sub_file:
                df_sub_file = pd.read_csv(os.path.join(input_dir, sub_file), sep=',')
                sub_file_list.extend(df_sub_file.to_numpy().tolist())
        df = pd.DataFrame(sub_file_list, columns=columns)
        # convert label column to integer
        # If pred = 1 the photon is bathymetry, if pred = 0 the photon is other. 
        if 'pred' in df.columns:
            df['pred'] = df['pred'].astype(int)
        if 'label' in df.columns:
            df['label'] = df['label'].astype(int)

        #Change bathymetry classification value from 1 to 40 to match ASPRS classifications.
        df.loc[df['pred'] == 1, 'pred'] = 40
        #Change other classification value from 0 to 1 to match ASPRS classifications.
        df.loc[df['pred'] == 0, 'pred'] = 1

        

        output_file = os.path.join(output_dir, file + '.csv')
        df.to_csv(output_file, sep=',', index=False, header=True)


if __name__ == '__main__':
    args = parse_args()
    # args.log_dir = './log/2023-07-26_19-32-32'
    # args.data_dir = 'output_0.5'
    # args.output_dir = 'output_0.5_merge'
    main(args)
