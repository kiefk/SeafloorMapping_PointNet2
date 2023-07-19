'''
Created by Yiwen Lin
Date: Jul 2023
'''
import os, argparse
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
    log_dir = os.path.join('log/part_seg', log_dir)
    file_dir = os.path.join(log_dir, args.data_dir)
    output_dir = os.path.join(log_dir, args.output_dir)
    file_list_dir = args.file_list

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(file_list_dir, 'r') as f_obj:
        file_list = [file.rstrip('\n') for file in f_obj.readlines()]

    col = ['x', 'y', 'elev', 'label', 'prob']

    for file in file_list:
        for track in ['1l', '1r', '2l', '2r', '3l', '3r']:
            sub_file_list = []
            for sub_file in os.listdir(file_dir):
                if file in sub_file and track in sub_file:
                    sub_file = os.path.join(file_dir, sub_file)
                    df_sub_file = pd.read_csv(sub_file, sep=' ', names=col)
                    sub_file_list.extend(df_sub_file.to_numpy().tolist())
            df = pd.DataFrame(sub_file_list, columns=col)
            output_file = os.path.join(output_dir, file + '_' + track + '.txt')  # or output to csv file
            df.to_csv(output_file, sep=' ', index=False)




if __name__ == '__main__':
    args = parse_args()
    main(args)
