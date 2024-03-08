
import argparse
import os

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='input data directory')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    mode='test'
    npoint=8192
    batch_size=1
    threshold=0.5
    num_votes=10

    # python preprocessing/preprocess_beam.py --data_dir ${args.data_dir} --removeLand --removeIrrelevant --utm
    cmd1 = 'python preprocessing/preprocess_beam.py --data_dir ' + args.data_dir + ' --removeLand --removeIrrelevant --utm'
    #Run the first preprocessing script and wait for it to finish before launching the next script
    return_code_1 = os.system(cmd1)
    if return_code_1 != 0:
        print("Run preprocess_beam script error")
    # python preprocessing/split_data_bulk.py --input_dir ${data_dir} --mode ${mode}
    cmd2 = 'python preprocessing/split_data_bulk.py --input_dir ' + args.data_dir + ' --mode ' + mode
    #Run the second preprocessing script and wait for it to finish before launching the prediction script
    return_code = os.system(cmd2)
    if return_code != 0:
        print("Run split_data_bulk script error")
    # python predict.py --batch_size ${batch_size} --num_point ${npoint} --data_root ${data_root} --conf --threshold ${threshold} --num_votes ${num_votes}
    cmd3 = 'python predict.py --batch_size ' + str(batch_size) + ' --num_point ' + str(npoint) + ' --data_root ' + args.data_dir + ' --conf --threshold ' + str(threshold) + ' --num_votes ' + str(num_votes)
    #Run the second preprocessing script and wait for it to finish before launching the prediction script
    return_code_3 = os.system(cmd3)
    if return_code_3 != 0:
        print("Run prediction script error")
    

