
import argparse
import subprocess

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='input data directory')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    mode='test'
    npoint=8192
    batch_size=1
    threshold=0.5
    num_votes=10

    # python preprocessing/preprocess_beam.py --data_dir ${args.data_dir} --removeLand --removeIrrelevant --utm
    cmd1 = ['python', 'preprocessing/preprocess_beam.py', '--data_dir', args.data_dir, '--removeLand', '--removeIrrelevant', '--utm']
    #Run the first preprocessing script and wait for it to finish before launching the next script
    subprocess.Popen(cmd1).wait()
    # python preprocessing/split_data_bulk.py --input_dir ${data_dir} --mode ${mode}
    cmd2 = ['python', 'preprocessing/split_data_bulk.py', '--input_dir', args.data_dir, '--mode', mode] 
    #Run the second preprocessing script and wait for it to finish before launching the prediction script
    subprocess.Popen(cmd2).wait()
    # python predict.py --batch_size ${batch_size} --num_point ${npoint} --data_root ${data_root} --conf --threshold ${threshold} --num_votes ${num_votes}
    cmd3 = ['python', 'predict.py', '--batch_size', str(batch_size), '--num_point', str(npoint), '--data_root', args.data_dir, '--conf', '--threshold', str(threshold), '--num_votes', str(num_votes)] 
    #Run the second preprocessing script and wait for it to finish before launching the prediction script
    subprocess.Popen(cmd3)
    

