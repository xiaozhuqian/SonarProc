import matlab.engine
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='jsf2mat')
    parser.add_argument('--input_dir', default='../../example_data/', help='directory of input file')
    parser.add_argument('--out_dir', default='../../outputs/mat/', help='dierctory to save backscatter and navigation data')

    args = parser.parse_args()
    return args


def read_jsf(jsf_dir, mat_dir):
    eng = matlab.engine.start_matlab()
    eng.cd(r'./prep/jsf_reading/', nargout=0)
    eng.jsf2mat(jsf_dir,mat_dir, nargout=0)
    eng.quit()

if __name__ == '__main__':
    args = parse_args()
    read_jsf(args.input_dir, args.out_dir)