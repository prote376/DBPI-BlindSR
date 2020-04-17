import os
import tqdm

from configs import Config
from data import DataGenerator
from DBPISR import DBPISR
from learner import Learner

def train(conf):
    sr_net = DBPISR(conf)
    learner = Learner()
    data = DataGenerator(conf, sr_net)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        g_in = data.__getitem__(iteration)
        sr_net.train(g_in)
        learner.update(iteration, sr_net)

    sr_net.finish(data.input_image)


def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images/', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='Results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    for filename in os.listdir(os.path.abspath(args.input_dir)):
        conf = Config().parse(create_params(filename, args))
        train(conf)
    prog.exit(0)


def create_params(filename, args):
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir)]
    if args.X4:
        params.append('--X4')
    return params


if __name__ == '__main__':
    main()
