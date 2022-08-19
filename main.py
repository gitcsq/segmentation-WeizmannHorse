import argparse
import train
import test


def get_args_parser():
    parser = argparse.ArgumentParser('UNet test', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1.0e-2, type=float)
    parser.add_argument('--output_dir', default='./checkpoint', type=str)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--save_freq', default=30, type=int)
    parser.add_argument('--checkpoint_path', default='./checkpoint/depth5-ep119.pth', type=str)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--mode', default='train', type=str)

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    assert args.mode in ['train', 'test']
    if args.mode == 'train':
        train.main(args)
    else:
        test.main(args)
