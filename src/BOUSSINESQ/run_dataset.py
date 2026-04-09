import argparse
import os
import numpy as np

from BOUSSINESQ.dataset import generate_dataset, save_dataset


def main():
    parser = argparse.ArgumentParser(description='Generate the shared Boussinesq dataset.')
    parser.add_argument(
        '--dataset-file',
        default=os.path.join('RESULTS', 'boussinesq_dataset.pth'),
        help='Path where the generated dataset will be saved.',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device used for dataset generation.',
    )
    parser.add_argument(
        '--param-values',
        nargs='+',
        type=float,
        default=list(np.arange(0.1, 5.01, 0.5)),
        help='List of parameter values for dataset generation.',
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.dataset_file) or '.', exist_ok=True)
    x_train, y_train = generate_dataset(args.param_values, device=args.device)
    save_dataset(x_train, y_train, args.dataset_file)
    print(f'dataset written to {args.dataset_file}')


if __name__ == '__main__':
    main()
