import os
import subprocess
import sys

# project root directory used to locate scripts and datasets
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# task list pairs script labels with executable paths and args
TASKS = [
    ('fno', 'FNO/train_fno.py', []),
    ('pino data', 'PINO/train_pino.py', ['--mode', 'data']),
    ('pino no_data', 'PINO/train_pino.py', ['--mode', 'no_data']),
    ('pinn data', 'PINN/train_pinn.py', ['--mode', 'data']),
    ('pinn no_data', 'PINN/train_pinn.py', ['--mode', 'no_data']),
]


def run_script(script_path, args=None):
    # run one python experiment script in the project root
    command = [PYTHON, os.path.join(ROOT_DIR, script_path)]
    if args:
        command.extend(args)
    print('running:', ' '.join(command))
    subprocess.run(command, cwd=ROOT_DIR, check=True)


if __name__ == '__main__':
    # execute each experiment training script then its matching plot script
    for label, train_script, train_args in TASKS:
        print(f'\n=== training {label} ===')
        run_script(train_script, train_args)

        plot_script = train_script.replace('train_', 'plots_')
        print(f'\n=== plotting {label} ===')
        run_script(plot_script, train_args)

    print('\nall experiments completed successfully.')
