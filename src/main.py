import os
import subprocess
import sys

# project root directory used to locate scripts and datasets
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

# task list pairs script labels with python module paths and args
TASKS = [
    ('fno', 'FNO.train_fno', []),
    ('pino data', 'PINO.train_pino', ['--mode', 'data']),
    ('pino no_data', 'PINO.train_pino', ['--mode', 'no_data']),
    ('pinn data', 'PINN.train_pinn', ['--mode', 'data']),
    ('pinn no_data', 'PINN.train_pinn', ['--mode', 'no_data']),
]


def run_script(script_path, args=None):
    # run one python experiment module in the project root
    command = [PYTHON, '-m', script_path]
    if args:
        command.extend(args)
    print('running:', ' '.join(command))
    env = os.environ.copy()
    env['PYTHONPATH'] = ROOT_DIR + os.pathsep + env.get('PYTHONPATH', '')
    subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)


if __name__ == '__main__':
    print('\n=== generating shared dataset ===')
    run_script('run_dataset')

    # execute each experiment training script then its matching plot script
    for label, train_script, train_args in TASKS:
        print(f'\n=== training {label} ===')
        run_script(train_script, train_args)

        plot_script = train_script.replace('train_', 'plots_')
        print(f'\n=== plotting {label} ===')
        run_script(plot_script, train_args)

    print('\nall experiments completed successfully.')
