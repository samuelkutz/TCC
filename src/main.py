from BOUSSINESQ.run_dataset import run_dataset
from FNO.train_fno import train_fno
from FNO.plots_fno import eval_fno
from PINO.train_pino import train_pino_data, train_pino_no_data
from PINO.plots_pino import eval_pino
from PINN.train_pinn import train_pinn_data, train_pinn_no_data
from PINN.plots_pinn import eval_pinn


if __name__ == '__main__':
    print('\n=== generating shared dataset ===')
    run_dataset()

    print('\n=== training fno ===')
    train_fno()
    print('\n=== plotting fno ===')
    eval_fno()

    print('\n=== training pino with data ===')
    train_pino_data()
    print('\n=== plotting pino with data ===')
    eval_pino('data')

    print('\n=== training pino without data ===')
    train_pino_no_data()
    print('\n=== plotting pino without data ===')
    eval_pino('no_data')

    print('\n=== training pinn with data ===')
    train_pinn_data()
    print('\n=== plotting pinn with data ===')
    eval_pinn('data')

    print('\n=== training pinn without data ===')
    train_pinn_no_data()
    print('\n=== plotting pinn without data ===')
    eval_pinn('no_data')

    print('\nall experiments completed successfully.')
