from bayes_opt import BayesianOptimization
from .simulation import run, get_data_points
from .test_data import data_dot_daily as data

formatted_data = get_data_points(data)


def objective_function(FAST, SLOW, RSI_SETTING):
    return run(formatted_data, int(FAST), int(SLOW), int(RSI_SETTING))


if __name__ == '__main__':

    pbounds = {'FAST': (2, 100), 'SLOW': (2, 100), 'RSI_SETTING': (3, 20)}

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(
        init_points=5,  # Initial random points
        n_iter=10,       # Number of optimization steps
    )

    best_params = optimizer.max['params']
    best_value = optimizer.max['target']

    print("Best Hyperparameters:", best_params)
    print("Best Objective Value:", best_value)
