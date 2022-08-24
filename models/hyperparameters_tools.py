import os
import pickle


def load_pickle_file(optimal_parameters_file_path):
    with open(optimal_parameters_file_path, "rb") as fp:
        optimal_parameters = pickle.load(fp)
    return optimal_parameters

def load_optimal_hyperparameters(optimal_parameters_file_path):
    with open(optimal_parameters_file_path, "rb") as fp:
        optimal_parameters = pickle.load(fp)
    for parameter_name, parameter_value in optimal_parameters.items():
        optimal_parameters[parameter_name] = parameter_value[0]
    return optimal_parameters


def export_optimal_hyperparameters(parameter_values, parameter_names, exp_save_path):
    result = {}
    for i, parameter_name in enumerate(parameter_names):
        if parameter_name not in result:
            result[parameter_name] = []
            result[parameter_name].append(parameter_values[i])
        else:
            print(f"Parameter {parameter_name} already saved, continuing ...")
    with open(os.path.join(exp_save_path, "optimal_parameters.pickle"), "wb") as fp:
        pickle.dump(result, fp)
    print("Optimal parameters saved.")
