#!/usr/bin/env python3
import warnings
import report
import json
import numpy as np
import argparse
import importlib
import ast
import lib

warnings.filterwarnings('ignore')

def get_user_selected_vars(all_variables):
    """Repeatedly asks user for variables which exist as headers in their data to be the model input variables."""
    while True:
        print("The variables identified from your input data are: ", end="")
        for var in all_variables:
            print(f"[blue bold]{var} ", end="")
        print("")
        selected_vars = input(
            "Which variables are model inputs ? (please input a comma separated list): "
        )
        selected_vars = selected_vars.split(",")
        for var in selected_vars:
            if len(selected_vars) == 0:
                print("You must select an input variable.")
            if var not in all_variables:
                print(
                    f"'{var}' is not a variable in your dataset. Please only input variables which are in your dataset."
                )
                break
        else:
            return selected_vars


def get_user_selected_output(all_variables):
    """Repeatedly asks user for a variable which exists as a header in their data to be the model output."""
    while True:
        print("The variables identified from your input data are: ", end="")
        for var in all_variables:
            print(f"[blue bold]{var} ", end="")
        print("")
        selected_var = input("Which variable is the model output ?: ")
        # Repeatedly ask the user to select a variable if the variable they supplied is not in their data.
        for var in selected_var:
            if len(selected_var) != 1:
                print("You must select one variable as a model output.")
            if var not in all_variables:
                print(
                    f"'{var}' is not a variable in your dataset. Please only output variables which are in your dataset."
                )
                break
        else:
            return selected_var


def get_args():
    parser = argparse.ArgumentParser(
        prog="Modelling Utils",
        description="Basic modelling utilities for multivariate single output functions.",
    )
    parser.add_argument("datafile")
    parser.add_argument(
        "std_dev",
        help="float, measured or estimated standard deviation of output variable. Constant standard deviation is assumed.",
    )
    parser.add_argument("model_file", help="predefined model file")
    parser.add_argument("--input_var")
    parser.add_argument("--output_var")
    return parser.parse_args()


class VariableUsageFinder(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.param_occurances = {}

    def visit_Subscript(self, node):
        # when the user createds an indexed variable i.e. the indexed "x"s, this created a "Subscript" node in the AST, search through the AST to find all of these nodes to find the user is supplying more data than the user is using in their model
        if node.value.id != "x":
            return
        # if havent started counting which indexes are used yet, then start
        if node.value.id not in self.param_occurances:
            self.param_occurances[node.value.id] = []
            self.param_occurances[node.value.id].append(node.slice.value)
        else:
            if node.slice.value not in self.param_occurances[node.value.id]:
                self.param_occurances[node.value.id].append(node.slice.value)


def user_selected_correct_amount_data(model_file, selected_variables):
    """Parses the AST of the model_file to find the occurances of an x[i] and then checks if there is more selected variables than are accessed by the model."""
    source = ast.parse(open(model_file, "r", encoding=None).read())
    finder = VariableUsageFinder()
    finder.visit(source)
    if len(selected_variables) != len(finder.param_occurances["x"]):
        print(
            f"You have selected data for {len(selected_variables)} variables, however the provided function only accesses x{[x for x in finder.param_occurances['x']]}"
        )
        return False
    return True

if __name__ == "__main__":
    args = get_args()
    user_mod = importlib.import_module(args.model_file)
    raw_data = lib.RawDataset().from_csv(args.datafile)

    if args.input_var:
        user_selected_variables = args.input_var.split(",")
    else:
        user_selected_variables = get_user_selected_vars(list(raw_data.data.keys()))
    if args.output_var:
        user_selected_output = args.output_var
    else:
        user_selected_output = get_user_selected_output(list(raw_data.data.keys()))
    clean_data = lib.CleanDataset().from_rawdataset(
        raw_data, user_selected_variables, user_selected_output
    )
    # Assumes constant variance
    std_dev = float(args.std_dev)
    std_dev_arr = np.ones_like(clean_data.y) * std_dev
    if not user_selected_correct_amount_data(
        str(args.model_file) + ".py", user_selected_variables
    ):
        exit(1)
    user_fn = user_mod.user_fn
    model = lib.FittedModel().from_fn(user_mod.user_fn, clean_data, std_dev_arr)
    printed_results = {}
    printed_results["optimal_parameters"] = model.popt
    printed_results["condition_number"] = model.condition_number()
    printed_results["correlation_matrix"] = model.correlation_matrix().tolist()
    printed_results["mean_sqr_err"] = model.MSE()
    printed_results["sobol_idxs"] = model.sobol_idx(1024, 2)
    print(json.dumps(printed_results, indent=4))
    report.render_report(model, clean_data)
    exit(0)
