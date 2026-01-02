from collections.abc import Callable
from scipy.optimize import curve_fit
from typing import Dict, List
import inspect
import numpy as np
import csv
from SALib.sample.sobol import sample
from SALib.analyze import sobol
class RawDataset:
    data: Dict[str, List[float]]

    def from_csv(self, filename):
        """Reads user data from a csv file and returns the data in a dictonary.
        The keys of the dictionary are the column headers in the CSV."""
        self.data = {}
        with open(filename, "r") as f:
            reader = csv.reader(f)
            col_titles = next(reader)
            for title in col_titles:
                self.data[title] = []
            for row in reader:
                iterator = iter(row)
                for title in col_titles:
                    self.data[title].append(float(next(iterator)))
        return self


class CleanDataset:
    x: np.ndarray
    y: np.ndarray
    mapping: Dict[int,str]
    output_var: str

    def from_rawdataset(self, raw: RawDataset, input_vars: List[str], output_var: str):
        """Converts data dictionary into an (k,M) (row, col) array,
        where k is number of variables in the model and M is number of datapoints.
        """
        # TODO: add ability for user to input a mapping of variables
        x = []
        i = 0
        input_vars.sort()
        self.mapping = {}
        for var in input_vars:
            print(f"Mapped {var} to x[{i}]")
            self.mapping[i]=var
            print("")
            x.append(raw.data[var])
            i += 1
        self.x = np.array(x)
        self.y = np.array(raw.data[output_var])
        self.output_var = output_var
        return self

    # def from_cleandata(self,data):
    # TODO: implement the ability to construct a CleanDataset from clean data


class FittedModel:
    _input_matrix: np.ndarray
    _residuals: np.ndarray
    _num_params: int
    pcov: np.ndarray
    popt: Dict[str, float]
    _function: Callable

    def from_fn(self, function: Callable, dataset: CleanDataset, std_dev: np.ndarray):
        """
        Fits the parameters of a model defined by function to fit dataset

        All arguments to function other than the first argument are assumed to be model parameters.

        Model outputs in the dataset vary with std_dev.

        Arguments:
        function -- A callable python function to be fit via SciPy curve_fit
        dataset -- A CleanDataset representing the training data for the model
        std_dev -- An array representing the standard deviation of each variable in the training data
        """
        self._function = function
        self._input_matrix = dataset.x
        self._num_params = dataset.x.shape[0]
        popt, self.pcov, info_dict, _, _ = curve_fit(
            function,
            dataset.x,
            dataset.y,
            sigma=std_dev,
            absolute_sigma=True,
            full_output=True,
        )
        argspec = inspect.getfullargspec(function)
        parameters = argspec.args[1:]
        self.popt = {}
        for i, parameter in enumerate(parameters):
            self.popt[parameter] = popt[i]
        self._residuals = info_dict["fvec"]
        return self

    def MSE(self):
        """ Returns mean squared error of model prediction against training data"""
        return np.mean(self._residuals**2)

    def condition_number(self) -> float | Exception:
        """Returns condition number of covariance matrix"""
        if not np.isfinite(self.pcov).all():
            return Exception("Parameter coveriance matrix has a non-finite element")
        return np.linalg.cond(self.pcov)

    def correlation_matrix(self) -> np.ndarray | Exception:
        """Returns model parameter correlation matrix"""
        if not np.isfinite(self.pcov).all():
            return Exception("Parameter coveriance matrix has a non-finite element")
        Dinv = np.diag(1 / np.sqrt(np.diag(self.pcov)))
        return Dinv @ self.pcov @ Dinv

    def studentised_residuals(self):
        """Returns the studentised residuals"""
        x_transpose = self._input_matrix.transpose()
        num_experiments, _ = x_transpose.shape
        x_transpose = np.c_[np.ones(num_experiments), x_transpose]
        H = x_transpose.dot(
            np.linalg.inv(x_transpose.T.dot(x_transpose)).dot(x_transpose.T)
        )
        std_dev = (
            1 / (num_experiments - self._num_params) * (np.sum(self._residuals**2))
        )
        return self._residuals / (np.sqrt(std_dev * (1 - np.diag(H))))

    def sobol_idx(self, num_samples: int, num_std_dev:int):
        """
        Samples model and returns 1st and 2nd order sobol indicies for each variable at each input values

        Uses sobol sequence to sample function for all input values defined in x for num_sample configurations of model parameter values.

        """

        problem = {
            "num_vars": len(list(self.popt.keys())),
            "names": list(self.popt.keys()),
            "bounds": np.c_[
                -num_std_dev * np.sqrt(np.diag(self.pcov)).T + list(self.popt.values()),
                num_std_dev * np.sqrt(np.diag(self.pcov)).T + list(self.popt.values()),
            ],
        }
        param_vals = sample(problem, num_samples, calc_second_order=False)
        model_output_for_all_param_vals = []
        for params in param_vals:
            model_output_for_all_param_vals.append(self._function(self._input_matrix, *params))
        output_subset_for_analysis = []
        sobol_idxs = {
            "first_order": {param_name: [] for param_name in problem["names"]},
            "total_order": {param_name: [] for param_name in problem["names"]},
        }
        max_idx = len(model_output_for_all_param_vals[0])
        for output_idx in range(0, max_idx):
            for model_output in model_output_for_all_param_vals:
                output_subset_for_analysis.append(model_output[output_idx])
            sobol_idx = sobol.analyze(
                problem,
                np.array(output_subset_for_analysis).flatten(),
                calc_second_order=False,
                print_to_console=False,
            )
            output_subset_for_analysis = []
            for i, param_name in enumerate(problem["names"]):
                sobol_idxs["first_order"][param_name].append(float(sobol_idx["S1"][i]))
                sobol_idxs["total_order"][param_name].append(float(sobol_idx["ST"][i]))

        return sobol_idxs

    # TODO: Let the user aggregate the data passed to the sobol index calculation
    # def sobol_idx_with_agg(self):
    #    pass

