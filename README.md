# Modelling Utils
#### Description:
This tool simplifies model fitting and analysis.

You only need two things:
* A function defined in python representing your model
* Data defined in a CSV format which the model is supposed to fit

This tool is designed to allow you to rapidly go from having data and a proposed
model to a fitted model with standardised analysis of your model fit allowing you
to compare various models, and allowing you to identify areas of improve in your models.

* You are provided with a goodness-of-fit metric, the mean squared error.
* You are provided with the correlation matrix which can help identify redundant combinations of parameters,
and the condition number which can help identify if your model is overparameterised.
* You are provided with a plot of your studentised residuals , which should be normally distributed if there is no systematic error,
the shape of the studentised residual plot can help identify what systematic error exists in your model.
* You are provided with a plot of the confidence ellipsoid of each combination of parameters to help identify parameter correlation,
and help choose which kind of optimality should be used when designing experiments to improve parameter estimates.
* You are provided with plots of the total order sobol index of each parameter over time to show which parameters
are most important to the model prediction at a given input.


#### Usage:
```bash
python tool.py data.csv 0.1 user_fn --input_var=x,y --output_var=z
```
user\_fn.py:
```python
import numpy as np
def user_fn(x,a,b):
    return a*np.sin(x[0])+np.exp(b*x[1])
```
data.csv
```csv
x,y,z
0.000000000000000000e+00,1.000000000000000000e+01,-1.754836588871176994e-01
2.004008016032064049e-02,9.979959919839679117e+00,-1.286262666707294611e-01
4.008016032064128098e-02,9.959919839679358233e+00,2.052333455528486672e-01

```
#### Output:
##### Terminal Output:
The tool writes JSON to stdout with the following structure:
```json
{
    "optimal_parameters": {
        "a": 0.050045896053248905,
        "b": -1.4286735928332486
    },
    "condition_number": 61.80734741389808,
    "correlation_matrix": [
        [
            1.0,
            -0.3095007949004337
        ],
        [
            -0.30950079490043364,
            1.0
        ]
    ],
    "mean_sqr_err": 3.9845655064777,
    "sobol_idxs": {
        "first_order": {
            "a": [
                1,
                2,
                3
            ],
            "b": [
                1,
                2,
                3
            ]
        },
        "total_order": {
            "a": [
                1,
                2,
                3
            ],
            "b": [
                1,
                2,
                3
            ]
        }
    }
}
```


##### Report:
A HTML report is also generated with interactive graphs using plotly. The report is structured and formatted using bootstrap.
#### Contents:
* ./examples/data/data.csv: example data used to fit example function
* ./examples/data/user\_fn.py
* ./examples/templates/template.html: [Jinja]("https://jinja.palletsprojects.com/en/stable/") template used for generating report.html
* ./modelling_utils.py: Library containing the RawDataset, CleanDataset, and FittedModel classes. This was factored out of the tool.py to allow others to use the fitting code without forcing them to use the tool,
* ./examples/tool.py: script to parse arguments and fit the model using the classes defined in lib.py
* ./examples/report.py: Library used to generate the HTML report.
* ./examples/reporttext.toml: Config file containing the captions in the HTML report
#### Modelling Theory:
This tool performs parameter estimation and global sensitivity analysis for user-defined models. The outputs provide information about model fit, parameter identifiability, and the relative importance of each parameter.

- **Optimal Parameters** - Best fit of the model parameters to the dataset. Obtained via finding a local optimum to the least squares problem. The implemented curve fitting algorithm via SciPy defaults to the Levenberg-Marquardt algorithm. Future work could be to generalise the model fitting to allow for the use of deterministic global optimisation algorithms.

- **Condition Number** - Large values indicate redundant model variables.

- **Correlation Matrix** - Pairwise correlations between parameter estimates. High correlations suggest that parameters are not independently identifiable (i.e. multiple values of this combination of two parameters could explain the data equally well).

- **Mean Squared Error (MSE)** - The average squared deviation between observed outputs and model predictions. This measures goodness-of-fit, with smaller values indicating better agreement between the model and data.

- **Sobol Sensitivity Indices** - Variance-based measures of parameter importance:
    - *First-order indices* capture the fraction of output variance explained by each parameter alone.
    - *Total-order indices* capture the total contribution of each parameter, including all interactions with others.
    - If the total order $\approx$ first order: There is little interaction between this parameter and other parameters
    - If the total order $=0$: The model isn't dependant on this parameter, fix it to a value to improve estimates of other parameters.


#### Roadmap:
- [ ] Add citations for statements in the captions of figures
- [ ] Add Chi^2 goodness of fit test
- [ ] Add ability to override the axis labels on the figures
- [x] Update dependant variable to be on the y axis of the graphs


