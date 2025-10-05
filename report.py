from jinja2 import Template
import plotly.graph_objects as go
import numpy as np
from user_fn import user_fn
import plotly.express as px
from lib import FittedModel
from itertools import combinations
import tomllib

# Example data
def render_report(model: FittedModel, clean_dataset):

    with open("reporttext.toml","rb") as f:
        config = tomllib.load(f)
    report = {"title": "Report", "conclusion": "This is the conclusion of the report."}

    correlation_matrix = np.c_[
        np.array(list(model.popt.keys())), model.correlation_matrix()
    ].tolist()
    for row_idx, row in enumerate(correlation_matrix):
        for col_idx, value in enumerate(row):
            if col_idx > 0:
                correlation_matrix[row_idx][col_idx] = float(value)
    parameters = [{"name": key, "value": value} for key, value in model.popt.items()]
    figures = []
    for row_num,row in enumerate(model._input_matrix):
        fig1 = px.scatter(
            x=[float(x) for x in row],
            y=[float(y) for y in clean_dataset.y.flatten()],
            title="Data with Fitted Model",
        )
        fig1.add_scatter(
            x=[float(x) for x in row],
            y=user_fn(clean_dataset.x, *[v for k, v in model.popt.items()]).flatten(),
            mode="lines",
            name="Fitted Model",
        )
        fig1.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(b=50, t=50, l=50, r=50),
            xaxis_title=clean_dataset.mapping[row_num],
            yaxis_title=clean_dataset.output_var
        )
        fig1_html = fig1.to_html(include_plotlyjs="cdn", full_html=False)
        figures.append({"id": "figure1", "html": fig1_html, "caption": f"{config["fitted_model"]["caption"]}"})

        #Plot 2: Studentised Residuals
        try:
            fig2 = px.scatter(
                x=row,
                y=model.studentised_residuals().flatten(),
                title="Studentised Residuals",
            )
            fig2.update_layout(margin=dict(b=50, t=50, l=50, r=50))
            fig2_html = fig2.to_html(include_plotlyjs=False, full_html=False)
        except:
            fig2_html="<p></p>"

        figures.append({
            "id": "figure2",
            "html": fig2_html,
            "caption": f"{config["studentised_residuals"]["caption"]}",
        })

        # Plot 3: Total order sobol over time
        sobol_dict = model.sobol_idx(1000,1)
        fig3 = px.scatter()
        for param_name, sobol_idxs in sobol_dict["total_order"].items():
            fig3.add_scatter(
                x=row, y=sobol_idxs, mode="lines", name=param_name
            )

        fig3.update_layout(
        xaxis_title=clean_dataset.mapping[row_num],
        yaxis_title="Total order sobol index",
        title="Total Order Sobol Index of Model Parameters",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(b=50, t=50, l=50, r=50),
        )

        fig3_html = fig3.to_html(
        include_plotlyjs=False, full_html=False
        )  # Figures for Template
        figures.append({
            "id": "figure3",
            "html": fig3_html,
            "caption": f"{config["sobol_index"]["caption"]}",
        })
    def add_confidence_ellipse(cov,mean_x,mean_y,param_1_index,param_2_index, fig, n_std=3, **kwargs):
        # Original source of this algorithm : https://gist.github.com/CarstenSchelp/b992645537660bda692f218b562d0712, adapted to plotly using ChatGPT
        pearson = cov[param_1_index, param_2_index] / np.sqrt(cov[param_1_index, param_1_index] * cov[param_2_index, param_2_index])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        scale_x = np.sqrt(cov[param_1_index, param_1_index]) * n_std
        scale_y = np.sqrt(cov[param_2_index, param_2_index]) * n_std

        theta = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = ell_radius_x * np.cos(theta)
        y_ellipse = ell_radius_y * np.sin(theta)

        theta_rot = np.deg2rad(45)
        x_rot = x_ellipse * np.cos(theta_rot) - y_ellipse * np.sin(theta_rot)
        y_rot = x_ellipse * np.sin(theta_rot) + y_ellipse * np.cos(theta_rot)

        x_scaled = x_rot * scale_x
        y_scaled = y_rot * scale_y

        x_final = x_scaled + mean_x
        y_final = y_scaled + mean_y

        fig.add_trace(go.Scatter(x=x_final, y=y_final, **kwargs))
    parameter_names = list(model.popt.keys())
    ellipsoids={}
    for param_1,param_2 in combinations(parameter_names,2):
        fig = px.scatter()
        param_1_index=parameter_names.index(param_1)
        param_2_index=parameter_names.index(param_2)
        add_confidence_ellipse(model.pcov,model.popt[param_1],model.popt[param_2],param_1_index,param_2_index,fig)
        fig.update_layout(title=f"{param_1}-{param_2} Confidence Ellipsoid",xaxis_title=param_1,yaxis_title=param_2)
        ellipsoids[(param_1,param_2)]=fig

    for ellipsoid_name,ellipsoid in ellipsoids.items():
        figures.append(
            {
                "id": f"figure{ellipsoid_name}",
                "html": ellipsoid.to_html(include_plotlyjs=False,full_html=False),
                "caption": f"{config["confidence_ellipsoid"]["caption"]}",
            }
        )
    template = Template(open("templates/template.html", "r").read())
    measures = [{"name":"MSE","value":model.MSE()},{"name":"Covariance condition number","value":model.condition_number()}]
    rendered_html = template.render(
        report=report,
        correlation_matrix=correlation_matrix,
        parameters=parameters,
        measures=measures,
        figures=figures,
    )

    with open("report.html", "w") as f:
        f.write(rendered_html)
