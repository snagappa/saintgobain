#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 11:07:56 2019

@author: snagappa
"""

import numpy as np
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

import data_loader
import data_analysis
from data_analysis import Regressor


def create_geodict():
    map_dict = dict(
        type="scattergl",
        # locationmode="country names",
        # lon=[],
        # lat=[],
        x=[],
        y=[],
        text=[],
        mode="markers",
        marker=dict(
            size=0,
            opacity=0.4,
            color=[]
        )
    )

#    geodict = dict(
#        scope="europe",
#        projection=dict(type="natural earth"),
#        showland=True,
#        landcolor="rgb(250, 250, 250)",
#        subunitcolor="rgb(217, 217, 217)",
#        countrycolor="rgb(217, 217, 217)"
#        #bgcolor="#191A1A",
#        #countrywidth=1.0,
#        #subunitwidth=1.0,
#        #resolution=50
#    )

    layout = dict(
        autosize=True,
        height=750,
        hovermode="closest",
        # plot_bgcolor="#191A1A",
        # paper_bgcolor="#020202",
        # geo=geodict,
        xaxis=dict(
            range=[-5.56, 9.67]),
        yaxis=dict(
            range=[41.30, 51.13],
            scaleanchor='x',
            scaleratio=1.0)
    )

    return map_dict, layout


external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
    ]
app = dash.Dash(external_stylesheets=external_stylesheets)
df = data_loader.df
df_referendum = data_loader.df_referendum
df_latlon = data_loader.df_latlon
df_pop_str_edu = data_loader.df_pop_str_edu
df_popfiscal = data_loader.df_popfiscal
dfcorr = data_loader.dfcorr
regressors, rmse, rmse_scores, x_keys, y_keys, data = (
    data_analysis.load_regresults())
big_regressors, big_rmse, big_rmse_scores, big_x_keys, big_y_keys, big_data = (
    data_analysis.load_regresults(
        "results/regressors_bigcities_impfeatures.p"))


def plot_overall_results():
    mainland, overseasterr, abroad = data_loader.get_referendum_by_regions()
    referendum_regions = ["Mainland", "Overseas Territories", "Abroad"]

    name_fields = ["Libellé de la commune", "Libellé du département"]
    top_ca_ix = np.argsort(df["Choix A (%)"].values)[-10:]
    top_cb_ix = np.argsort(df["Choix B (%)"].values)[-10:]

    choix_a_frac = df["Choix A"].sum()/(df["Choix A"] + df["Choix B"]).sum()
    choix_b_frac = df["Choix B"].sum()/(df["Choix A"] + df["Choix B"]).sum()
    layout = html.Div(
        [
            html.Div(
                [
                    html.H3("Overall result"),
                    dcc.Markdown(
                        "We plot the overall result by aggregating valid "
                        "votes from all regions. Referendum results ignore "
                        "abstentions and blanc votes."),
                    html.P("Choix A: " +
                           str(np.round(choix_a_frac*100, 3)) + " %"),
                    html.P("Choix B: " +
                           str(np.round(choix_b_frac*100, 3)) + " %"),
                    html.P("Referendum result: " +
                           ("Choix A" if choix_a_frac > choix_b_frac
                            else "Choix B")),
                    html.H5("Top 10 cities voting for Choix A by percent "
                            "[Commune, Department %]"),
                    html.P(', '.join([
                        '[' + ', '.join(df.iloc[ix_][name_fields].values)
                        + ', ' + str(np.round(df.iloc[ix_]["Choix A (%)"], 1)) + ']'
                        for ix_ in top_ca_ix])),
                    html.H5("Top 10 cities voting for Choix B by percent "
                            "[Commune, Department %]"),
                    html.P(', '.join([
                        '[' + ', '.join(df.iloc[ix_][name_fields].values)
                        + ', ' + str(np.round(df.iloc[ix_]["Choix B (%)"], 1)) + ']'
                        for ix_ in top_cb_ix])),
                    html.H5("Results according to Mainland France, French "
                            "Overseas Territories and France Abroad"),
                    dcc.Graph(figure=dict(
                        data=[
                            dict(
                                x=referendum_regions,
                                y=[mainland[cat],
                                   overseasterr[cat],
                                   abroad[cat]],
                                name=cat,
                                type="bar"
                                )
                            for cat in mainland.index],
                        layout={"barmode": 'stack'}
                        )
                    )
                ], className="ten columns"
            )
        ], className="row"
    )
    return layout


def plot_histograms():
    layout = html.Div(
        [
            html.H3("Histogram of Votes by Category")
        ] +
        [
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(
                                id='hist_'+key_,
                                figure={
                                    'data': [
                                        {
                                            'x': df[key_],
                                            'name': key_,
                                            'type': 'histogram',
                                            'hoverinfo': 'x+y',
                                            'title': key_
                                        }
                                    ],
                                    'layout': {
                                        'title': key_}
                                }
                            )
                        ], className="five columns"
                    )
                    for key_ in options
                ], className="row"
            )
            for options in (['Choix A (%)', 'Choix B (%)'],
                            ['Abstentions (%)', 'Blancs (%)'])
        ]
    )
    return layout


def plot_correlation():
    map_data, layout = create_geodict()
    layout.pop("xaxis")
    layout.pop("yaxis")
    layout = html.Div(
        [
            html.H3("Correlations between the variables"),
            html.P("We plot the correlation between the various input and "
                   "output variables. All population fields are converted "
                   "to fraction of the population (of the commune) by "
                   "normalising by the total population for that commune."),
            html.Div(
                [
                    dcc.Graph(
                        id='correlations',
                        figure={
                            'data': [
                                {
                                    'x': dfcorr.columns.tolist(),
                                    'y': dfcorr.index.tolist(),
                                    'z': dfcorr.values,
                                    'type': 'heatmap',
                                    # 'hoverinfo': 'x+y',
                                    'title': "Correlations"
                                }
                            ],
                            'layout': {
                                "height": 950,
                                "margin": {'l': 600}}
                        }
                    )
                ]
            ),
            html.P("We can plot the a scatter of the variables to visually "
                   "examine any potential relationships."),
            html.Div(
                [
                    dcc.Dropdown(
                        id='xcorr_key',
                        options=[
                            {'label': key_, 'value': key_}
                            for key_ in dfcorr.index],
                        value=dfcorr.index[0]
                    ),
                    dcc.Dropdown(
                        id='ycorr_key',
                        options=[
                            {'label': key_, 'value': key_}
                            for key_ in dfcorr.columns],
                        value=dfcorr.columns[0]
                    ),
                    dcc.Graph(
                        id="corr_plot",
                        figure=go.Figure(data=[map_data],
                                         layout=go.Layout(**layout))
                    )
                ], className="ten columns"
            )
        ], className="row"
    )
    return layout


def map_layout():
    plot_options = [key_ for key_ in df.keys()[4:]
                    if key_ not in ("Latitude", "Longitude")]
    map_data, layout = create_geodict()
    return html.Div(
        [
            html.H3("Visualising the data"),
            html.P("Data is plotted according to the geographic location of "
                   "the communes. The region is restricted to mainland "
                   "France. The data also contains results from regions "
                   "outside mainland France, this data is not shown in the "
                   "figure."),
            html.P("Colour values are scaled according to data from the "
                   "entire data set. Some ranges may not correspond to any "
                   "data shown on the map."),
            html.P("Colous is used to indicate the field of interest."
                   "<br>Marker size is indicative of the 'Inscrits' field."),
            html.Div(
                [
                    dcc.Dropdown(
                        id='map_key',
                        options=[
                            {'label': key_, 'value': key_}
                            for key_ in plot_options],
                        value='Inscrits'
                    ),
                    dcc.Graph(
                        id="map_view",
                        figure=go.Figure(data=[map_data],
                                         layout=go.Layout(**layout))
                    )
                ], className="ten columns"
            )
        ], className="row"
    )


def feature_importance_graph():
    layout = html.Div(
        [
            html.H3("Predictors of Referendum Outcome"),
            html.P("The feature importances (or coefficients) indicate which "
                   "features in the input influence the predicted values.<br>"
                   "These can be plotted below for the various regressors."),
            html.Div(
                [
                    dcc.Dropdown(
                        id="featimp_regressor",
                        options=[
                            {"label": key_, "value": key_}
                            for key_ in regressors],
                        value="RandomForest"
                    ),
                    dcc.Dropdown(
                        id="featimp_category",
                        options=[
                            {"label": key_, "value": key_}
                            for key_ in y_keys],
                        value=y_keys[0]
                    ),
                    dcc.Graph(
                        id="featimp_graph",
                        figure=go.Figure(data=[],
                                         layout=go.Layout())
                    )
                ], className="ten columns"
            )
        ], className="row"
    )
    return layout


def performance_graph():
    layout = html.Div(
        [
            html.H3("Performance of the Regressors"),
            html.P("We show the predicted vs true values of the corresponding "
                   "outcome variable for the training and testing data sets."),
            html.Div(
                [
                    dcc.Dropdown(
                        id="perf_datasel",
                        options=[
                            {"label": "All cities/All features",
                             "value": "all"},
                            {"label": "Big cities (population >10e3) / Important features",
                             "value": "big"}],
                        value="all"),
                    dcc.Dropdown(
                        id="perf_regressor",
                        options=[
                            {"label": key_, "value": key_}
                            for key_ in regressors],
                        value="RandomForest"),
                    dcc.Dropdown(
                        id="perf_category",
                        options=[
                            {"label": key_, "value": key_}
                            for key_ in y_keys],
                        value=y_keys[0]
                    ),
                    dcc.Graph(
                        id="perf_graph",
                        figure=go.Figure(data=[], layout=go.Layout())
                    )
                ], className="ten columns"
            )
        ], className="row"
    )
    return layout


def serve_layout():
    layout = html.Div(
        [
            html.H1("Referendum Analysis"),
            html.P("We assume the referendum categories (A/B/Abst/Blanc) to "
                   "be the output variables in our analysis. We aim to study "
                   "the main drivers behind a particular category in terms of "
                   "some input variables describing each commune "
                   "(population age, gender, education, etc.)."),
            html.P("Referendum data contains results from regions outside "
                   "mainland France. Supporting data corresponding to "
                   "population, education, etc. does not list values for "
                   "these regions. Consequently, we constrain our analysis "
                   "of the data to mainland France"),
            plot_overall_results(),
            plot_histograms(),
            plot_correlation(),
            map_layout(),
            feature_importance_graph(),
            performance_graph()
        ]
    )
    return layout


app.layout = serve_layout


@app.callback(Output("map_view", "figure"),
              [Input("map_key", "value")])
def draw_map(map_key):
    map_data, layout = create_geodict()
    x = df["Longitude"].values
    y = df["Latitude"].values
    hovertext = np.asarray([
        name_[0] + ', ' + name_[1] + '<br>' + map_key + ' = ' + str(name_[2])
        for name_ in
        zip(df["Libellé de la commune"].values,
            df["Libellé du département"].values,
            df[map_key].values)])
    marker_size = df["Inscrits"].values
    color = df[map_key].values
    valid_ix = np.isfinite(color)

    map_data["x"] = x[valid_ix]
    map_data["y"] = y[valid_ix]
    map_data["hovertext"] = hovertext[valid_ix]
    map_data["hoverinfo"] = "text"
    max_size_raw = df["Inscrits"].max()
    max_size_marker = 80
    map_data["marker"] = dict(
        opacity=0.5,
        size=marker_size[valid_ix],
        sizemode="area",
        sizeref=2. * max_size_raw / (max_size_marker ** 2),
        sizemin=2,
        color=color[valid_ix],
        colorscale="Jet",
        showscale=True
        )
    figure = {"data": [map_data], "layout": layout}
    return figure


@app.callback(Output("corr_plot", "figure"),
              [Input("xcorr_key", "value"),
               Input("ycorr_key", "value")])
def draw_correlation(xcorr_key, ycorr_key):
    map_data, layout = create_geodict()
    layout.pop("xaxis")
    layout.pop("yaxis")
    color_key = "Population en 2013 (princ)"

    map_data["x"] = df[xcorr_key].values
    map_data["y"] = df[ycorr_key].values
    map_data["hovertext"] = [
        name_[0] + ', ' + name_[1]
        for name_ in
        zip(df["Libellé de la commune"].values,
            df["Libellé du département"].values)]
    map_data["hoverinfo"] = "x+y+text"
    max_size_raw = df["Inscrits"].max()
    max_size_marker = 80
    map_data["marker"] = dict(
        # opacity=0.5,
        size=df[color_key].values,
        sizemode="area",
        sizeref=2. * max_size_raw / (max_size_marker ** 2),
        sizemin=2,
        color=df[color_key].values,
        colorscale="Jet",
        showscale=True,
        colorbar={
            "title": color_key}
        )
    layout["xaxis"] = {
        "title": xcorr_key}
    layout["yaxis"] = {
        "title": ycorr_key}

    figure = {"data": [map_data], "layout": layout}
    return figure


@app.callback(Output("featimp_graph", "figure"),
              [Input("featimp_regressor", "value"),
               Input("featimp_category", "value")])
def draw_featimp(regr_name, cat_name):
    cat_idx = np.where(np.asarray(y_keys) == cat_name)[0][0]
    regr = regressors[regr_name].regressor[cat_idx]

    feature_importances = getattr(
        regr,
        "feature_importances_", None)
    if feature_importances is None:
        feature_importances = regr.coef_

    trace = dict(
        type="bar",
        x=x_keys,
        y=feature_importances,
        hoverinfo="x+y",
        )
    layout = {
        "height": 800,
        "margin": {'b': 500}}
    figure = {"data": [trace], "layout": layout}
    return figure


@app.callback(Output("perf_graph", "figure"),
              [Input("perf_regressor", "value"),
               Input("perf_category", "value"),
               Input("perf_datasel", "value")])
def draw_perf(regr_name, cat_name, datasel):
    cat_idx = np.where(np.asarray(y_keys) == cat_name)[0][0]
    if datasel == "all":
        regr = regressors[regr_name].regressor[cat_idx]
        plot_data = {
            "train": {
                'x': data["default"]["train_x"],
                'y': data["default"]["train_y"].values[:, cat_idx]},
            "test": {
                'x': data["default"]["test_x"],
                'y': data["default"]["test_y"].values[:, cat_idx]}
            }
    else:
        regr = big_regressors[regr_name].regressor[cat_idx]
        plot_data = {
            "train": {
                'x': big_data["default"]["train_x"],
                'y': big_data["default"]["train_y"].values[:, cat_idx]},
            "test": {
                'x': big_data["default"]["test_x"],
                'y': big_data["default"]["test_y"].values[:, cat_idx]}
            }
    figdata = []
    pred_acc = {}
    for tt_ in plot_data.keys():
        x_ = plot_data[tt_]["x"]
        ytrue_ = plot_data[tt_]["y"]
        y_ = regr.predict(x_)
        pred_acc[tt_] = (
            "RMSE = " +
            str(round(data_analysis.mean_squared_error(ytrue_, y_)**0.5, 2)) +
            ", MAE = " +
            str(round(data_analysis.mean_absolute_error(ytrue_, y_), 2)) +
            ", R2 = " +
            str(round(data_analysis.r2_score(ytrue_, y_), 2)))
        trace, layout = create_geodict()
        trace["x"] = ytrue_
        trace["y"] = y_
        trace["hovertext"] = [
            name_[0] + ', ' + name_[1]
            for name_ in
            zip(df["Libellé de la commune"].values,
                df["Libellé du département"].values)]
        trace["hoverinfo"] = "x+y+text"
        trace["marker"] = dict(
            size=4)
        trace["name"] = tt_.title() + " (" + pred_acc[tt_] + ")"
        figdata.append(trace)

    figdata.append(
        {
            "type": "scatter",
            "mode": "lines",
            "line": {
                "opacity": 0.5,
                "color": "black"},
            "x": [0, 100],
            "y": [0, 100],
            "hovertext": "none",
            "text": "none",
            "showlegend": False
        }
    )
    test_plot, layout = create_geodict()
    layout.pop("xaxis")
    layout.pop("yaxis")
    layout["height"] = 750
    layout["width"] = 750
    layout["xaxis"] = {
        "title": "True",
        "range": [0, 100]}
    layout["yaxis"] = {
        "title": "Predicted",
        "range": [0, 100],
        "scaleanchor": 'x',
        "scaleratio": 1.0}
    layout["title"] = cat_name

    figure = {"data": figdata, "layout": layout}
    return figure


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=True, threaded=True)
