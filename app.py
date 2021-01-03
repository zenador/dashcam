# -*- coding: utf-8 -*-
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
# from dash.exceptions import PreventUpdate
import json
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os#, sys
from flask.helpers import get_root_path
from db import Db

# data

def flatten_df(df, serialised=False):
    params = pd.json_normalize(list(map(lambda x: json.loads(x), df["parameters"]))).add_prefix('parameters.')
    metrics = pd.json_normalize(list(map(lambda x: json.loads(x), df["metrics"]))).add_prefix('metrics.')
    if serialised:
        df['date'] = df[['year', 'month', 'day']].apply(lambda x: "{:04d}-{:02d}-{:02d}".format(*x), axis=1) # from csv
    else:
        df['date'] = df[['year', 'month', 'day']].apply('-'.join, axis=1) # from presto
    rest = df[["source", "model_name", "model_version", "date"]]
    return rest.join(params).join(metrics)

def recalculate(df):
    df = df.copy(deep=True)
    df["metrics.precision"] = df["metrics.tp"] / (df["metrics.tp"] + df["metrics.fp"])
    df["metrics.recall"] = df["metrics.tp"] / (df["metrics.tp"] + df["metrics.fn"])
    df["metrics.f1"] = (2 * df["metrics.precision"] * df["metrics.recall"]) / (df["metrics.precision"] + df["metrics.recall"])
    df["metrics.fpr"] = df["metrics.fp"] / (df["metrics.fp"] + df["metrics.tn"])
    df["metrics.total"] = df[["metrics.tp", "metrics.tn", "metrics.fp", "metrics.fn"]].apply(sum, axis=1)
    df["metrics.accuracy"] = (df["metrics.tp"] + df["metrics.tn"]) / df["metrics.total"]
    df["metrics.pos_pred_pct"] = (df["metrics.tp"] + df["metrics.fp"]) / df["metrics.total"]
    df["metrics.pos_label_pct"] = (df["metrics.tp"] + df["metrics.fn"]) / df["metrics.total"]
    return df.round(5)

def regroup(df):
    df = df[["date", "metrics.tp", "metrics.tn", "metrics.fp", "metrics.fn"]].groupby(["date"]).sum()
    return recalculate(df) * 100

# graph

colors = {
    'background': '#111111',
    'text': '#7FDBFF',
}

default_query = """SELECT *
FROM results
where model_name = 'demo'
--and year||month||day >= ''
--order by cast(json_extract_scalar(metrics, '$.total') as integer) desc
--limit 100"""

main_children = [
    dcc.Textarea(
        id='query_box',
        placeholder='Presto Query',
        value=default_query,
        style={'width': '100%'},
        rows=6,
    ),
    html.Button('Run', id='run_qry'),
    html.Button('Reset', id='reset'),
]
inputs = []
form_controls = []

def make_form_control(name, fixed_set):
    form_control_name = "checklist_{}".format(name)
    label_name = "label_{}".format(name)
    div = html.Div(id="config_{}".format(name), children=[
        html.Label(children=name if fixed_set else "", id=label_name),
        dcc.Dropdown(
            id=form_control_name,
            options=[{'label': val, 'value': val} for val in fixed_set],
            value=fixed_set,
            multi=True,
            optionHeight=20,
            style={} if fixed_set else {'display': 'none'},
        ),
    ])
    main_children.append(div)
    inputs.append(Input(form_control_name, 'value'))
    if not fixed_set:
        form_controls.extend([
            Output(label_name, 'children'),
            Output(form_control_name, 'options'),
            Output(form_control_name, 'value'),
            Output(form_control_name, 'style'),
        ])

default_measures = ["precision", "recall", "f1", "fpr", "accuracy", "pos_pred_pct", "pos_label_pct"]
make_form_control("_measures", default_measures)
default_graph_modes = ["lines", "text"]
make_form_control("_graph_modes", default_graph_modes)

MAX_CONFIG = 20
for i in range(MAX_CONFIG):
    make_form_control(i, [])

main_children.extend([
    dcc.Loading(
        id="loading-main",
        children=[
            dcc.Graph(id='graph'),
            dash_table.DataTable(
                id='table',
                sort_action='native',
                # n_fixed_rows=1,
                style_header={
                    'backgroundColor': 'rgb(30, 30, 30)',
                    'padding-right': '1em',
                },
                style_cell={
                    'backgroundColor': 'rgb(50, 50, 50)',
                    'color': 'white',
                },
                style_table={'overflowX': 'auto'},
            ),
        ],
        type="circle",
    ),
    html.Div(id='df_hidden_cache', style={'display': 'none'}),
])

def make_line(df, name, graph_modes):
    y_values = df["metrics."+name].values
    mode = "+".join(["markers"] + graph_modes)
    return go.Scatter(
        x=df.index.values,
        y=y_values,
        text=list(map(lambda x: '{}: {:,.3f}'.format(name, x), y_values)),
        mode=mode,
        hoverinfo='x+y+name',
        name=name,
        # marker={
        #     'size': 15,
        #     'opacity': 0.5,
        #     'line': {'width': 0.5, 'color': 'white'}
        # }
    )

def make_graph(df, measures, graph_modes):
    return {
        'data': [make_line(df, measure, graph_modes) for measure in measures],
        'layout': {
            'plot_bgcolor': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {
                'color': colors['text'],
            },
            "yaxis": {
                "range": [0, 105],
            },
        },
    }

def get_field_names(df):
    return list(filter(lambda x: x.startswith("parameters."), df.columns))

magicTrue = "_True_"
magicFalse = "_False_"
magicNan = "_nan_"

def bool_to_str(x):
    if x is True:
        return magicTrue
    elif x is False:
        return magicFalse
    else:
        return x

def str_to_bool(x):
    if x == magicTrue:
        return True
    elif x == magicFalse:
        return False
    else:
        return x

def null_to_str(x):
    if pd.isna(x): # np.isnan doesn't work for some reason
        return magicNan
    else:
        return x

def str_to_null(x):
    if x == magicNan:
        return np.nan
    else:
        return x

def make_native(x):
    # convert from numpy to native python data types
    if isinstance(x, (np.ndarray, np.generic)):
        return x.item()
    else:
        return x

def tag(val, name):
    if name == "country":
        if val == "CN":
            return "China"
        elif val == "JP":
            return "Japan"
        elif val == "KR":
            return "Korea"
    if name == "colour":
        if val == 1:
            return "Red"
        elif val == 2:
            return "Blue"
        elif val == 3:
            return "Green"
    return val

def make_full_path(filename):
    # file_dir = os.path.dirname(os.path.realpath('__file__'))
    # file_dir = sys.path[0]
    file_dir = get_root_path(__name__)
    full_path = os.path.join(file_dir, filename)
    return full_path

def setup_dash_model_app(app):
    app.layout = html.Div(id='main', children=main_children)

    with app.server.app_context():
        conn = Db()

    @app.callback(
        Output("df_hidden_cache", 'children'),
        [Input('run_qry', 'n_clicks')],
        [State('query_box', 'value')]
    )
    def update_df(run_clicks, query):
        if run_clicks is None: # run button not pressed yet, first init
            # raise PreventUpdate()
            source_filename = "example.csv"
            source_data = pd.read_csv(make_full_path(source_filename))
            df = flatten_df(source_data, serialised=True)
        else:
            source_data = pd.read_sql_query(query.replace('%', '%%'), con=conn.get_db(), params={})
            df = flatten_df(source_data)
        return df.to_json(orient='split')

    @app.callback(
        form_controls,
        [Input("df_hidden_cache", 'children'), Input('reset', 'n_clicks')]
    )
    def update_form(cached_df, reset_clicks):
        collated = []
        df = pd.read_json(cached_df, orient='split', convert_dates=False)
        field_names = get_field_names(df)
        for field_name in field_names[:MAX_CONFIG]:
            name = field_name.replace("parameters.", "")
            options = []
            values = sorted(df[field_name].unique())
            values = list(map(make_native, values))
            values = list(map(bool_to_str, values))
            values = list(map(null_to_str, values))
            for val in values:
                options.append({'label': str(tag(val, name)), 'value': val})
            collated.extend([name, options, values, {}])
        for _ in range(MAX_CONFIG - len(field_names)):
            collated.extend(["", [], [], {'display': 'none'}])
        return collated

    @app.callback(
        [
            Output('graph', 'figure'),
            Output('table', 'columns'),
            Output('table', 'data'),
        ],
        inputs + [Input('df_hidden_cache', 'children')]
    )
    def update_graph_table(*filters):
        filters = list(filters)
        cached_df = filters.pop()
        measures = filters.pop(0)
        graph_modes = filters.pop(0)
        filtered = pd.read_json(cached_df, orient='split', convert_dates=False)
        field_names = get_field_names(filtered)
        filters = filters[:len(field_names)]
        for i, values in enumerate(filters):
            values = list(map(str_to_bool, values))
            values = list(map(str_to_null, values))
            filtered = filtered[filtered[field_names[i]].isin(values)]
        filtered = recalculate(filtered)
        dff = regroup(filtered)
        return [
            make_graph(dff, measures, graph_modes),
            [{"name": i, "id": i} for i in filtered.columns],
            # filtered.astype({k:"str" for k in filtered.select_dtypes(include=['bool']).columns.tolist()}).to_dict('records'), # convert bool to string to work around bug where datatable doesn't display bool that appears on python 2 with older pandas
            filtered.to_dict('records'),
        ]
