# -*- coding: utf-8 -*-
import dash
from app import setup_dash_model_app

'''
external_scripts = [
    {
        'src': "https://code.jquery.com/jquery-3.4.1.min.js",
        'integrity': "sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo=",
        'crossorigin': 'anonymous',
    },
]
'''
external_scripts = []

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ["https://cdnjs.cloudflare.com/ajax/libs/meyer-reset/2.0/reset.min.css"]
external_stylesheets = []

app = dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets)
setup_dash_model_app(app)

if __name__ == '__main__':
    app.run_server(debug=True)
