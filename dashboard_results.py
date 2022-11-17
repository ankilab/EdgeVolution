import dash
from dash import html, dcc
import pandas as pd
import json
import os
import numpy as np

ga_run = 'ga_20220930-134649'
path = 'Results/' + ga_run

with open(path + '/params.json') as f:
    params = json.loads(f.read())


def get_all_fitnesses():
    fitnesses_all = []
    for i in range(1, params['generations'] + 1):
        fitnesses_generation = []
        try:
            for individual in os.listdir(path + f'/Generation_{i}'):
                with open(path + f'/Generation_{i}/{individual}/results.json') as f:
                    results = json.loads(f.read())
                    print(results['memory_footprint'])
                    try:
                        fitnesses_generation.append(results['test_acc'])
                    except:
                        # this means that there is no fitness in the file
                        # because the model was too big and so the fitness was not determined
                        fitnesses_generation.append(-1)
            fitnesses_all.append(fitnesses_generation)
        except:
            pass
    return fitnesses_all

def get_all_accuracies():
    pass

def get_accuracies_generation(generation):
    pass


###########################################
# Get all data
###########################################
fitnesses_all = get_all_fitnesses()

print(np.arange(1, len(fitnesses_all)))
print(fitnesses_all)

data = pd.read_csv("avocado.csv")
data = data.query("type == 'conventional' and region == 'Albany'")
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
data.sort_values("Date", inplace=True)

###########################################
# Dash app
###########################################
app = dash.Dash(__name__)
app.layout = html.Div(
    children=[
        html.H1(children=f"{ga_run}",),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": np.arange(1, len(fitnesses_all)),
                        "y": np.mean(fitnesses_all, axis=1),
                        "type": "lines",
                        "error_y": dict(type='data', array=np.std(fitnesses_all, axis=1), visible=True),
                    },
                ],
                "layout": {"title": "Average fitness over Generations"},
            },
        ),
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": data["Date"],
                        "y": data["Total Volume"],
                        "type": "lines",
                    },
                ],
                "layout": {"title": "Avocados Sold"},
            },
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)