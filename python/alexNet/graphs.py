import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import time

# Read data from the training_log.csv file
df = pd.read_csv('./logs/training_log.csv')
df['Time (s)'] = df['Time (s)'].cumsum()  # Cumulative time for x-axis

# Initialize the Dash app
app = dash.Dash(__name__)

# Set up the layout
app.layout = html.Div(style={'backgroundColor': '#2e2e2e', 'height': '100vh'},
                      children=[
                          html.H1(children='Live Training Log', style={'textAlign': 'center', 'color': '#ffffff'}),
                          dcc.Graph(id='live-update-graph', config={'displayModeBar': False}),
                          dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
                      ])

# Callbacks to update the graph
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph(n):
    if n >= len(df):
        raise dash.exceptions.PreventUpdate

    # Get the latest data
    latest_data = df.iloc[:n+1]

    # Create traces
    training_loss = go.Scatter(
        x=latest_data['Time (s)'],
        y=latest_data['Loss'],
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='orange'),
    )

    training_accuracy = go.Scatter(
        x=latest_data['Time (s)'],
        y=latest_data['Training Accuracy'],
        mode='lines+markers',
        name='Training Accuracy',
        line=dict(color='green'),
    )

    # Create the layout
    layout = go.Layout(
        title='Training Performance',
        plot_bgcolor='#2e2e2e',
        paper_bgcolor='#2e2e2e',
        font=dict(color='white'),
        xaxis=dict(title='Time (s)', titlefont=dict(color='white')),
        yaxis=dict(title='Value', titlefont=dict(color='white')),
    )

    return {'data': [training_loss, training_accuracy], 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
