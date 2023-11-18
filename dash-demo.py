# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
# pip install dash-mantine-components

# Incorporate data
df = pd.read_csv('贺总.csv')
# print(df.loc[df.pairs=='心脏得分-心率'])
# Initialize the app
zuhe=df['pairs'].unique()
app = Dash(__name__)



app.layout = html.Div(children=[

    html.Div(children='历史脏腑同频系数分布'),
    html.Hr(),
    dmc.RadioGroup(
            [dmc.Radio(i, value=i) for i in  zuhe],
            id='mingxi-graph-input',
            value='value',
            size="sm"
        ),
    dcc.Graph(figure={}, id='mingxi-graph'),


    html.Div(children='所有数据记录'),

    dash_table.DataTable(data=df.to_dict('records'), page_size=6),

    
    
])
# Add controls to build the interaction
@callback(

    Output(component_id='mingxi-graph', component_property='figure'),
    Input(component_id='mingxi-graph-input', component_property='value')
)
def update_graph(col_chosen):
    # df=df.loc[df.pairs==col_chosen]
    

    # fig = px.line(df.loc[df.pairs==col_chosen], x='date', y='value')

    fig = px.histogram(df.loc[df.pairs==col_chosen], x='value')
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
