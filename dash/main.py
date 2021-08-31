import dash

from dash.dependencies import Input, Output
import dash_table 

import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader import data as web
import dash_daq as daq
from datetime import datetime as dt
from theme import theme, rootLayout


app= dash.Dash()
app.layout = html.Div( [
    dcc.Dropdown(
        id='option',
        options=[
            {'label':'Google', 'value':'GOOGL'},
            {'label':'Apple', 'value':'AAPL'},
            {'label':'Microsoft', 'value':'MSFT'},
            {'label':'Tesla', 'value':'TSLA'},
            {'label':'Micron', 'value':'Micron'},
        ],
        value='GOOGL'
    ),
    html.Br(),
    dcc.Graph(id='ploting'),
    html.Br(),
    dash_table.DataTable(
        id = 'datatable',
        columns=[
                {"name":'Date' , "id": 'Date'},
                {"name":'High' , "id": 'High'},
                {"name":'Low' , "id": 'Low'},
                {"name":'Volume' , "id": 'Volume'},
                {"name":'Adj Close' , "id": 'Adj Close'},
                {"name":'Close' , "id": 'Close'}],
                page_current=0,
        page_size=5,
        page_action='custom'       
    )]
    ,style={'width':'100%'})


@app.callback(Output('ploting', 'figure'), [Input('option', 'value')])
def graph(selected_dropdown_value):
    df  = web. DataReader(selected_dropdown_value, 'yahoo',dt(2020,1,1), dt.now())
    return{
        'data':[{'x':df.index, 
        'y':df.Close}],
        'layout':{'margin':{'l':40,'r':0,'t':40,'b':30}}
    }


@app.callback(
                Output('datatable', 'data'), 
                Input('option', 'value'),
                Input('datatable','page_current'),
                Input('datatable','page_size'))
def table(selected_dropdown_value, page_current, page_size):
    df  = web. DataReader(selected_dropdown_value, 'yahoo',dt(2020,1,1), dt.now())
    import pandas as pd
    df=df.reset_index()
    df['Date']=pd.to_datetime(df['Date'])
    return df.iloc[
        page_current*page_size:(page_current+ 1)*page_size
    ].to_dict('records')
if __name__=='__main__':
    app.run_server(debug=True)