# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import MSTL
import numpy as np
from datetime import datetime, timedelta, date

# Incorporate data
df = pd.read_csv('贺总.csv')

df['phoneno'] = df['phoneno'].astype(str)
users=df['phoneno'].unique()

df['index'] = df.groupby('date')['value'].transform('sum')
df['date']=pd.to_datetime(df["date"])
print('起始日期:',min(df['date']))
print('起始日期:',max(df['date']))

# Initialize the app
zuhe_options=df['pairs'].unique()
print(zuhe_options)
print(zuhe_options.sort())
app = Dash(__name__)


app.layout = html.Div(children=[
html.H2("历史自和系数"),    
    html.Div(children='说明-第一个版本涵盖了心率+心肝脾肺肾六大维度15个组合的同频程度维度的15个特征'),

    html.Hr(),
    dmc.Text("请选择用户", transform="capitalize"),    
    dmc.RadioGroup(
            [dmc.Radio(i, value=i) for i in  np.insert(users, 0, '全部', axis=0)],
            id='mingxi-phones-input',
            value='phones',
            size="sm"
        ),
    
    html.Hr(),

        dmc.DateRangePicker(
            id="date-range-picker",
            label="待计算的日期范围",
            description="You can also provide a description",
            minDate=min(df['date']),
            maxDate=max(df['date']),
            
            value=[min(df['date']), max(df['date'])],
            style={"width": 330},
        ),
        dmc.Space(h=10),
        dmc.Text(id="selected-date-date-range-picker"),

    html.Hr(),
    
    dmc.Text("请选择要查看的细分类型", transform="capitalize"),    
    # dmc.RadioGroup(
    #         [dmc.Radio(i, value=i) for i in  ['全部']],
    #         id='mingxi-quanbu-input',
    #         value='quanbu',
    #         size="sm"
    #     ),

    dmc.RadioGroup(
            [dmc.Radio(i, value=i) for i in np.insert(zuhe_options, 0, '全部', axis=0)],
            id='mingxi-zuhe-input',
            value='zuhe',
            size="sm"
        ),
    html.Hr(),
    
    dmc.Text("请选择分解算法", transform="capitalize"),    

    dmc.RadioGroup(
            [dmc.Radio(i, value=i) for i in  ['pusu','mtl']],
            id='mingxi-suanfa-input',
            value='suanfa',
            size="sm"
        ),
    html.Hr(),
    html.Div(children='历史走势'),

    dcc.Graph(figure={}, id='mingxi-graph'),
     html.Div(children='得分分布范围'),
   
    dcc.Graph(figure={}, id='mingxi-graph-hist'),
     html.Div(children='外部因素影响/得分波动趋势'),

    dcc.Graph(figure={}, id='mingxi-graph-trend'),
     html.Div(children='人体固有周期/得分波动周期'),
    
    dcc.Graph(figure={}, id='mingxi-graph-seasonal'),
     html.Div(children='人体内稳态/得分波动内在本质'),
    
    dcc.Graph(figure={}, id='mingxi-graph-resid'),

    # html.Div(children='所有数据记录'),

    
    
])
# Add controls to build the interaction
@callback(

    Output(component_id='mingxi-graph', component_property='figure'),
    Output(component_id='mingxi-graph-hist', component_property='figure'),
    
    Output(component_id='mingxi-graph-trend', component_property='figure'),
    Output(component_id='mingxi-graph-seasonal', component_property='figure'),
    Output(component_id='mingxi-graph-resid', component_property='figure'),

    Input(component_id='mingxi-phones-input', component_property='value'),
    # Input(component_id='mingxi-quanbu-input', component_property='value'),
    Input("date-range-picker", "value"),

    Input(component_id='mingxi-zuhe-input', component_property='value'),
    Input(component_id='mingxi-suanfa-input', component_property='value')
    

)
def update_graph(phones,daterange,zuhe,suanfa):
    print(phones,daterange,zuhe,suanfa)
    data=None
    if phones=='全部':
        print('使用所有用户数据')
        
        data=df
    elif phones in users:
        print(f'使用电话号码是{phones}用户的数据')
        
        data=df.loc[df.phoneno==phones]
    else:
        print('请选择用户的电话号码')
        
        data=df
    print('完成用户数据过滤')
    print('请选择日期范围')
    
    if len(daterange)==2:

        s_date=daterange[0]
        e_date=daterange[1]
        s_date=pd.to_datetime(s_date)
        e_date=pd.to_datetime(e_date)

        print(f'检测到起止日期为:{type(s_date)}{s_date},{e_date}')
        if e_date <min(data['date']):
            print('数据中的日期均晚于您选择的结束日期，重新选择靠后的结束日期')
        elif s_date>max(data['date']):
            print('数据中的日期均早于您选择的开始日期，重新选择靠前的开始日期')
        else:
            if s_date<min(data['date']):
                s_date=min(data['date'])
            if e_date>max(data['date']):
                e_date=max(data['date'])
            print(f'根据起止日期过滤数据:{s_date},{e_date}')
            # df['date']=pd.to_datetime(df["date"])
            print(type(data['date']))
            print(data.head(5))
            data = data.query('date >= @s_date and date <= @e_date')

            # data = data[(data["date"] >s_date & data["date"] < e_date)]
            print(f'完成日期范围数据过滤:{len(data)}')
    else:
        print('日期范围不对劲')

    print(f'完成用户数据日期范围过滤:{len(data)}')
    print(data.head(5))
    new=None
    fig=None
    fig_trend=None
    fig_seasonal=None
    fig_resid=None
    if zuhe =='全部':
        print('选择显示总和')
        new=data
        fig = px.line(new, x='date', y='index')
        print('完成用户数据总体同频程度的曲线图绘制')
        fig_hist = px.histogram(new, x='value', histnorm='percent')

        result=None
        if suanfa=='pusu':
            result = seasonal_decompose(new['value'], model='additive',period=7)
            if result:
                print(result.trend)
                print(result.seasonal)
                print(result.resid)
                # print(result.observed)

                fig_trend = px.line(result.trend)
                fig_seasonal = px.line(result.seasonal)
                fig_resid = px.line(result.resid)

            return fig,fig_hist,fig_trend,fig_seasonal,fig_resid
        elif suanfa=='mtl':
            stl_kwargs = {"seasonal_deg": 0} 
            model = MSTL(new['value'], periods=(24), stl_kwargs=stl_kwargs)
            result = model.fit()
            if result:
                print(result.trend)
                print(result.seasonal)
                print(result.resid)
                # print(result.observed)

                fig_trend = px.line(result.trend)
                fig_seasonal = px.line(result.seasonal)
                fig_resid = px.line(result.resid)

            return fig,fig_hist,fig_trend,fig_seasonal,fig_resid
        else:
            print('请选择算法')
        print('完成用户数据日期范围的趋势图分解绘制')                
    elif zuhe in zuhe_options:
        print('wait to choose')
        new=data.loc[data.pairs==zuhe]
        fig = px.line(data.loc[data.pairs==zuhe], x='date', y='value') 
        print('完成用户数据两两之间特定日期范围的曲线图绘制')
        fig_hist = px.histogram(new, x='value', histnorm='percent')

        result=None
        if suanfa=='pusu':
            result = seasonal_decompose(new['value'], model='additive',period=7)
            if result:
                print(result.trend)
                print(result.seasonal)
                print(result.resid)
                # print(result.observed)

                fig_trend = px.line(result.trend)
                fig_seasonal = px.line(result.seasonal)
                fig_resid = px.line(result.resid)

            return fig,fig_hist,fig_trend,fig_seasonal,fig_resid
        elif suanfa=='mtl':
            stl_kwargs = {"seasonal_deg": 0} 
            model = MSTL(new['value'], periods=(24), stl_kwargs=stl_kwargs)
            result = model.fit()
            if result:
                print(result.trend)
                print(result.seasonal)
                print(result.resid)
                # print(result.observed)

                fig_trend = px.line(result.trend)
                fig_seasonal = px.line(result.seasonal)
                fig_resid = px.line(result.resid)

            return fig,fig_hist,fig_trend,fig_seasonal,fig_resid
        else:
            print('请选择算法')
        print('完成用户数据日期范围的趋势图分解绘制')        
          
    else:
        print(f'选择显示组合项:{zuhe}')



# Run the app
if __name__ == '__main__':
    app.title = "用户阴阳自和趋势分析"
    
    app.run(debug=True)
