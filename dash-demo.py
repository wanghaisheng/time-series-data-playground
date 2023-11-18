# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_mantine_components as dmc
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import MSTL
import numpy as np
from datetime import datetime, timedelta, date
from scipy.stats import bootstrap
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




def average(data):
    return sum(data) / len(data)
def handbootstrap(data, B, c, func):
# ......
# 计算bootstrap置信区间
# :param data: array保存样本数据
# :param B:抽样次数通常B>=1000
# :param c:置信水平
# :param func:样本估计量
# :return:bootstrap置信区间上下限
# "..."
    array = np.array(data)
    n = len(array)
    sample_result_arr = []
    for i in range(B):
        index_arr = np.random.randint(0, n, size=n)
        data_sample = array[index_arr]
        sample_result = func(data_sample)
        sample_result_arr.append(sample_result)
    a = 1 - c
    k1 = int(B*a/2)
    k2 = int(B*(1-a/2))
    auc_sample_arr_sorted = sorted(sample_result_arr)
    lower= auc_sample_arr_sorted[k1]
    higher= auc_sample_arr_sorted[k2]
    return lower, higher


# https://library.virginia.edu/data/articles/bootstrap-estimates-of-confidence-intervals

def resample(data, seed):
    '''
    Creates a resample of the provided data that is the same length as the provided data
    '''
    import random
    random.seed(seed)
    res = random.choices(data, k=len(data))
    return res

# def next_bootstrap(data,cishu=10000,):
    
#     # Extract the distance, x, and velocity, y, values from our pandas dataframe 
#     distances = data["x"].values
#     velocities = data["y"].values

#     # Zip our distances and velocities together and store the zipped pairs as a list
#     dist_vel_pairs = list(zip(distances, velocities))

#     # Print out the first 5 zipped distance-velocity pairs
#     # print(dist_vel_pairs[:5])
#     # Generate 10,000 resamples with a list comprehension
#     boot_resamples = [resample(dist_vel_pairs, val) for val in range(cishu)]

#     # Calculate beta from linear regression for each of the 10,000 resamples and store them in a list called "betas"
#     betas = []

#     for res in boot_resamples:
#         # "Unzip" the resampled pairs to separate x and y so we can use them in the LinearRegression() function
#         dist_unzipped, vel_unzipped = zip(*res)
#         dist_unzipped = np.array(dist_unzipped).reshape((-1, 1))
        
#         # Find linear coefficient beta for this resample and append it to a list of betas
#         betas.append(LinearRegression(fit_intercept=False).fit(dist_unzipped, vel_unzipped).coef_[0])

#     # Print out the first 5 beta values 
#     # print(betas[:5])

#     # Calculate the values of 2.5th and 97.5th percentiles of our distribution of betas
#     conf_interval = np.percentile(betas, [2.5,97.5])
#     # print(conf_interval)
#     return conf_interval
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
    # print(data.head(5))
    new=None
    fig=None
    fig_trend=None
    fig_seasonal=None
    fig_resid=None
    rng = np.random.default_rng()
    
    if zuhe =='全部':
        print('选择显示总和')
        new=data
        fig = px.line(new, x='date', y='index')
        print('完成用户数据总体同频程度的曲线图绘制')
        
        res = bootstrap((new['index'],), np.std, confidence_level=0.95,
                        random_state=rng)   
        # descri=
        print(new['index'].describe(percentiles=[.05,.1,.25, .5, .75,.9,.95]))
        xbar = new['index'].mean()
        xstd = new['index'].std()
        print('标准差法异常值上限检测:\n',xbar + 2 * xstd,any(new['index'] > xbar + 2 * xstd))
        print('标准差法异常值下限检测:\n',xbar - 2 * xstd,any(new['index'] < xbar - 2 * xstd))

        #异常值 箱线图法
        Q1 = new['index'].quantile(q = 0.25)
        Q3 = new['index'].quantile(q = 0.75)
        IQR = Q3 -Q1
        print('箱线图法异常值上限检测:\n', Q3 + 1.5*IQR,any(new['index'] > Q3 + 1.5*IQR))
        print('箱线图法异常值下限检测:\n',Q1 - 1.5*IQR,any(new['index'] < Q1 - 1.5*IQR))

        
        
        # https://www.cnblogs.com/tinglele527/p/11955103.html
        # 异常值的判定方法：

        # 1.n个标准差法

        # 2.箱线图法

        # 标准差法，就是用以样本均值+样本标准差为基准，如果样本离平均值相差2个标准差以上的就是异常值

        # 箱线图法：以上下四分位作为参考， x > Q3+nIQR 或者 x < Q1 - nIQR 简单地理解，就是如果样本值不在上下四分位+标准差范围内，就是异常值        
        print(f'假设正态分布，整体自和系数的置信区间为:{res.confidence_interval}')     
        
        
        
        confidence_intervals = new['index'].quantile(q=[0.025, 0.975])
        print(confidence_intervals)
# https://www.analyticsvidhya.com/blog/2022/01/understanding-confidence-intervals-with-python/
        import scikits.bootstrap as boot
        conf_interval=boot.ci(new['index'], np.average)
        print(f'假设非正态分布，scikits整体自和系数的置信区间为:{conf_interval}')     

        
        conf_interval=handbootstrap(new['index'],10000,0.95,average)
        print(f'假设非正态分布，handbootstrap整体自和系数的置信区间为:{conf_interval}')     
        
 
        fig_hist = px.histogram(new, x='index', histnorm='percent')
        result=None
        if suanfa=='pusu':
            result = seasonal_decompose(new['index'], model='additive',period=7)
            if result:
                print(result.trend)
                for index,value in result.trend.itertuples():
                    if value >conf_interval[1]:
                        print(index)
                print(result.seasonal)
                print(result.resid)
                # print(result.observed)

                fig_trend = px.line(result.trend)
                fig_seasonal = px.line(result.seasonal)
                fig_resid = px.line(result.resid)

            return fig,fig_hist,fig_trend,fig_seasonal,fig_resid
        elif suanfa=='mtl':
            stl_kwargs = {"seasonal_deg": 0} 
            model = MSTL(new['index'], periods=(24), stl_kwargs=stl_kwargs)
            result = model.fit()
            if result:
                # print(result.trend)
                # print(result.seasonal)
                # print(result.resid)
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
        print(f'完成用户数据{zuhe}两两之间特定日期范围的曲线图绘制')
        print(new['value'].describe(percentiles=[.05,.1,.25, .5, .75,.9,.95]))
        xbar = new['value'].mean()
        xstd = new['value'].std()
        print('标准差法异常值上限检测:\n',xbar + 2 * xstd,any(new['value'] > xbar + 2 * xstd))
        print('标准差法异常值下限检测:\n',xbar - 2 * xstd,any(new['value'] < xbar - 2 * xstd))

        #异常值 箱线图法
        Q1 = new['value'].quantile(q = 0.25)
        Q3 = new['value'].quantile(q = 0.75)
        IQR = Q3 -Q1
        print('箱线图法异常值上限检测:\n', Q3 + 1.5*IQR,any(new['value'] > Q3 + 1.5*IQR))
        print('箱线图法异常值下限检测:\n',Q1 - 1.5*IQR,any(new['value'] < Q1 - 1.5*IQR))
        res = bootstrap((new['value'],), np.std, confidence_level=0.9,
                        random_state=rng)   
        print(f'假设正态分布，bootstrap 组合{zuhe}两两之间自和系数的置信区间为:{res.confidence_interval}')     

        
        confidence_intervals = new['value'].quantile(q=[0.025, 0.975])
        print(f'假设正态分布，quantile 组合{zuhe}两两之间自和系数的置信区间为:{res.confidence_interval}')     

        conf_interval=handbootstrap(new['value'],10000,0.9,average)
        print(f'假设非正态分布，handbootstrap 组合{zuhe}两两之间自和系数的置信区间为:{conf_interval}')     
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
                # print(result.trend)
                # print(result.seasonal)
                # print(result.resid)
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
# https://haochendaily.com/anomaly-detection-process.html
