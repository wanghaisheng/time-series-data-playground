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

app = Dash(__name__)
# 把顺序排列的一组数据分割为若干相等部分的分割点的数值即为相应的分位数（quantile）。中位数是分位数中最简单的一种，它将数据等分成两分。由于四分位数（quartile）则是将数据按照大小顺序排序后，把数据分割成四等分的三个分割点上的数值。对原始数据，四分位数的位置一般为，，。如果四分位数的位置不是整数，则四分位数等于前后两个数的加权平均。


# 十分位数（deciles）是将数据按照大小顺序排序后，把数据分割成十等分的九个分割点上的数值；百分位数（percentile）是将数据按照大小顺序排序后，把数据分割成一百等分的九十九个分割点上的数值。

df=None

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
    dmc.Text("请选择数据集", transform="capitalize"),    
    dmc.RadioGroup(
            [dmc.Radio(i, value=i) for i in  ['高原','国庆','贺总']],
            id='mingxi-dataset-input',
            value='dataset',
            size="sm"
        ), 
    
    
    
    html.Div(children={}, id="first_output_3"),
    # dmc.Table(children={}, id="first_output_3"),


    
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
    
    html.Div(children={}, id="statics-describe"),
    
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


@callback(
    Output("first_output_3", "children"),
    Input("mingxi-dataset-input",  component_property='value'))
def first_callback(dataset):
    if dataset=='dataset':
        dataset='贺总'
    print(f'选择的数据集是{dataset}')
    df = pd.read_csv(dataset+'.csv')

    df['phoneno'] = df['phoneno'].astype(str)
    datasetusers=df['phoneno'].unique()


    df['index'] = df.groupby('date')['value'].transform('sum')
    df['date']=pd.to_datetime(df["date"])
    # print('起始日期:',min(df['date']))
    # print('起始日期:',max(df['date']))

    # Initialize the app
    zuhe_options=df['pairs'].unique()
    # print(zuhe_options)
    # print(zuhe_options.sort())
    print(f'初始化所有查询条件{dataset}')
    
    content=html.Div(children=[

    dmc.Text("请选择用户", transform="capitalize"),    
    dmc.RadioGroup(
            [dmc.Radio(i, value=i) for i in  np.insert(datasetusers, 0, '全部', axis=0)],
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

    dmc.RadioGroup(
            [dmc.Radio(i, value=i) for i in np.insert(zuhe_options, 0, '全部', axis=0)],
            id='mingxi-zuhe-input',
            value='zuhe',
            size="sm"
        ),
    html.Hr(),])
    print(f'数据条数:{len(df)}')
    return content



# Add controls to build the interaction
@callback(

    Output(component_id='mingxi-graph', component_property='figure'),
    Output(component_id="statics-describe",component_property= "children"),
    
    Output(component_id='mingxi-graph-hist', component_property='figure'),
    
    Output(component_id='mingxi-graph-trend', component_property='figure'),
    Output(component_id='mingxi-graph-seasonal', component_property='figure'),
    Output(component_id='mingxi-graph-resid', component_property='figure'),
    Input(component_id='mingxi-dataset-input', component_property='value'),

    Input(component_id='mingxi-phones-input', component_property='value'),
    # Input(component_id='mingxi-quanbu-input', component_property='value'),
    Input("date-range-picker", "value"),

    Input(component_id='mingxi-zuhe-input', component_property='value'),
    Input(component_id='mingxi-suanfa-input', component_property='value')
    

)
def update_graph(dataset,phones,daterange,zuhe,suanfa):
    print('绘制条件为',dataset,phones,daterange,zuhe,suanfa)
    if suanfa=='suanfa':
        print('wait choose all query param')
    else:
        print('all query param selected,start to draw')
        if dataset=='dataset':
            dataset='贺总'
        print(f'选择的数据集是{dataset}')
        df = pd.read_csv(dataset+'.csv')
        df['phoneno'] = df['phoneno'].astype(str)
        datasetusers=df['phoneno'].unique()


        df['index'] = df.groupby('date')['value'].transform('sum')
        df['date']=pd.to_datetime(df["date"])
        # print('起始日期:',min(df['date']))
        # print('起始日期:',max(df['date']))

        # Initialize the app
        zuhe_options=df['pairs'].unique()    
        if df is None:
            print('请选择其他存在的数据集')
        print(f'根据绘制条件，加载的数据条数为:{len(df)}')
        data=None
        print('核对已选择的用户范围')
        
        if phones=='全部':
            print('使用所有用户数据')
            
            data=df
        elif phones in datasetusers:
            print(f'使用电话号码是{phones}用户的数据')
            
            data=df.loc[df.phoneno==phones]
        else:
            print('请选择用户的电话号码')
            
            data=df
        print('完成用户数据过滤')
        print('核对已选择日期范围')
        
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
        print('核对已选择组合项')
        if zuhe=='zuhe':
            zuhe="全部"
        if zuhe =='全部':
            print('选择显示总和')
            new=data
            fig = px.line(new, x='date', y=['index','value'])
            print('完成用户数据总体同频程度的曲线图绘制')
            
            res = bootstrap((new['index'],), np.std, confidence_level=0.95,
                            random_state=rng)   
            describe=new['index'].describe(percentiles=[.05,.1,.15,.25, .5, .75,.85,.9,.95])
            print('原始同频程度统计特征statics',describe)
            # https://github.com/pandas-dev/pandas/pull/27228
            print(type(describe))
            
            # print('html format',describe.to_frame().to_html())
            # print('string format',describe.to_frame().to_string())
            # descri=describe.to_frame().to_string()

            header = [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("指标"),
                            html.Th("原始值"),
                            html.Th("去除瞬时影响后的值"),                           
                        ]
                    )
                )
            ]
            prev_rows=[]
            for indx,val in describe.items():
                row=[indx,val]
                # row = html.Tr([html.Td(indx), html.Td(val)])
                prev_rows.append(row)

            # body = [html.Tbody(rows)]

            # descri=dmc.Table(header + body)





            
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
                    # print(result.trend)
                    # for index,value in result.trend.itertuples():
                    #     if value >conf_interval[1]:
                    #         print(index)
                    # print(result.seasonal)
                    # print(result.resid)
                    # print(result.observed)

                    fig_trend = px.line(result.trend)
                    describe=result.trend.describe(percentiles=[.05,.1,.15,.25, .5, .75,.85,.9,.95])
                    print('用户全部数据同频程度普通算法分解成趋势信号的统计特征statics',describe)    
                    fenjie=[]
                    print('before result',prev_rows)
                    for indx,val in describe.items():
                        fenjie.append(val)
                    biaoge=list(zip(prev_rows,fenjie))
                    rows=[]
                    for i in biaoge:

                        row = html.Tr([ html.Td(i[0][0]),html.Td(i[0][1]),html.Td(i[1])])
                        rows.append(row)

                    body = [html.Tbody(rows)]

                    descri=dmc.Table(header + body)
                    print('表格拼接完成')

                    
                                    
                    fig_seasonal = px.line(result.seasonal)
                    fig_resid = px.line(result.resid)
                    print('用户全部数据的朴素方法分解完成')
                return fig,descri,fig_hist,fig_trend,fig_seasonal,fig_resid
            elif suanfa=='mtl':
                stl_kwargs = {"seasonal_deg": 0} 
                model = MSTL(new['index'], periods=(24), stl_kwargs=stl_kwargs)
                result = model.fit()
                if result:
                    # print(result.trend)
                    # print(result.seasonal)
                    # print(result.resid)
                    # print(result.observed)
                    describe=result.trend.describe(percentiles=[.05,.1,.15,.25, .5, .75,.85,.9,.95])
                    print('用户全部数据同频程度MSTL分解成趋势信号的统计特征statics',describe)       
                    fenjie=[]
                    print('before result',prev_rows)
                    for indx,val in describe.items():
                        fenjie.append(val)
                    biaoge=list(zip(prev_rows,fenjie))
                    rows=[]
                    for i in biaoge:

                        row = html.Tr([ html.Td(i[0][0]),html.Td(i[0][1]),html.Td(i[1])])
                        rows.append(row)
                    print(rows)

                    body = [html.Tbody(rows)]

                    descri=dmc.Table(header + body)
                    fig_trend = px.line(result.trend)
                    fig_seasonal = px.line(result.seasonal)
                    fig_resid = px.line(result.resid)
                    print('用户全部数据的mstl方法分解完成')

                return fig,descri,fig_hist,fig_trend,fig_seasonal,fig_resid
            else:
                print('请选择算法')
            print('完成用户数据日期范围的趋势图分解绘制')                
        elif zuhe in zuhe_options:
            print('wait to choose')
            new=data.loc[data.pairs==zuhe]
            fig = px.line(data.loc[data.pairs==zuhe], x='date', y='value') 
            print(f'完成用户数据{zuhe}两两之间特定日期范围的曲线图绘制')
            describe=new['value'].describe(percentiles=[.05,.1,.15,.25, .5, .75,.85,.9,.95])
            print('statics',describe)
            # https://github.com/pandas-dev/pandas/pull/27228
            print(type(describe))
            
            # print('html format',describe.to_frame().to_html())
            # print('string format',describe.to_frame().to_string())
            # descri=describe.to_frame().to_string()

            header = [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("指标"),
                            html.Th("绝对值"),
                            html.Th("去除瞬时影响后的值"),                           
                        ]
                    )
                )
            ]
            prev_rows=[]
            for indx,val in describe.items():
                row=[indx,val]
                # row = html.Tr([html.Td(indx), html.Td(val)])
                prev_rows.append(row)

            # body = [html.Tbody(rows)]

            # descri=dmc.Table(header + body)



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
                    # print(result.trend)
                    # print(result.seasonal)
                    # print(result.resid)
                    # print(result.observed)
                    describe=result.trend.describe(percentiles=[.05,.1,.15,.25, .5, .75,.85,.9,.95])
                    print('同频程度普通算法分解成趋势信号的统计特征statics',describe)      
                    fenjie=[]
                    print('before result',prev_rows)
                    for indx,val in describe.items():
                        fenjie.append(val)
                    biaoge=list(zip(prev_rows,fenjie))
                    rows=[]
                    for i in biaoge:

                        row = html.Tr([ html.Td(i[0][0]),html.Td(i[0][1]),html.Td(i[1])])
                        rows.append(row)
                    print(rows)

                    body = [html.Tbody(rows)]

                    descri=dmc.Table(header + body)
                    fig_trend = px.line(result.trend)
                    fig_seasonal = px.line(result.seasonal)
                    fig_resid = px.line(result.resid)
                    print(f'用户两两{zuhe}数据的朴素方法分解完成')

                return fig,descri,fig_hist,fig_trend,fig_seasonal,fig_resid
            elif suanfa=='mtl':
                stl_kwargs = {"seasonal_deg": 0} 
                model = MSTL(new['value'], periods=(24), stl_kwargs=stl_kwargs)
                result = model.fit()
                if result:
                    # print(result.trend)
                    # print(result.seasonal)
                    # print(result.resid)
                    # print(result.observed)
                    describe=result.trend.describe(percentiles=[.05,.1,.15,.25, .5, .75,.85,.9,.95])
                    print('同频程度MSTL分解成趋势信号的统计特征statics',describe)  
                    fenjie=[]
                    print('before result',prev_rows)
                    for indx,val in describe.items():
                        fenjie.append(val)
                    biaoge=list(zip(prev_rows,fenjie))
                    rows=[]
                    for i in biaoge:

                        row = html.Tr([ html.Td(i[0][0]),html.Td(i[0][1]),html.Td(i[1])])
                        rows.append(row)
                    print(rows)

                    body = [html.Tbody(rows)]

                    descri=dmc.Table(header + body)
                    fig_trend = px.line(result.trend)
                    fig_seasonal = px.line(result.seasonal)
                    fig_resid = px.line(result.resid)
                    print(f'用户两两{zuhe}数据的mstl方法分解完成')

                return fig,descri,fig_hist,fig_trend,fig_seasonal,fig_resid
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
