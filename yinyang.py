import pandas as pd
import plotly.express as px
import numpy as np

data=pd.read_csv('贺总.csv')

xiangsheng=[
    '肝脏得分-心脏得分',
    '肾脏得分-肝脏得分',
    '脾脏得分-心脏得分',
    '肺脏得分-肾脏得分',
    '肺脏得分-脾脏得分'

]

xiangke=[

    '肾脏得分-心脏得分',
    '脾脏得分-肝脏得分',
    '脾脏得分-肾脏得分',
    '肺脏得分-心脏得分',
    '肺脏得分-肝脏得分'

]
yin=[]
yang=[]
def judge_yinyang(key,value):
    if key in xiangsheng:
        if value < 0:
            value=abs(value)
            yin.append(value)
            yang.append(0)
        else:
            yang.append(value)
            yin.append(0)

    else:
        if value < 0:
            value=abs(value)

            yang.append(value)
            yin.append(0)

        else:
            yin.append(value)
            yang.append(0)

for row in data.itertuples():
    judge_yinyang(getattr(row, 'pairs'),getattr(row, 'value'))
data['yin']=yin
data['yang']=yang



yin_yang=data.groupby('date').agg(yin_sum=('yin','sum'),yang_sum=('yang','sum'))
yin_yang['yin-yang']=yin_yang['yang_sum'].sub(yin_yang['yin_sum']) 

print(type(yin_yang))
print(yin_yang.head(1))
print(yin_yang.describe(percentiles=[.25,.5,.75]))
yin_yang.to_csv('hezong-yinyang.csv')
fig = px.histogram(yin_yang, x="yin-yang")
fig.show()

px.line(yin_yang,x=yin_yang.index,y=['yin_sum', 'yang_sum','yin-yang'])




# px.line(x=data[data['pairs']=='肝脏得分-心脏得分']['date'],y=data[data['pairs']=='肝脏得分-心脏得分']['value'])
