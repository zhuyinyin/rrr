# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:23:47 2020

@author: Z20080498
"""
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

 # mse均方誤差 # 準確率 袋外分數
from sklearn.metrics import mean_squared_error,accuracy_score, roc_auc_score 



df =  pd.read_csv(r'D:\spyder_project\Data_analysis\训练数据集与特征数据集\train_15.csv')
feature = pd.read_csv(r'D:\spyder_project\Data_analysis\训练数据集与特征数据集\feature_16.csv')


def my_holiday(datetime):
    
    # print(datetime['timestamp'].strftime('%Y-%m-%d'))
    tuples = ('20150101', '20150102', '20150103',
              '20150218', '20150219', '20150220', '20150221', '20150222', '20150223', '20150224',
              '20150404', '20150405', '20150406',
              '20150501', '20150502', '20150503',
              '20150620', '20150621', '20150622',
              '20150903', '20150904', '20150905', '20150926', '20150927',
              '20151001', '20151002', '20151003', '20151004', '20151005', '20151006', '20151007',
              '20160101', '20160102', '20160103',
              '20160207', '20160208', '20160209', '20160210', '20160211', '20160212', '20160213',
              '20160402', '20160403', '20160404', '20160430',
              '20160501', '20160502',
              '20160609', '20160610', '20160611',
              '20160915', '20160916', '20160917',
              '20161001', '20161002', '20161003', '20161004', '20161005', '20161006', '20161007',
              '20161231')
    if datetime['timestamp'].strftime('%Y%m%d') in tuples:
        return 1
    else:
        return 0

def my_hour_week_or_holiday(df):
    """
    周末假期和時間對數量的影響

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    if df['my_weekend'] or df['my_holiday']:
        if df['hour'] in (11, 12, 13, 14, 15, 16):
            return 1
        else:
            return 0
    else:
        return 0


def my_hour_work(df):
    
    if df['my_weekend'] or df['my_holiday']:
        
        return 0
    else:
        if df['hour'] in (7,8,9,17, 18):
            return 1
        else:
            return 0


def FeatureEngineering(df):
    """
    特征工程
    return df
    """
    # 從前往後填充
    df = df.fillna(method='ffill')
    
    df['timestamp'] = df['timestamp'].apply(lambda x: pd.to_datetime(x))
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['woy'] = df['timestamp'].dt.weekofyear

    # 每週第六天和第七天為週末weekend
    df['my_weekend'] = df[['dow']].apply(lambda x: 1 if x['dow'] in (5, 6) else 0   , axis = 1)
        
    # 是否為假日
    df['my_holiday'] = df[['timestamp']].apply(my_holiday, axis = 1)
    
    df['my_hour_week_or_holiday'] = df[['hour', 'my_weekend', 'my_holiday']].apply(my_hour_week_or_holiday, axis = 1)
    
    df['my_hour_work'] = df[['hour', 'my_weekend', 'my_holiday']].apply(my_hour_work, axis = 1)
    
    return df

df = FeatureEngineering(df)
feature = FeatureEngineering(feature)


col = ['weather', 'temp1', 'temp2', 'wind speed',
        'season', 
        'weekend', 
        'holiday',
        'humidity ',
        'month', 'hour', 'dow', 'woy',
        'my_weekend',
        
        # 'my_holiday', 
        # 'my_hour_week_or_holiday', 
        # 'my_hour_work'
        ]

x = df[col]
y = df['count ']

# *************************************************數據集和測試集劃分*****************
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=0)




# **************************************************隨機森鈴**************************
# 1080 +8  1160 +10  1180+10 1200 +10
# n_estimators=30,max_features=8,max_depth=13, min_samples_split=5
params = {'n_estimators': 1170,   # 分類器個數
            # 'max_depth': 13,  # 决策树最大深度max_depth
            'random_state': 0,  # 隨機種子,便於調參
            "max_features": 8,  # 最大特征數
            "oob_score" :True,
            # 'min_samples_split' : 2,  # 内部节点再划分所需最小样本数
            'n_jobs': -1  # 并行數量默認1  最大-1
            }

lgs = RandomForestRegressor(**params)
my_model = lgs.fit(x_train, y_train)
print('袋外分數:', lgs.oob_score_)
# 测试值
y_model = my_model.predict(x_test)



# ***********************************************網格搜索****************************
param_grid = [
{
    # 200
    # 'n_estimators': range(1100, 1220, 20),
    'max_features': [8, 9, 10,11],
 # 'max_leaf_nodes':range(4, 9,1)
#  'max_depth':range(3,14,2),
#  'min_samples_split':range(2,10,1)
},
# {
#     'bootstrap': [False],
#  'n_estimators': [3, 10, 30],
#   'max_features': [2, 3, 4]
# }
]
# 12
# n_estimators=30,max_features=8,max_depth=13, min_samples_split=5,n_estimators=1080
forest_reg = RandomForestRegressor(oob_score=True,n_estimators=1200)
# 網格搜索模型
grid_search = GridSearchCV(forest_reg, param_grid, cv=7)


# grid_search.fit(x_train, y_train)
# print('最優參數:', grid_search.best_params_)
# # 網格搜索最優參數模型
# grid= grid_search.best_estimator_
# my_model = grid.fit(x_train, y_train)
# print('袋外分數:', grid.oob_score_)
# # 測試值
# y_model = my_model.predict(x_test)

# ***********************************************網格搜索結束************************




# print('測試值:', y_model)
# 真實值
# print('真實值:', np.array(y_test))
plt.figure(facecolor='w')
t = np.arange(len(x_test))
plt.plot(t, y_test, 'r-', linewidth=2, label=u'real')
plt.plot(t, y_model, 'g-', linewidth=2, label=u'predict')
plt.legend(loc='upper right')
# plt.title(u'线性回归预测销量', fontsize=18)
plt.grid(b=True)
# plt.show()

# ************************************************模型評估*************************
mse = mean_squared_error(np.array(y_test), y_model)
# acc = accuracy_score(y_test, y_model)


print('均方誤差 mse:', mse)
# print('正確率:', acc)



# ***********************************************輸出csv******************************
# print(feature)
result = my_model.predict(feature[col])
r = pd.DataFrame(result, columns=['Count'])

r['id'] = np.arange(8716,17416)
re = r[['id', 'Count']]
re.to_csv(r'D:\spyder_project\Data_analysis\predict_16.csv', index=False)






















