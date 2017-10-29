import pandas as pd
import json
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib
import math
import sys

from collections import Counter
from sklearn.neighbors import LocalOutlierFactor
from helper import *

df = pd.read_csv('data/sample_data_example.csv')#{}'.format(sys.argv[1]))

amounts= []
dates = []
for _, transaction in df.iterrows():
    if isinstance(get_date(transaction['shipping_info']), str):
        srt2date = datetime.datetime.strptime(get_date(transaction['shipping_info']),  '%Y/%m/%d').date()
        dates.append(srt2date)
        amounts.append(round(get_amount(transaction['cart_info']), 2))

featuresdf = pd.DataFrame()
featuresdf['date'] = [d for d in dates]
featuresdf['value'] = amounts

def control_chart(feature1, feature2, k=3, show=False):
    std = np.std(feature1)
    m = np.mean(feature1)
    
    ctrl_stdup = m + (k*std)
    ctrl_stddown = m - (k*std)
    
    out_val=[]
    out_dates=[]
    
    pos_val= []
    pos_dates= []
    
    for i, point in enumerate(feature1):
        if point >= ctrl_stdup or point <= ctrl_stddown:
            out_val.append(point)
            out_dates.append(feature2[i])
        else:
            pos_val.append(point)
            pos_dates.append(feature2[i])
    
    
    score = (len(out_val)/len(feature1))*100
    if show:
        #show plot
        print('the total score of control chart is: ',score)
        if isinstance(out_dates[0], datetime.date):
            dates_list = feature2
            plot_dates = matplotlib.dates.date2num(dates_list)
            plt.plot_date(pos_dates, pos_val, c= 'b')
            plt.plot_date(out_dates, out_val, c='r')
        else:
            plt.plot(pos_dates, pos_val, c= 'b')
            plt.plot(out_dates, out_val, c='r')


        plt.plot((min(feature2), max(feature2)), (m, m), 'k-')
        plt.plot((min(feature2), max(feature2)), (ctrl_stdup, ctrl_stdup), 'k-', c='g')
        plt.plot((min(feature2), max(feature2)), (ctrl_stddown, ctrl_stddown), 'k-', c='g')
        plt.show()
        
    return score


def lof(feature1 , feature2, show=False, n_neighbors= 15):
    # Generate train data
    #dates = data['date'].as_matrix()
    if isinstance(feature2[0], datetime.date):
        dates_dict = {d:i for i,d in enumerate(set(np.sort(dates)))}

        date2int = [dates_dict[d] for d in dates]
    else:
        date2int = feature2
    X = list(zip(feature1, date2int))
    
    # Generate some abnormal novel observations
    # fit the model
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, algorithm='auto')
    y_pred = clf.fit_predict(X)
    
    out_val=[]
    out_dates=[]
    
    pos_val= []
    pos_dates= []
    for i, pred in enumerate(y_pred):
        if pred == -1:
            out_val.append(feature1[i])
            out_dates.append(feature2[i])
        else:
            pos_val.append(feature1[i])
            pos_dates.append(feature2[i])

    score = (len(out_val)/len(y_pred))*100
    
    if show:
        print('total score of Local Outlier Factor:',score)
        if isinstance(out_dates[0], datetime.date):
            plt.plot_date(pos_dates, pos_val, c= 'b')
            plt.plot_date(out_dates, out_val, c='r')
        else:
        	plt.plot(pos_dates, pos_val, c='b')
        	plt.plot(out_dates, out_val, c='r')
        plt.show()

    return score


def benfords(values, show=False):
    
    digits=[]
    for v in values:
        digits.append(int(str(v)[0]))
    
    d_count= Counter(digits)
    num= list(range(1,10))
    total_count=[d_count[i] for i in num]
    prc=[]
    for i in num:
        prc.append((d_count[i]/ len(digits))*len(digits))
        
    benford = [100*math.log10(1 + 1./i) for i in range(1,10)]
    
    error = abs(np.sum(np.array(benford) - np.array(prc)))

    score = (error/sum(benford))*100
    
    score = (1/(1+np.exp(-score)))*100
    
    index = np.arange(len(num))
    bar_width = 0.35
    if show:
        print('the total score of benfords is ', score)
        plt.bar(index, prc, bar_width, color='b', label='values')
        plt.bar(index + bar_width, benford, bar_width, color='r',label='benford')
        plt.xticks(index + bar_width / 2, (1,2,3,4,5,6,7,8,9))
        plt.show()
    
    return score


total_score = (benfords(featuresdf['value'], show=True) + lof(featuresdf['value'], featuresdf['date'], show=True) + control_chart(featuresdf['value'], featuresdf['date'], show=True))/3
print('The total fraud score is: ', total_score)