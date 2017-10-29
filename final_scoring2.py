import pandas as pd
import warnings
import numpy as np
import time
import math
import re
import sys
import json
import datetime
from datetime import date
# Ignore all warnings
warnings.filterwarnings("ignore")
pd.options.display.max_info_rows = 200
pd.options.display.max_columns = 40

start = time.time()
try: df = pd.read_csv(sys.argv[1])
except: df = pd.read_csv(sys.argv[1], '|')

print("Time to import csv to pandas : "+str((time.time() - start)/60)+ " minutes or "+str(time.time() - start)+" seconds")

pd.options.display.max_colwidth = 200

# Weights dictionary - To be updated by Noor
cols_weights_ep = {
                "browserlanguage": 0.5,
                "useragent": 1,
                "deviceid": 1,
                "device_type": 1,
                "browserplatform": 1,
                "browserparent": 1,
                "browsername": 1,
                "device_pointing_method": 0.5,
                "city": 1.5,
                "country": 2,
                "region": 0.5,
                "ip": 0.5
               }

cols_weights_ship = {
                "shippingcountry": 6,
                "shippingzipcode": 4,
                "shippingstreet": 3,
                "shippingstate": 2,
                "shippingphonenumber": 1,
                "shippingnamefirst": 2,
                "shippingnamelast": 1,
                "billingzipcode": 2
                }


event_trigger_signals = ["account_create_velocity", 
                         "account_testing",
                         "event_velocity",
                         "geo_anonymous",
                         "input_anomaly",
                         "input_scripted",
                         "login_accounts",
                         "login_failure",
                         "login_velocity",
                         "net_anomaly_ip",
                         "net_anomaly_ua",
                         "shiptobill_distance"]

# Increase ratio - To be updated by Noor
inc_ratio = 3
increase_weight = 1/inc_ratio

# Taking EP and ship model
ep_cols = ['accountid','browserlanguage','useragent','deviceid','device_type', 'browserplatform','browserparent', 'browsername', 'device_pointing_method','city','country','region','ip']
ship_cols = ["shippingcountry", "shippingzipcode", "shippingstreet", "shippingstate", "shippingphonenumber", "shippingnamefirst","shippingnamelast","billingzipcode"]

# Adding actiondata for shipping model
short_df = df[ep_cols+['eventtriggeredsignals','cart_info','actiondata', 'requesttime', 'unixtime']]


# This function takes in a single row and takes out all shipping params from actiondata
# Returns one param at a time
def get_data( row, colname ):
    if(str(row['actiondata'])=='-1'):
        return ''
    else:
        t = row['actiondata']
        m = re.search(colname+'":"(.*?)"',t)
        if m:
            return m.group(1)
        else:
            return ''

# This function adds the shipping cols to short_df
def add_shipping_cols(df):
    for colname in ship_cols:
        df[colname] = df.apply(lambda row: get_data(row, colname), axis=1)
    return df

# Adding shipping cols
short_df = add_shipping_cols(short_df)   # SHIPPING

# Creating columns for scores
# short_df['numer_of_transaction'] = 'NA'

short_df['date'] = 'NA'
short_df['total_Cart_amount'] = 'NA'
short_df['eventtriggered_score'] = 'NA'
short_df['score_2_cart'] = 'NA'
short_df['score_2_ep'] = 'NA'
short_df['score_2_ship'] = 'NA'
short_df['score_2_final'] = 'NA'
#short_df['score_1'] = 'NA'

# Global variables for score 2 calculation
encountered_row = []
score = 100
prev_penalized_list = {}

"""
# Returns score for every transaction for part 1 (for every accountid)
def calc_score_1( row, account_distinct_dict ):
    #print("="*20)
    #print("Calculating SCORE 1")
    del row['score_1'], row['score_2_ep']
    to_match = '!'.join([str(i) for i in row])
    #li = [i.strip() for i in row.to_string(header=False,index=False).split('\n')]
    #to_match  = '!'.join(li)
    max_item = max(list(account_distinct_dict.values()))
    if(to_match in list(account_distinct_dict.keys())):
        return round((account_distinct_dict[to_match]/max_item) * 100,2)
    else:
        print("TO MATCH : "+to_match)
        print("\nAccount KEYS\n")
        for i in account_distinct_dict.keys():
            print(i)
        return "Error"
"""

# Returns score for every transaction for Part 2 (for every accountid)
def calc_score_2_ep( row, cols_weights ):
    print("="*20)
    print("Calculating SCORE 2")
    global encountered_row, score, prev_penalized_list
    to_penalize = 0
    to_add = 0
    if (len(encountered_row)==0):
        encountered_row = {k: [v] for k,v in row.iteritems()}  # Update enc row - Elements like {'columnname':[value]}
        return score
    else:
        for key in row.keys():
            if(key in ['score1', 'score_2_ep', 'score_2_ship', 'accountid']):  # Go to another key (we don't want these)
                continue
            if(row[key] in encountered_row[key]):
                if key in prev_penalized_list and prev_penalized_list[key]>0:  # Column penalized in last transaction
                    print(key + " back to normal - increasing " + str((1/3) * cols_weights[key]))
                    to_add += increase_weight * cols_weights[key]   # We add 1/3rd of weight to it (as updated above)
                    prev_penalized_list[key] = prev_penalized_list[key] - 1
                else:    # Did not penalize last time
                    pass   # Just ignore
            else:
                print(key + " changed - decreasing " + str(cols_weights[key]))
                to_penalize += cols_weights[key]  # Penalizing
                if key in prev_penalized_list:
                    prev_penalized_list[key] += inc_ratio
                else:
                    prev_penalized_list[key] = inc_ratio  # Column penalized in this transaction
                encountered_row[key].append(row[key])  # After penalizing , now it is encountered

        score = score - to_penalize + to_add  # updating score for next transaction
        print("Penalized list counts : " + str(prev_penalized_list))
        return round(score,2)

# This function sets global params to default
def cleanup():
    global encountered_row, score, prev_penalized_list
    encountered_row = []
    score = 100
    prev_penalized_list = {}

##############################################################################

#Ricardo's Code:
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
        if point > ctrl_stdup or point < ctrl_stddown:
            out_val.append(point)
            out_dates.append(feature2[i])
        else:
            pos_val.append(point)
            pos_dates.append(feature2[i])
    
    scores_list = []
    for o in out_val:
        if o > ctrl_stdup:
            scores_list.append(o-ctrl_stdup if o-ctrl_stdup < 100 else 100)
        if o < ctrl_stddown:
            scores_list.append(ctrl_stddown-o if ctrl_stdup-o < 100 else 100)
    
    score = 100 - (sum(scores_list)*0.1) if 100 - (sum(scores_list)*0.1) > 0 else 0

    if not scores_list :
        scores_list.append(100)
    else:
        scores_list =  100 - np.array(scores_list)
    if show:
        #show plot
        print('the score is: ',score)
        if isinstance(out_dates[0], datetime.date):
            dates_list = feature2
            plot_dates = matplotlib.dates.date2num(dates_list)
            plt.plot_date(pos_dates, pos_val, c= 'b')
            plt.plot_date(out_dates, out_val, c='r')
        else:
            plt.scatter(pos_dates, pos_val, c= 'b')
            plt.scatter(out_dates, out_val, c='r')


        plt.plot((min(feature2), max(feature2)), (m, m), 'k-')
        plt.plot((min(feature2), max(feature2)), (ctrl_stdup, ctrl_stdup), 'k-', c='g')
        plt.plot((min(feature2), max(feature2)), (ctrl_stddown, ctrl_stddown), 'k-', c='g')
        plt.show()

    return scores_list

def get_amount(data):
    """Calculates the cart amount as a categorical variable given the JSON data
    for a specific data entry.

    :param data: The JSON data string to parse."""

    data = json.loads(data)
    products = data.get("CartProduct", {"all": []})

    # Make sure we get all products in the cart.
    if "all" in products: products = products["all"]
    else : products = [products]

    amount = 0.0
    for p in products:
        try: amount += float(p["productPrice"]) * float(p["productQuantity"])
        except: pass
        #print(float(p["productPrice"]), float(p["productQuantity"]),' total amount, ',amount)
    return amount

def get_date(data):
    """Calculates the cart amount as a categorical variable given the JSON data
    for a specific data entry.

    :param data: The JSON data string to parse."""
    data = json.loads(data)
    dates = data.get("ReceiptData", {"orderDate": []})
    # Make sure we get all products in the cart.
    if isinstance(dates['orderDate'], unicode):
        str2date = datetime.datetime.strptime(str(dates['orderDate']), "%Y/%m/%d").date()
        return str2date
    elif isinstance(dates['orderDate'], str):
        str2date = datetime.datetime.strptime(dates['orderDate'], "%Y/%m/%d").date()
        return str2date
    else:
        return date(2001, 1, 11)

def calc_cart_score(row, accountid):

    acc_trans = []

    for _, transaction in df.iterrows():
        if transaction['accountid'] == accountid and get_date(row['actiondata'])>get_date(transaction['actiondata']):
            acc_trans.append(get_amount(transaction['cart_info']) )

    amounts = acc_trans + [get_amount(row['cart_info'])]

    score = control_chart(amounts, list(range(len(amounts))))[-1]

    return score


def event_triggered(row):

    triggered_signals = 0
    try: row = json.loads(row)
    except: row = []

    for e in row:
        #print(row)
        if e in event_trigger_signals:
            triggered_signals += 1
            if e == "geo_anonymous":
                triggered_signals += 4

    score = 1-(triggered_signals*0.1) if triggered_signals <= 4 else 1-0.4
    return score


df_fraud = pd.read_csv('fraud_list.csv')

score2_on = True
shipp_on = True
cart_on = True

working_models = 3
try:
    if sys.argv[3]:
        if 'no-cart' in sys.argv[3]:
            cart_on = False
            working_models -= 1

        if 'no-shipp' in sys.argv[3]:
            shipp_on = False
            working_models -= 1

        if 'no-score2' in sys.argv[3]:
            score2_on = False
            working_models -= 1
except:
    pass

##############################################################################

acc_transactions = {name:0 for name in short_df['accountid'].unique()}

def count_transaction(accountid):

    global acc_transactions

    acc_transactions[accountid] += 1

    return acc_transactions[accountid]
# Main function that calls other functions for scoring

def score_for_account( account_name):

    # Getting account wise data
    account_df = short_df[short_df['accountid']==account_name].sort_values('unixtime')
    account_df = account_df.replace(np.nan, -1)   # Replacing nulls with 1's

    # Score 1 - deal later
    #account_distinct_entries = {'!'.join(str(x) for x in i[0:len(i)-2]): int(i[-1]) for i in account_df.groupby(ep_cols, as_index=False).count().values.tolist()}
    #account_df['score_1'] = account_df.apply(lambda row: calc_score_1(row, account_distinct_entries), axis=1)

    # Calling calc_score_2 function for every row
    global acc_transactions
    account_df['date'] = account_df.apply(lambda row: str(get_date(row['actiondata'])), axis=1)

    account_df['total_Cart_amount'] = account_df.apply(lambda row: get_amount(row['cart_info']), axis=1)

    account_df['numer_of_transaction'] = account_df.apply(lambda row: count_transaction(row['accountid']), axis=1)

    account_df['eventtriggered_score'] = account_df.apply(lambda row: (1-event_triggered(row['eventtriggeredsignals']))*100, axis=1)
    if cart_on:
        account_df['score_2_cart'] = account_df.apply(lambda row: calc_cart_score(row, account_name ), axis=1)
    else:
        account_df['score_2_cart'] = account_df.apply(lambda row: 0, axis=1)
    cleanup()
    if shipp_on:
        account_df['score_2_ep'] = account_df.apply(lambda row: calc_score_2_ep(row[ep_cols], cols_weights_ep), axis=1)
    else:
        account_df['score_2_ep'] = account_df.apply(lambda row: 0, axis=1)
    cleanup()
    if score2_on:
        account_df['score_2_ship'] = account_df.apply(lambda row: calc_score_2_ep(row[ship_cols], cols_weights_ship), axis=1)
    else:
        account_df['score_2_ship'] = account_df.apply(lambda row: 0, axis=1)
    cleanup()  # Resetting global variables after every account

    if account_name in list(df_fraud['customer_email']):
        account_df['score_2_final'] = account_df.apply(lambda row:0,axis=1)
    elif acc_transactions[account_name] <= 1:
        account_df['score_2_final'] = account_df.apply(lambda row:"n",axis=1)
    else:
        account_df['score_2_final'] = account_df.apply(lambda row: round((0.3*float(row['score_2_ep']) + 0.4*float(row['score_2_ship'])+ 0.3*float(row['score_2_cart'])),2)*event_triggered(row['eventtriggeredsignals']), axis=1)

    return account_df

start = time.time()
for account in short_df['accountid'].unique(): #short_df['accountid'].unique():

    print("!!!!!! "+account+" !!!!!!")
    new_df = score_for_account(account)
    short_df.update(new_df)

print("Total runtime : "+str((time.time() - start)/60)+ " minutes or "+str(time.time() - start)+" seconds")
print 'test'

#short_df[short_df['accountid']=='02davist11@gmail.com'].sort('unixtime')
short_df.sort_values('unixtime').to_csv(sys.argv[2])
