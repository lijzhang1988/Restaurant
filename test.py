import requests, datetime
from flask import Flask, jsonify, json
from pymongo import MongoClient

# Init app
#app = Flask(__name__)

def gettoken():
    
    api_url = 'http://9.110.88.163:8390/api/v20/admin/login?method=login'

    params = {
        "username": 'esadmin',
        "password": 'Zaq11qaz',
        "local": 'en_US',
        "output": 'application/json'
    }

    res = requests.get(api_url, params)
    result = res.json()
    str_sentiment = json.dumps(result)
    dict_sentiment = json.loads(str_sentiment)
    result = dict_sentiment['es_apiResponse']['es_securityToken']

    return result

def getShopName():
    api_url = "http://9.110.88.163:8393/api/v10/search/facet"

    params = { 
        "collection": "col_45903", 
        "query": '*:*',
        "facet": '{"count":9999,"depth":1,"namespace":"keyword","id":"$.metaData.StroeName"}',
        "queryLang": 'ja',
        "output": 'application/json'
    }
    res = requests.get(api_url, params)
    result_shopname = res.json()
    list_shopname = ['']
    str_shopname = json.dumps(result_shopname)
    dict_shopname = json.loads(str_shopname)
#    print(dict_shopname)
    if dict_shopname['es_apiResponse'] == None:
        result = []
    else:
#dict for multiple shop, list for single shop
        if isinstance(dict_shopname['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue'], dict):
            data_shopname = '"' + dict_shopname['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']['label'] + '"'
            list_shopname.append(data_shopname)
        else:
            for i in range(0, len(dict_shopname['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue'])):
                data_shopname = '"' + dict_shopname['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue'][i]['label'] + '"'
                list_shopname.append(data_shopname)
        result = list_shopname

    return result

def getsentiment(res_shopname):

    rstring = ''
    for j in res_shopname:
        inside_code=ord(j)
        if inside_code == 12288:                                         
            inside_code = 32 
        elif (inside_code >= 65281 and inside_code <= 65374): 
            inside_code -= 65248
        rstring += chr(inside_code)
    api_url = "http://9.110.88.163:8393/api/v10/search/facet"
    #query = '(*:*) AND (' + ' OR keyword::/"メタデータ"/"店名"/'.join(res)[4:] + ')'
    query = '(*:*) AND (keyword::/"メタデータ"/"店名"/' + rstring + ')'
    params = { 
        "collection": "col_45903", 
        "query": query,
        "facet": '{"count":500,"depth":1,"namespace":"keyword","id":"$.metaData.StroeName"}',
        "sentiment": True,
        "enablePreDefinedResults": True,
        "enableSpellCorrection": True,
        "enableFacetPath": True,
        "queryLang": 'ja',
        "linguistic": 'engine',
        "nearDuplication": 'shingle',
        "synonymExpansion": 'automatic',
        "output": 'application/json'
    }
    res = requests.get(api_url, params)
    result = res.json()
    #print(json.dumps(result, indent = 4, ensure_ascii = False))
    return result
#@app.route('/***',methods=['GET','POST'])
def setsentiment(res_sentiment):

    str_sentiment = json.dumps(res_sentiment)
    dict_sentiment = json.loads(str_sentiment)
    pos_count = dict_sentiment['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']['es_property'][3]['value']
    neg_count = dict_sentiment['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']['es_property'][4]['value']
    amb_count = dict_sentiment['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']['es_property'][5]['value']
    total_count = dict_sentiment['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']['weight']
    data_sentiment = {
        'shop_name'        : dict_sentiment['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']['label'],
        'total_count'      : total_count,
        'positive_count'   : pos_count,
        'ambivalent_count' : amb_count,
        'negative_count'   : neg_count,
        'neutral_count'    : dict_sentiment['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']['es_property'][6]['value'],  
    }
    result = data_sentiment
    return result
#    return jsonify({'result':result})

def getPositiveFacet(res_shopname):

    api_url = "http://9.110.88.163:8393/api/v10/search/facet"
    query = '(*:*) AND (keyword::/"メタデータ"/"店名"/' + res_shopname + ')'
    params = { 
        "collection": 'col_45903', 
        "query": query,
        "facet": '{"count":51,"depth":1,"namespace":"keyword","id":"$._sentiment.phrase.positive"}',
        "sentiment": False,
        "output": "application/json"
    }

    res = requests.get(api_url, params)
    result = res.json()        

    return result

def getNegativeFacet(res_shopname):

    api_url = "http://9.110.88.163:8393/api/v10/search/facet"
    query = '(*:*) AND (keyword::/"メタデータ"/"店名"/' + res_shopname + ')'
    params = { 
        "collection": 'col_45903', 
        "query": query,
        "facet": '{"count":51,"depth":1,"namespace":"keyword","id":"$._sentiment.phrase.negative"}',
        "sentiment": False,
        "output": "application/json"
    }

    res = requests.get(api_url, params)
    result = res.json()  

    return result

def setPositiveFacet(res_facet):
    
    list_facet = ['']
    list_res = []
    str_facet = json.dumps(res_facet)
    dict_facet = json.loads(str_facet)
    if dict_facet['es_apiResponse'] == None:
        result = ''
    else:
        if isinstance(dict_facet['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue'], dict):
            list_res = [dict_facet['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']]
        else:
            list_res = dict_facet['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']
        len_facet = len(list_res)
        for i in range(0, len_facet):
            list_facet.append(list_res[i]['label'])
        result = ' OR keyword::/"Sentiment Analysis"/"Phrase"/"Positive"/'.join(list_facet)[4:]

    return result

def setNegativeFacet(res_facet):
    
    list_facet = ['']
    list_res = []
    str_facet = json.dumps(res_facet)
    dict_facet = json.loads(str_facet)
    if dict_facet['es_apiResponse'] == None:
        result = ''
    else:
        if isinstance(dict_facet['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue'], dict):
            list_res = [dict_facet['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']]
        else:
            list_res = dict_facet['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']
        len_facet = len(list_res)
        for i in range(0, len_facet):
            list_facet.append(list_res[i]['label'])
        result = ' OR keyword::/"Sentiment Analysis"/"Phrase"/"Negative"/'.join(list_facet)[4:]

    return result

def getTimeSeries(res_shopname, dict_facet, time_now, time_then):

    api_url = "http://9.110.88.163:8393/api/v10/search/facet"

    params = { 
        "collection": 'col_45903', 
        "query": '((*:*) AND (keyword::/"メタデータ"/"店名"/' + res_shopname + ')) AND (' + dict_facet + ')',
        "facet": '{"count":-1,"depth":1,"namespace":"date","id":"$.month","segmentation":{"returnZeroCount":true,"offset":0,"length":101,"returnEarliestDate":true}}',
        "querylang": 'ja',
        "linguistic":'engine',
        "enablePreDefinedResults": True,
        "enableSiteCollapsing":False,
        "nearDuplication": 'shingle',
        "synonymExpansion": 'automatic',
        "documentPart": 'aggregation',
        "summaryLengthRatio": 100,
        "enableFacetPath": True,
        "sentiment": False,
        "output": 'application/json'
    }
    res = requests.get(api_url, params)
    json_res = res.json()
    list_json = []
    list_series = []
    str_series = json.dumps(json_res)
    dict_series = json.loads(str_series)
    if dict_series['es_apiResponse'] == None:
       result = '' 
    else:
        if isinstance(dict_series['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue'], dict):
            list_json = [dict_series['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']]
        else:
            list_json = dict_series['es_apiResponse']['ibmsc_facet']['ibmsc_facetValue']
        len_series = len(list_json)
        for i in range(0, len_series):
            month = list_json[i]['label']
            if  time_now >= month and time_then < month: 
                data_series = {
                    "month"  : month,
                    "count" : list_json[i]['weight']
                }
                list_series.append(data_series)
        result = list_series
    return result

#@app.route('/***',methods=['GET','POST'])
def getCompare(res_shopname, dict_pos_series, dict_neg_series, time_then, period):
    
    data_result = {}
    list_result = [{'shop_name' : res_shopname}]
    for i in range(1,  (period * 12) + 1):
        month = (time_then + datetime.timedelta(days = i * 30)).strftime('%Y%m')
        data_result = {
            "month"    : month,
            "positive" : 0,
            "negative" : 0,
        }
        list_result.append(data_result)
    for i in range(0,  period * 12):
        month = (time_then + datetime.timedelta(days = i * 30)).strftime('%Y%m')
        for j in range(0, len(dict_pos_series)):
            if dict_pos_series[j]['month'] == month:
                list_result[i]['positive'] = dict_pos_series[j]['count'] 
        for k in range(0, len(dict_neg_series)):       
            if dict_neg_series[k]['month'] == month:
                list_result[i]['negative'] = dict_neg_series[k]['count']   
    result = list_result
    return result
#    return jsonify({'result':result})

def getAddress(res_shopname):
    #print(res_shopname)
    api_url = "http://9.110.88.163:8393/api/v10/search"

    params = { 
        "collection": "col_45903", 
        "query": res_shopname,
        "queryLang": 'ja',
        "start": 0,
        "result": 1,
        "output": 'application/json'
    }
    res = requests.get(api_url, params)
    json_address = res.json()
    str_address = json.dumps(json_address)
    dict_address = json.loads(str_address)
    list_address = []
    if 'es_result' not in dict_address['es_apiResponse']:
        result =  {
        "address"   : ''
        }
    else:
        if isinstance(dict_address['es_apiResponse']['es_result'], dict):
            list_address = [dict_address['es_apiResponse']['es_result']]
        else:
            list_address = dict_address['es_apiResponse']['es_result']
        data_address = {
            "address"   : list_address[0]['ibmsc_field'][1]['#text']
        }
        result = data_address
    #print(result)
    return result

def InsertData(res_address, res_sentiment, res_wex):
    #print(res_address)
    #print(res_sentiment)
    conn = MongoClient(host='localhost',port=27017)
    db=conn.wex_db
    wex_set = db.wex_set
    list_res = []
    for i in range(0,len(res_address)):
        res_wex[i].insert(1,res_address[i])
        del res_sentiment[i]['shop_name']
        res_wex[i].insert(2,res_sentiment[i])
        data_res = {
            'shop_name'   : res_wex[i][0]['shop_name'],
            'address'     : res_wex[i][1]['address'],
            'sentiment'   : res_wex[i][2],
            'time_series' : res_wex[i][3:],
        }
        list_res.append(data_res)
        wex_set.insert_one(
            {
                'shop_name'   : list_res[i]['shop_name'],
                'address'     : list_res[i]['address'],
                'sentiment'   : list_res[i]['sentiment'],
                'time_series' : list_res[i]['time_series'],
            }
        )    
    return list_res

if __name__ == "__main__":
#    app.run(debug=True)
#get token for manage
    token = gettoken()

#get shop name
    res_shopname = getShopName()
    #print(res_shopname)
    #res_temp = ['ガスト 豊中南店','ガストロテカ ビメンディ（gastroteka bimendi）','ガスト 大阪ATC店','ステーキガスト 大阪鷺洲店','ガスト 大阪産業大学前店']
    #res_temp = ['いきなり!ステーキ 三鷹東八道路店']
    #res_temp = request.get_json()['shopname']
    '''
    res_shopname = ['']
    for i in range(0, len(res_temp)):
        res_shopname.append('"'+res_temp[i]+'"')
    
    if res_shopname == []:
        pass
    else:
    '''    
#get sentiment
        
        #print(dict_sentiment)
#get positive facet 
    result_wex = []
    result_add = []
    result_sentiment=  []
    for i in range(1, len(res_shopname)):  
        print(i)
        res_sentiment = getsentiment(res_shopname[i])
        dict_sentiment = setsentiment(res_sentiment)
        result_sentiment.append(dict_sentiment)
        res_pos_facet = getPositiveFacet(res_shopname[i])
        res_neg_facet = getNegativeFacet(res_shopname[i])
        dict_pos_facet = setPositiveFacet(res_pos_facet)
        dict_neg_facet = setNegativeFacet(res_neg_facet)
#get time series base on facet
        period = 1
#    period = request.get_json().['period']
        time_now = datetime.datetime.now()
        time_then = (time_now - datetime.timedelta(days = period * 365))
        dict_pos_series = getTimeSeries(res_shopname[i], dict_pos_facet, time_now.strftime('%Y%m'), time_then.strftime('%Y%m'))
        dict_neg_series = getTimeSeries(res_shopname[i], dict_neg_facet, time_now.strftime('%Y%m'), time_then.strftime('%Y%m'))
        res_wex = getCompare(res_shopname[i], dict_pos_series, dict_neg_series, time_then, period)
        result_wex.append(res_wex)
#insert into mongodb
        res_address = getAddress(res_shopname[i])
        result_add.append(res_address)
#            print('for loop:', datetime.datetime.now())
    #print(result_add)
    #print(result_sentiment)
    res_mongo = InsertData(result_add, result_sentiment, result_wex)
    #print(res_mongo)
    
#delete file after wex analyze
#        print(json.dumps(result_wex, indent = 4, ensure_ascii = False))
    
