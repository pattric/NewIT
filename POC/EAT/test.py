# coding: utf-8

import json
import requests

#
url = 'http://127.0.0.1:5000/predict'
    
#
def call_api(desc='VSD-P-522 failed causing a R-221 shutdown'):
    #
    data = json.dumps({'description': desc})
    headers = {'content-type': 'application/json'}
    req = requests.post(url, data, headers)
    #
    print(req.text)

#
call_api()
#
print('done')
