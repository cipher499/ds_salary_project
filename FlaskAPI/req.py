""""
Created on Tue, July 18, 2023
'
@author: cipher499
"""

import requests
from data_input import data_in

URL = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}
data = {"input": data_in}

#uncomment and run after having run the lines above
#r = requests.get(URL, headers=headers, json=data)

#r.json()