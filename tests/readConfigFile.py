import os
import tempfile

import pytest
import configparser
from jsonconfigparser import JSONConfigParser

_data_sql = JSONConfigParser(os.path.join(os.path.dirname(__file__), 'data.sql'))
print("--- Values ---- user")
print(_data_sql.get("user", "users"))
# print(_data_sql["user"]["users"])
print("--- Values ---- post")
print(_data_sql.get("post", "posts"))

import json
#Bibliografy
#
# https://martin-thoma.com/configuration-files-in-python/
#
#
with open('data_json.sql') as json_data_file:
    data = json.load(json_data_file)
print( data['users'][0])
