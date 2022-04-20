import os
import tempfile

import pytest
import configparser
import json
from jsonconfigparser import JSONConfigParser
from flaskrapp.opinion_process import create_app
from flaskrapp.opinion_process.mongodb import get_db, init_db


# To run the tests, use the pytest command.
#pytest
#
#
# To measure the code coverage of your tests, use the coverage command to run pytest
# coverage run -m pytest
#
#You can either view a simple coverage report in the terminal:
#
#coverage report
#
#An HTML report allows you to see which lines were covered in each file:
#
#coverage html
#
#
# Read JSON Config File
#Bibliografy
#
# https://pypi.org/project/json-config-parser/
#
#

# with open(os.path.join(os.path.dirname(__file__), 'data.sql'), 'rb') as f:
#config = configparser.ConfigParser()
#_data_sql = JSONConfigParser(os.path.join(os.path.dirname(__file__), 'data.sql'))
json_data_file = open(os.path.join(os.path.dirname(__file__), 'data_json.sql'))
data = json.load(json_data_file)
#_data_sql = config.read(os.path.join(os.path.dirname(__file__), 'data.sql'))
# print("--- Values ---- user ")
# print (_data_sql.get("user","users"))
# print("--- Values ---- post")
# print(_data_sql.get("post","posts"))
# _data_sql = f.read().decode('utf8')


@pytest.fixture
def app():
    db_fd, db_path = tempfile.mkstemp()

    app = create_app({
        'TESTING': True,
        'DATABASE': db_path,
        'DATABASE_ADDRESS':'localhost',
        'DATABASE_NAME':'wisepocketDB',
        'DATABASE_PORT':'27017',
         'DATABASE_TYPE':'MONGO'
    })

    with app.app_context():
        init_db()
        print(data['users'][0])
        #Insert in user collection

        #Insert in post collection
        #print (_data_sql["post"]["posts"])
        #get_db().executescript(_data_sql)

    yield app

    os.close(db_fd)
    os.unlink(db_path)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def runner(app):
    return app.test_cli_runner()


class AuthActions(object):
    def __init__(self, client):
        self._client = client

    def login(self, username='test', password='test'):
        return self._client.post(
            '/auth/login',
            data={'username': username, 'password': password}
        )

    def logout(self):
        return self._client.get('/auth/logout')


@pytest.fixture
def auth(client):
    return AuthActions(client)