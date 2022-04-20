from docutils.nodes import address
from pymongo import MongoClient
import pymongo
# Bibliografy
#
#  url: https://faker.readthedocs.io/en/master/
#
from faker import Faker
from random import randint
from pymongo import errors
import click
from flask import current_app, g
from flask.cli import with_appcontext

def get_db():
    if 'db' not in g:
        # Provide the mongodb atlas url to connect python to mongodb using pymongo
        server_address = current_app.config['DATABASE_ADDRESS']
        database_name =   current_app.config['DATABASE_NAME']
        database_port = current_app.config['DATABASE_PORT']
        # 'mongodb://localhost:27017/'
        CONNECTION_STRING = "mongodb://" +server_address+":"+database_port +"/"+ database_name
        # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
        from pymongo import MongoClient
        #client = MongoClient(CONNECTION_STRING)
        port = 27017
        #client = MongoClient(address=server_address, port=int(database_port), maxPoolSize=50)
        print (" PORT = ", int(database_port))
        client = MongoClient(port= int (database_port), maxPoolSize=50)
        print("<== Create conection data ==>")

        # Create the database for our example (we will use the same database throughout the tutorial
        return client[database_name]

    return g.db

def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():

    #Intialize database conection
    db = get_db()

    #Open a file in app path
    with current_app.open_resource('mongoschema.sql') as f:
        # import the 'errors' module from PyMongo
        collectionames = f.readlines()
        for icolName in collectionames:
            colname = icolName.decode('utf-8')
            polarities = ['positive','negative','neutral']
            try:
                print ("Collection name ", colname)
                col_dict = db.validate_collection(colname)
                print(colname, "has", col_dict['nrecords'], "documents on it.\n")
            except errors.OperationFailure as err:
                col_dict = None
                db[colname]
                fake = Faker()
                if colname == 'users':
                 result = db[colname].insert_one({ 'name':fake.name(),'address':fake.address()})
                elif colname == 'reviews':
                  resultreview = db[colname].insert_one({ 'name':fake.name(),'addres':fake.address(), 'time':fake.date(),'polarity': polarities[randint(0, (len(polarities)-1))],'text':fake.text()})
        for idb in db.list_collection_names():
          print(idb)


def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
#*
#
#
#
#*
def insert_sentences(_sentence, _idopinion, _idcamapaing, _start, _end, _language = 'en'):
    db = get_db()
    if db != None and _sentence != '' and _sentence != None:
        result = db["opinion_sencentes"].insert_one({'sentence': _sentence,
                                                     'idpinion': _idopinion,
                                                      'idcampaign':_idcamapaing,
                                                      'language':_language,
                                                      'start':_start,
                                                       'end':_end})
        return result
    else:
        raise Exception("insert_sentences: Invalid data to insert in DB")
#*
#
#  Insert aspecto opinion in sentence after Machile Learning and NLP processing
#
#*
def insert_opinion_aspect(_aspect, _polarity, _idsentence, _idopinion, _idcamapaing, _start, _end, _language = 'en'):
    db = get_db()
    if db != None and _aspect != '' and _aspect != None and _polarity != "":
        result = db["opinion_aspects"].insert_one({'aspect': _aspect,
                                                      'polarity':_polarity,
                                                      'idpinion': _idopinion,
                                                      'idcampaign':_idcamapaing,
                                                      'idsentence':_idsentence,
                                                      'language': _language,
                                                      'start':_start,
                                                       'end':_end})
        return result
    else:
        raise Exception("insert_opinion_aspect: Invalid data")
#*
#
#  Insert entity  in sentence after Machile Learning and NLP processing
#
#*
def insert_opinion_entity(_entity, _entity_type, _idsentence, _idopinion, _idcamapaing, _start, _end, _language = 'en'):
    db = get_db()
    if db != None and _entity != '' and _entity != None and _entity_type != "":

        ##
        ##  Warning maybe repetead entities in several sentences
        ##

        result = db["sentence_entities"].insert_one({'entity': _entity,
                                                      'entity_type':_entity_type,
                                                      'idpinion': _idopinion,
                                                      'idcampaign':_idcamapaing,
                                                      'idsentence':_idsentence,
                                                      'language': _language,
                                                      'start':_start,
                                                       'end':_end})
        return result
    else:
        raise Exception("Invalid data")

@click.command('init-mongodb')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the mongo database.')