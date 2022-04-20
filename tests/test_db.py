from pymongo import MongoClient
import pymongo
import pytest
# importing ObjectId from bson library
from bson.objectid import ObjectId
from random import choice

from faker import Faker

from flaskrapp.opinion_process.mongodb import get_db, init_db, insert_sentences,insert_opinion_aspect, insert_opinion_entity


# def test_get_close_db(app):
#     with app.app_context():
#         db = get_db()
#         assert db == get_db()
#
#     # with pytest.raises(pymongo.ProgrammingError) as e:
#     #     db.execute('SELECT 1')
#
#     #assert 'closed' in str(e.value)

def test_insert_sentence_in_DB(app):
    with app.app_context():
        db = get_db()
        assert db == get_db()

        fake = Faker()
        sentence = fake.sentence(nb_words=150, variable_nb_words=True)
        id_opinion =  fake.isbn10()
        id_campaing = fake.isbn10()
        start_setence = 23
        end_sentence = 3*start_setence
        language = 'en'
        result = insert_sentences(sentence,id_opinion,id_campaing,start_setence,end_sentence, language)

        if result == None:
            raise  Exception(" Bad insert opinion")
        else:
           mycol = db["opinion_sencentes"]
           print("****** Id ******** ", result.inserted_id)

           myquery = {'_id': ObjectId(result.inserted_id)}

           #Query in Mongo and Python
           #
           # Bibliografy:
           #
           # https://www.w3schools.com/python/python_mongodb_query.asp
           #
           # https://www.analyticsvidhya.com/blog/2020/08/query-a-mongodb-database-using-pymongo/#h2_5
           #
           #mydoc = mycol.find(myquery)

           mydoc = mycol.find_one(ObjectId(result.inserted_id))

           if mydoc == None:
               raise Exception(" Not insert sentence")
           else:
               print("****** Data to Show ******** " )
               # for i in mydoc:
               #     print(i)
               print(mydoc)
               assert sentence == mydoc['sentence']

        #Inser

    # with pytest.raises(pymongo.ProgrammingError) as e:
    #     db.execute('SELECT 1')

    #assert 'closed' in str(e.value)
# def test_init_db_command(runner, monkeypatch):
#     class Recorder(object):
#         called = False
#
#     def fake_init_db():
#         Recorder.called = True
#
#     monkeypatch.setattr('flaskrapp.opinion_process.mongodb.init_db', fake_init_db)
#     result = runner.invoke(args=['init-mongodb'])
#     assert 'Initialized' in result.output
#     assert Recorder.called


def test_insert_aspect_in_DB(app):
    with app.app_context():
        db = get_db()
        assert db == get_db()

        fake = Faker()

        polarities = ['positive','negative','neutral']
        current_polarity = choice(polarities)

        sentence = fake.word()
        id_opinion =  fake.isbn10()
        id_campaing = fake.isbn10()
        start_setence = 23
        end_sentence = 3*start_setence
        language = 'en'
        result = insert_opinion_aspect(sentence,current_polarity,id_opinion,id_campaing,start_setence,end_sentence, language)

        if result == None:
            raise  Exception(" Bad insert opinion")
        else:
           mycol = db["opinion_aspects"]
           print("****** Id ******** ", result.inserted_id)

           myquery = {'_id': ObjectId(result.inserted_id)}

           #Query in Mongo and Python
           #
           # Bibliografy:
           #
           # https://www.w3schools.com/python/python_mongodb_query.asp
           #
           # https://towardsdatascience.com/generating-fake-data-with-python-c7a32c631b2a
           #
           # https://www.analyticsvidhya.com/blog/2020/08/query-a-mongodb-database-using-pymongo/#h2_5
           #
           #mydoc = mycol.find(myquery)

           mydoc = mycol.find_one(ObjectId(result.inserted_id))

           if mydoc == None:
               raise Exception(" Not insert sentence")
           else:
               print("****** Data to Show ******** " )
               # for i in mydoc:
               #     print(i)
               print(mydoc)
               assert sentence == mydoc['aspect']

def test_insert_entity_in_DB(app):
    with app.app_context():
        db = get_db()
        assert db == get_db()

        fake = Faker()

        polarities = ['ORG', 'CARDINAL', 'DATE', 'GPE', 'PERSON', 'MONEY', 'PRODUCT', 'TIME', 'PERCENT', 'WORK_OF_ART', 'QUANTITY', 'NORP', 'LOC', 'EVENT', 'ORDINAL', 'FAC', 'LAW', 'LANGUAGE']
        current_polarity = choice(polarities)

        sentence = fake.word()
        id_opinion =  fake.isbn10()
        id_campaing = fake.isbn10()
        start_setence = 23
        end_sentence = 3*start_setence
        language = 'en'
        result = insert_opinion_entity(sentence,current_polarity,id_opinion,id_campaing,start_setence,end_sentence, language)

        if result == None:
            raise  Exception(" Bad insert opinion aspect")
        else:
           mycol = db["sentence_entities"]
           print("****** Id ******** ", result.inserted_id)

           myquery = {'_id': ObjectId(result.inserted_id)}

           #Query in Mongo and Python
           #
           # Bibliografy:
           #
           # https://www.w3schools.com/python/python_mongodb_query.asp
           #
           # https://www.analyticsvidhya.com/blog/2020/08/query-a-mongodb-database-using-pymongo/#h2_5
           #
           #mydoc = mycol.find(myquery)

           mydoc = mycol.find_one(ObjectId(result.inserted_id))

           if mydoc == None:
               raise Exception(" Not insert entity")
           else:
               print("****** Data to Show entity ******** " )
               # for i in mydoc:
               #     print(i)
               print(mydoc)
               assert sentence == mydoc['entity']