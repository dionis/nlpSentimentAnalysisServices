import os
from flask import Flask

#Instructor, init_predictor, getABSAModelClassify
from .nlp import absa_predictor
from . import mongodb as db
#from . import auth as auth
#from . import mongodb

# Star APP Windows
#
# set FLASK_APP=flaskr
# set FLASK_ENV=development
#
# flask run

# ABSA to get Sentiment Analysis
# https://pypi.org/project/aspect-based-sentiment-analysis/
#
# Others:
#   https://github.com/zarmeen92/ABSA-Toolkit
#


##  
#    Python traslate 
#       https://pypi.org/project/translate/
#      Spanish to Englis and ABSA in spanish  (Hard in aspect level, search ingles aspect traduction
#      in spanish's sentece)
#
#
#

#--------------------------------------------------------------------------
#
#  CREATE AND IMPORT PYTHON MODULE
#
#   https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Modules_and_Packages.html
#
#   https://www.askpython.com/python/python-packages
#


def create_app(test_config=None):
    # create and configure the app

    app = Flask(__name__, instance_relative_config=True)


    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
        DATABASE_ADDRESS='localhost',
        DATABASE_NAME='wisepocketDB',
        DATABASE_PORT='27017',
        DATABASE_TYPE='MONGO'
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    opt, model_classes, tokenizer, spacyModule, appr = absa_predictor.init_predictor()

    app.opt = opt
    app.model_classes = model_classes
    app.spacyModule = spacyModule
    app.tokenizer = tokenizer
    app.appr = appr

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # from . import sqlitedb as db
    # db.init_app(app)


    db.init_app(app)
    # a simple page that says hello


    from . import process
    app.register_blueprint(process.prc)
    #app.add_url_rule('/wprocess', endpoint='processEx2')

    #Authentication data
    #app.register_blueprint(auth.bp)

    @app.route('/hello')
    def hello():
            return 'Hello, World!'

    # @app.route('/process_opinion', methods=['POST', 'GET'])
    # def process():
    #     error = None
    #     app.logger.debug('Call request')
    #     # app.logger.warning('A warning occurred (%d apples)', 42)
    #     # app.logger.error('An error occurred')
    #     if request.method == 'POST':
    #         app.logger.debug(request.form)
    #         idOpinion = request.form['id_opinion']
    #         app.logger.debug('Opinion id ==> ')
    #         app.logger.debug(idOpinion)
    #         idCampaign = request.form['id_campaign']
    #         app.logger.debug('Campign id ==> ')
    #         app.logger.debug(idCampaign)
    #         textOpinion = request.form['textopinion']
    #         app.logger.debug('Opinion text data ==> ')
    #         app.logger.debug(textOpinion)
    #
    #         return {
    #             "id_opinion": idOpinion,
    #             "id_campaign": idCampaign,
    #             "language": 'es',
    #             "opinion_info": {
    #                 'sentences': [
    #                     {
    #                         'text': 'Test1',
    #                         'aspects': [{'text': 'aspect1', 'polarity': 'positive', 'pos': 23, 'length': 5},
    #                                     {'text': 'ascpect2', 'polarity': 'negative', 'pos': 34, 'length': 12}],
    #                         'entities': [{'text': 'entity1', 'type': 'person', 'pos': 13, 'length': 5},
    #                                      {'text': 'entity2', 'type': 'place', 'pos': 4, 'length': 10}]
    #                     },
    #                     {
    #                         'text': 'Test2',
    #                         'aspects': [{'text': 'aspect1', 'polarity': 'positive', 'pos': 23, 'length': 5},
    #                                     {'text': 'ascpect2', 'polarity': 'negative', 'pos': 34, 'length': 12}],
    #                         'entities': [{'text': 'entity1', 'type': 'person', 'pos': 13, 'length': 5},
    #                                      {'text': 'entity2', 'type': 'place', 'pos': 4, 'length': 10}]
    #                     }
    #                 ]
    #             },
    #         }


    return app

