
from flask import (
    Blueprint, current_app, flash, g, redirect, render_template, request, session, url_for,jsonify
)

from .nlp.absa_predictor import Instructor, init_predictor, getABSAModelClassify
from werkzeug.security import check_password_hash, generate_password_hash

#from sqlitedb import get_db

from . import mongodb
#from mongodb import get_db

# Bibliografy:
#
# https://flask.palletsprojects.com/en/2.0.x/appcontext/
# https://flask.palletsprojects.com/en/2.0.x/reqcontext/#notes-on-proxies


prc = Blueprint('wnlp', __name__, url_prefix='/wnlp')
ERROR_LANGUAGE = "Not language detected in Text"
@prc.route('/sentence_sppliter', methods=['POST', 'GET'])
def sentenceSplitterEx():
    error = None
    #prc.logger.debug('Call request == in sentenceSplitterEx')

    ###Get parameter

    ###Use Spacy module
    app = current_app._get_current_object()
    spacyModule = app.spacyModule

    #prc.logger.warning('A warning occurred (%d apples)', 42)
    #prc.logger.error('An error occurred')
    if request.method == 'GET':
        return "OK"
    elif request.method == 'POST':
        #prc.logger.debug( request.form)
        idOpinion = request.form['id_opinion']
        #prc.logger.debug('Opinion id ==> ')
        #prc.logger.debug(idOpinion)
        idCampaign  = request.form['id_campaign']
        #prc.logger.debug('Campign id ==> ')
        #prc.logger.debug(idCampaign)
        textOpinion = request.form['textopinion']
        #prc.logger.debug('Opinion text data ==> ')
        #prc.logger.debug(textOpinion)

        #Send a JSON response in Request
        #
        # https://riptutorial.com/flask/example/5831/return-a-json-response-from-flask-api
        #
        #

        try:
            sentencesAnalysis = spacyModule.getSplitterSentences(textOpinion)
            if sentencesAnalysis == None:
                #Error in analysis
                return jsonify(
                    id_opinion=idOpinion,
                    id_campaign=idCampaign,
                    response="ERROR",
                    cause=sentencesAnalysis.cause_error)
            else:
                 sentencesList = sentencesAnalysis.sentences
                 sentences = []
                 language = sentencesAnalysis.language
                 for isentence in sentencesList:
                    sentences.append({'text': isentence})
                 return jsonify(
                     id_opinion= idOpinion,
                     id_campaign= idCampaign,
                     language=language,
                     opinion_info= {
                         'sentences':  sentences
                     })

        except Exception as e:
            return jsonify(
                        id_opinion=idOpinion,
                        id_campaign=idCampaign,
                        response="ERROR",
                        cause=e)


        # return {
        #     "id_opinion": idOpinion,
        #     "id_campaign": idCampaign,
        #     "language":'es',
        #     "opinion_info": {
        #         'sentences':[
        #             {
        #                 'text':'Test1',
        #                 'aspects':[ { 'text':'aspect1','polarity':'positive','pos':23,'length':5},
        #                             { 'text':'ascpect2', 'polarity':'negative','pos':34,'length':12}],
        #                 'entities': [{ 'text':'entity1','type':'person','pos':13,'length':5},
        #                             { 'text':'entity2', 'type':'place','pos':4,'length':10}]
        #             },
        #             {
        #                 'text': 'Test2',
        #                 'aspects':[{'text': 'aspect1', 'polarity': 'positive', 'pos': 23, 'length': 5},
        #                             {'text': 'ascpect2', 'polarity': 'negative', 'pos': 34, 'length': 12}],
        #                 'entities': [{'text': 'entity1', 'type': 'person', 'pos': 13, 'length': 5},
        #                              {'text': 'entity2', 'type': 'place', 'pos': 4, 'length': 10}]
        #             }
        #         ]
        #     },
        # }

@prc.route('/entities_detector', methods=['POST', 'GET'])
def  entitiesDetectorEx():
    error = None
    #prc.logger.debug('Call request == in entitiesDetectorEx')

    ###Get parameter

    ###Use Spacy module
    app = current_app._get_current_object()
    spacyModule = app.spacyModule

    # prc.logger.warning('A warning occurred (%d apples)', 42)
    # prc.logger.error('An error occurred')
    if request.method == 'GET':
        return "OK"
    elif request.method == 'POST':
        #prc.logger.debug(request.form)
        idOpinion = request.form['id_opinion']
        #prc.logger.debug('Opinion id ==> ')
        #prc.logger.debug(idOpinion)
        idCampaign = request.form['id_campaign']
        #prc.logger.debug('Campign id ==> ')
        #prc.logger.debug(idCampaign)
        textOpinion = request.form['textopinion']
        #prc.logger.debug('Opinion text data ==> ')
        #prc.logger.debug(textOpinion)

        try:
            sentencesAnalysis = app.spacyModule = spacyModule.getEntitiesSentences(textOpinion)
            if sentencesAnalysis == None:
                # Error in analysis
                return jsonify(
                    id_opinion= idOpinion,
                    id_campaign= idCampaign,
                    response= "ERROR",
                    cause= sentencesAnalysis.cause_error
                )
            else:
                sentencesList = sentencesAnalysis.sentences

                #All entities in text
                #We need figure out whic sentece is associated to each entities and
                #associated in return Data Structure

                entities = []
                language = sentencesAnalysis.language
                if language == "" or language == None:
                   #Error not languaje detected
                   return jsonify(
                       id_opinion = idOpinion,
                       id_campaign =  idCampaign,
                       response = "ERROR",
                       cause = ERROR_LANGUAGE
                   )
                else:
                    for ent in sentencesAnalysis.entities:
                        entities.append({
                              "text":ent.text,
                              "start_intext":ent.start_char,
                              "end_intext":ent.end_char,
                              "type":ent.label_
                        })
                    return jsonify (
                        id_opinion = idOpinion,
                        id_campaign = idCampaign,
                        language = language,
                        opinion_info = {
                            'entities': entities
                        }
                    )
        except Exception as e:
            return jsonify(
                id_opinion = idOpinion,
                id_campaign = idCampaign,
                response = "ERROR",
                cause = e
            )


@prc.route('/language_detector', methods=['POST', 'GET'])
def languageDetectorEx():
    error = None
    #prc.logger.debug('Call request == in languageDetectorEx')

    ###Get parameter

    ###Use Spacy module
    app = current_app._get_current_object()
    spacyModule = app.spacyModule

    # prc.logger.warning('A warning occurred (%d apples)', 42)
    # prc.logger.error('An error occurred')
    if request.method == 'GET':
        return "OK"
    elif request.method == 'POST':
        #prc.logger.debug(request.form)
        idOpinion = request.form['id_opinion']
        #prc.logger.debug('Opinion id ==> ')
        #prc.logger.debug(idOpinion)
        idCampaign = request.form['id_campaign']
        #prc.logger.debug('Campign id ==> ')
        #prc.logger.debug(idCampaign)
        textOpinion = request.form['textopinion']
        #prc.logger.debug('Opinion text data ==> ')
        #prc.logger.debug(textOpinion)

        try:
            sentencesAnalysis = app.spacyModule = spacyModule.getLanguage()
            if sentencesAnalysis == None:
                # Error in analysis
                return {
                    "id_opinion": idOpinion,
                    "id_campaign": idCampaign,
                    "response": "ERROR",
                    "cause": sentencesAnalysis.cause_error
                }
            else:
                sentencesList = sentencesAnalysis.sentences
                sentences = []
                language = sentencesAnalysis.language
                if language == "" or language == None:
                    # Error not languaje detected
                    return {
                        "id_opinion": idOpinion,
                        "id_campaign": idCampaign,
                        "response": "ERROR",
                        "cause": ERROR_LANGUAGE
                    }
                else:
                    return {
                        "id_opinion": idOpinion,
                        "id_campaign": idCampaign,
                        "language": language,
                        "opinion_info": {
                            'sentences': sentences
                        }
                    }
        except Exception as e:
            return {
                "id_opinion": idOpinion,
                "id_campaign": idCampaign,
                "response": "ERROR",
                "cause": e
            }



@prc.route('/process_opinion', methods=['POST', 'GET'])
def processEx():
    error = None
    #prc.logger.debug('Call request == in processEx')

    app = current_app._get_current_object()
    spacyModule = app.spacyModule
    #prc.logger.warning('A warning occurred (%d apples)', 42)
    #prc.logger.error('An error occurred')
    if request.method == 'GET':
        return "OK"
    elif request.method == 'POST':
        #prc.logger.debug( request.form)
        returned_data = request.get_json()
        if 'id_opinion' in request.form.keys() or returned_data == None:
            returned_data = request.form
        else:
            print(" ===> Send JSON values <===")

        idOpinion = returned_data['id_opinion']
        #prc.logger.debug('Opinion id ==> ')
        #prc.logger.debug(idOpinion)
        idCampaign  = returned_data['id_campaign']
        #prc.logger.debug('Campign id ==> ')
        #prc.logger.debug(idCampaign)
        textOpinion = returned_data['textopinion']
        #prc.logger.debug('Opinion text data ==> ')
        #prc.logger.debug(textOpinion)

        acpectOpinion = returned_data['aspectopinion']


        #Process of ABSA Opinion

        #1- Extract possible opinion in text

        #2- For each aspect in sentences
        #      Note: In text could have many sentences and in
        #            each sentences several possible aspect


        # examples:
        #
        # works well, and i am extremely happy to be back to an $T$.
        # apple os
        # 1
        #
        # i trust the $T$ at go sushi, it never disappoints.
        # people
        # 1
        #
        # the $T$ that came out were mediocre. portions of the
        # food
        # -1
        #
        # great food but the $T$ was dreadful !
        # service
        # 0

        #0- Uncased all text
        #1- Split in sentences
        sentencesAnalysis = spacyModule.getLanguage(textOpinion)

        #2- In each sentence find Aspect candidates (Sust, Adjective, compound n-grams in some lexicon as senticnet)
        if sentencesAnalysis == None:
            # Error in analysis
            return jsonify(
                id_opinion=idOpinion,
                id_campaign=idCampaign,
                response="ERROR",
                cause=sentencesAnalysis.cause_error)
        else:
           sentences = []
           polarity = spacyModule.getABSAModelClassify(textOpinion, acpectOpinion, app.appr, app.opt)
           if polarity != None:
                sentences.append({ 'sentence_text':textOpinion, 'aspect_text':acpectOpinion.text, 'aspect_type':acpectOpinion.type, 'polarity':polarity})

           return {
                "id_opinion": idOpinion,
                "id_campaign": idCampaign,
                "language":'es',
                "opinion_info": {
                    'sentences':sentences
                },
           }


@prc.route('/classify_opinion', methods=['POST', 'GET'])
def processClassifyOpinionEx():
    error = None
    #prc.logger.debug('Call request == in processEx')

    app = current_app._get_current_object()
    spacyModule = app.spacyModule
    #prc.logger.warning('A warning occurred (%d apples)', 42)
    #prc.logger.error('An error occurred')
    if request.method == 'GET':
        return "OK"
    elif request.method == 'POST':
        #prc.logger.debug( request.form)
        returned_data = request.get_json()
        if 'id_opinion' in request.form.keys() or returned_data == None:
            returned_data = request.form
        else:
            print(" ===> Send JSON values <===")
        idOpinion = returned_data['id_opinion']
        #prc.logger.debug('Opinion id ==> ')
        #prc.logger.debug(idOpinion)
        idCampaign  = returned_data['id_campaign']
        #prc.logger.debug('Campign id ==> ')
        #prc.logger.debug(idCampaign)

        textOpinion = returned_data['textopinion']
        #prc.logger.debug('Opinion text data ==> ')
        #prc.logger.debug(textOpinion)

        acpectOpinion = returned_data['aspectopinion']

        sentences = []
        try:
            sentencesAnalysis = spacyModule.getLanguage(textOpinion)
            if sentencesAnalysis == None:
                # Error in analysis
                return jsonify(
                    id_opinion=idOpinion,
                    id_campaign=idCampaign,
                    response="ERROR",
                    cause=sentencesAnalysis.cause_error)
            else:
                sentencesList = sentencesAnalysis.sentences
                language = sentencesAnalysis.language
                if language != 'en': #Is an error and return
                    return jsonify(
                        id_opinion=idOpinion,
                        id_campaign=idCampaign,
                        response="ERROR",
                        cause="Languague not available in Sentiment Analysis module : " + language )
                else:
                    #
                    #Transform to Complex JSOn object in python (Flask)
                    #Bibliografy:
                    #
                    # https://stackoverflow.com/questions/48510665/how-to-serialize-complex-objects-to-json-with-flask
                    #
                    # https://stackoverflow.com/questions/29283267/python-json-complex-objects-accounting-for-subclassing
                    #
                    polarity = getABSAModelClassify(textOpinion, acpectOpinion, app.tokenizer, app.appr, app.opt)
                    classification_value = 0;
                    if polarity != None:

                        classification_value = polarity.cpu().detach().numpy()
                        if len(classification_value) <= 0 or classification_value == None:
                            classification_value = 0
                        else:
                            classification_value = classification_value[0].item()
                        sentences.append(
                            {'sentence_text': textOpinion, 'aspect_text': acpectOpinion, 'aspect_type': 'ASPECT',
                             'polarity': classification_value})
                    if len (sentences) <= 0:
                        return jsonify(
                            id_opinion=idOpinion,
                            id_campaign=idCampaign,
                            response="ERROR",
                            cause="Not sentence information")
                    else:
                          return jsonify(
                            id_opinion= idOpinion,
                            id_campaign = idCampaign,
                            language = language,
                            opinion_class=classification_value,
                            opinion_sentence =sentences[0]
                          )
        except Exception as e:
            return jsonify(
                id_opinion=idOpinion,
                id_campaign=idCampaign,
                response="ERROR",
                cause=e)

            #Process of ABSA Opinion

        #1- Extract possible opinion in text

        #2- For each aspect in sentences
        #      Note: In text could have many sentences and in
        #            each sentences several possible aspect


        # examples:
        #
        # works well, and i am extremely happy to be back to an $T$.
        # apple os
        # 1
        #
        # i trust the $T$ at go sushi, it never disappoints.
        # people
        # 1
        #
        # the $T$ that came out were mediocre. portions of the
        # food
        # -1
        #
        # great food but the $T$ was dreadful !
        # service
        # 0

        #0- Uncased all text
        #1- Split in sentences

from itertools import starmap
def insertInformationInDataBase(isentence,idOpinion, idCampaign, language):

    #Insert sentences in data base
    id_opinion  = session['idOpinion'];
    id_campaing = session['idOpinion'];
    #isentence, idOpinion, idCampaign, language = element
    app = current_app._get_current_object()
    result = mongodb.insert_sentences(isentence.text, id_opinion, id_campaing, isentence.start, isentence.end, language)

    if result != None:
        print("****** Id ******** ", result.inserted_id)
        sentenceid = result.inserted_id

        # Insert entity in database
        for ent in isentence.entities:
            resultEntity = mongodb.insert_opinion_entity(ent.text, ent.label,sentenceid, id_opinion, ent.start_char,
                                                 ent.ent_char, language)

        # Insert asociated aspect
        for aspect in isentence.aspects:
            polarity = getABSAModelClassify(isentence.text, aspect.ext, app.tokenizer, app.appr, app.opt)

            resultAspect = mongodb.insert_opinion_aspect(aspect.text, polarity, sentenceid, id_opinion, aspect.start_char,
                                             aspect.ent_char, language)

        for aspect in isentence.aspects_noun:
            polarity = getABSAModelClassify(isentence.text, aspect.ext, app.tokenizer, app.appr, app.opt)
            resultAspect = mongodb.insert_opinion_aspect(aspect.text,polarity, sentenceid, id_opinion, aspect.start_char,
                                                 aspect.ent_char, language)

    #For each aspect classify with NLP computational model
    #and insert in databse

    pass


@prc.route('/classify_opinion_nlp', methods=['POST', 'GET'])
def processClassifyOpinionNLPEx():
    error = None
    #prc.logger.debug('Call request == in processEx')

    app = current_app._get_current_object()
    spacyModule = app.spacyModule
    #prc.logger.warning('A warning occurred (%d apples)', 42)
    #prc.logger.error('An error occurred')
    if request.method == 'GET':
        return "OK"
    elif request.method == 'POST':
        # Convert response data to json
        returned_data = request.get_json()
        if  'id_opinion' in request.form.keys() or returned_data == None:
            returned_data = request.form
        else:
            print (" ===> Send JSON values <===")
        #print("Called Data")
        #print(returned_data)

        print("Called Data")
        #prc.logger.debug( request.form)
        #idOpinion = request.form['id_opinion']
        idOpinion = returned_data['id_opinion']
        #prc.logger.debug('Opinion id ==> ')
        #prc.logger.debug(idOpinion)
        idCampaign  = returned_data['id_campaign']
        #prc.logger.debug('Campign id ==> ')
        #prc.logger.debug(idCampaign)

        textOpinion = returned_data['textopinion']
        #prc.logger.debug('Opinion text data ==> ')
        #prc.logger.debug(textOpinion)

        acpectOpinion = returned_data['aspectopinion']

        officianLanguage = ['es','en']
        sentences = []
        try:
            sentencesAnalysis = spacyModule.getProcessText(textOpinion)
            if sentencesAnalysis == None:
                # Error in analysis
                return jsonify(
                    id_opinion=idOpinion,
                    id_campaign=idCampaign,
                    response="ERROR",
                    cause=sentencesAnalysis.cause_error)
            else:
                sentencesList = sentencesAnalysis.sentences
                language = sentencesAnalysis.language
                if not  language in officianLanguage: #Is an error and return
                    return jsonify(
                        id_opinion=idOpinion,
                        id_campaign=idCampaign,
                        response="ERROR",
                        cause="Languague not available in Sentiment Analysis module : " + language )
                else:
                    #
                    #Transform to Complex JSOn object in python (Flask)
                    #Bibliografy:
                    #
                    # https://stackoverflow.com/questions/48510665/how-to-serialize-complex-objects-to-json-with-flask
                    #
                    # https://stackoverflow.com/questions/29283267/python-json-complex-objects-accounting-for-subclassing
                    #
                    if sentencesList != None:
                        val = list(starmap( insertInformationInDataBase, zip( sentencesList, idOpinion, idCampaign, language) ))
                    polarity = getABSAModelClassify(textOpinion, acpectOpinion, app.tokenizer, app.appr, app.opt)
                    classification_value = 0;
                    if polarity != None:

                        classification_value = polarity.cpu().detach().numpy()
                        if len(classification_value) <= 0 or classification_value == None:
                            classification_value = 0
                        else:
                            classification_value = classification_value[0].item()
                        sentences.append(
                            {'sentence_text': textOpinion, 'aspect_text': acpectOpinion, 'aspect_type': 'ASPECT',
                             'polarity': classification_value})
                    if len (sentences) <= 0:
                        return jsonify(
                            id_opinion=idOpinion,
                            id_campaign=idCampaign,
                            response="ERROR",
                            cause="Not sentence information")
                    else:
                          return jsonify(
                            id_opinion= idOpinion,
                            id_campaign = idCampaign,
                            language = language,
                            opinion_class=classification_value,
                            opinion_sentence =sentences[0]
                          )
        except Exception as e:
            return jsonify(
                id_opinion=idOpinion,
                id_campaign=idCampaign,
                response="ERROR",
                cause=e)

#####################################################################
#
#       Spacy Examples
#
######################################################################

# nlp = spacy.load('./en_core_web_md/en_core_web_md-2.1.0')
#
# doc = nlp("Next week I'll   be in Madrid.")
# for token in doc:
#     print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
#         token.text,
#         token.idx,
#         token.lemma_,
#         token.is_punct,
#         token.is_space,
#         token.shape_,
#         token.pos_,
#         token.tag_
#     ))
# #
# # print ("********* Sentence detection  ********")
# #
# # doc = nlp("These are apples. These are oranges.")
# #
# # for sent in doc.sents:
# #     print(sent)
# #
# # print ("********* Part Of Speech Tagging  ********")
# #
# # doc = nlp("Next week I'll be in Madrid.")
# # print([(token.text, token.tag_) for token in doc])
# #
# #
# # print ("********* Named Entity Recognition  ********")
# #
# # doc = nlp("Next week I'll be in Madrid.")
# # for ent in doc.ents:
# #     print(ent.text, ent.label_)
# #
# #
# # print ("********* IOB style tagging  ********")
# #
# # doc = nlp("Next week I'll be in Madrid.")
# # iob_tagged = [
# #     (
# #         token.text,
# #         token.tag_,
# #         "{0}-{1}".format(token.ent_iob_, token.ent_type_) if token.ent_iob_ != 'O' else token.ent_iob_
# #     ) for token in doc
# # ]
# #
# # print(iob_tagged)
# #
# # # In case you like the nltk.Tree format
# # print(conlltags2tree(iob_tagged))
# #
# #
# # print ("********* spaCy NER also has a healthy variety of entities. You can view the full list  ********")
# #
# # doc = nlp("I just bought 2 shares at 9 a.m. because the stock went up 30% in just 2 days according to the WSJ")
# # for ent in doc.ents:
# #     print(ent.text, ent.label_)
# #
# # print ("********* Let’s use displaCy to view a beautiful visualization of the Named Entity   ********")
# # doc = nlp('I just bought 2 shares at 9 a.m. because the stock went up 30% in just 2 days according to the WSJ')
# # displacy.render(doc, style='ent', jupyter=False)
# #
# # print ("********* spaCy automatically detects noun-phrases as well  ********")
# # doc = nlp("Wall Street Journal just published an interesting piece on crypto currencies")
# # for chunk in doc.noun_chunks:
# #     print(chunk.text, chunk.label_, chunk.root.text)
#
# print ("********* Dependency Parsing  ********")
# doc = nlp('Wall Street Journal just published an interesting piece on crypto currencies')
#
# for token in doc:
#     print ("Position: {0} == text: {1}".format(token.i,token.text))
#     print("{0}/{1} <--{2}-- {3}/{4}".format(
#         token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))
#
#
#
# # print ("********* Language Detector  ********")
# # nlp.add_pipe(LanguageDetector(), name="language_detector", last=True)
# # text = "This is English text. Er lebt mit seinen Eltern und seiner Schwester in Berlin. Yo me divierto todos los días en el parque. Je m'appelle Angélica Summer, j'ai 12 ans et je suis canadienne."
# # doc = nlp(text)
# # # document level language detection. Think of it like average language of document!
# # print(doc._.language)
# # # sentence level language detection
# # for i, sent in enumerate(doc.sents):
# #     print(sent, sent._.language)