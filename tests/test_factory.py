from flaskrapp.opinion_process import create_app


# def test_config():
#     assert not create_app().testing
#     assert create_app({'TESTING': True}).testing
#
# def test_register(client, app):
#     assert client.get('/wnlp/sentence_sppliter').status_code == 200
#
# def test_splitterSenteces_PostRequestSpanish(client, app):
#     assert client.get('/wnlp/sentence_sppliter').status_code == 200
#
#     response = client.post('/wnlp/sentence_sppliter', data={'id_opinion': '2314422',
#                                                             'id_campaign': '33211121',
#                                                              'textopinion':'Igualmente lograron el Oncoced, destinado a la terapia electroquímica de tumores sólidos para disminuir su volumen y convertirlos en operables. Ambos resultados continúan perfeccionándose en busca de nuevas prestaciones y el aumento de su seguridad y eficacia.'})
#
#     assert 'es' in response.json['language']
#     assert 2  == len(response.json['opinion_info']['sentences'])
#
# def test_splitterSenteces_PostRequestEnglish(client, app):
#         assert client.get('/wnlp/sentence_sppliter').status_code == 200
#
#         response = client.post('/wnlp/sentence_sppliter', data={'id_opinion': '2314422',
#                                                                 'id_campaign': '33211121',
#                                                                 'textopinion': 'This first traditional approach replicates the entire dataset from the remote storage to the local storage of each server for training. The copy process is easy with tools available online. This approach yields the highest I/O throughput as all data is local, maximizing the chance to keep all GPUs busy.'})
#
#         assert 'en' in response.json['language']
#         assert 3 == len(response.json['opinion_info']['sentences'])
#
# def test_getEntities_PostRequestEnglish(client, app):
#         assert client.get('/wnlp/entities_detector').status_code == 200
#
#         response = client.post('/wnlp/entities_detector', data={'id_opinion': '2314422',
#                                                                 'id_campaign': '33211121',
#                                                                 'textopinion': 'This first traditional approach replicates the entire dataset from the remote storage to the local storage of each server for training. The copy process is easy with tools available online. This approach yields the highest I/O throughput as all data is local, maximizing the chance to keep all GPUs busy.'})
#
#         assert 'en' in response.json['language']
#         assert 1 == len(response.json['opinion_info']['entities'])
#     #assert b'Title is required.' in response.data

def test_ClassifyOpinion_English(client, app):
    assert client.get('/wnlp/classify_opinion').status_code == 200

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

    response = client.post('/wnlp/classify_opinion', data={'id_opinion': '2314422',
                                                            'id_campaign': '33211121',
                                                            'textopinion': 'i trust the $T$ at go sushi, it never disappoints',
                                                             'aspectopinion':'people'})

    assert 'en' in response.json['language']
    #assert 4 == len(response.json['opinion_sentence'])





    ######
    #   Important we need to load weight in model for trained examples
    #   because now is random behaviour (i.e., Sometime 1 or 2 or 2)
    #   assert 1 != response.json['opinion_class'] !!!!Susccesfull result
    ####
    #assert 0 != response.json['opinion_class']

def test_ClassifyOpinionNLP_English(client, app):
    assert client.get('/wnlp/classify_opinion_nlp').status_code == 200

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

    response = client.post('/wnlp/classify_opinion_nlp', data={'id_opinion': '2314422',
                                                            'id_campaign': '33211121',
                                                            'textopinion': "Apple is looking at buying U.K. startup for $1 billion. But New York find people",
                                                             'aspectopinion':'people'})

    assert 'en' in response.json['language']
    #assert 4 == len(response.json['opinion_sentence'])


def test_ClassifyOpinionNLP_Spanish(client, app):
    assert client.get('/wnlp/classify_opinion_nlp').status_code == 200

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

    response = client.post('/wnlp/classify_opinion_nlp', data={'id_opinion': '2314422',
                                                            'id_campaign': '33211121',
                                                            'textopinion': 'Pedro es el que corre. En Madrid todos lo quieren',
                                                             'aspectopinion':'quieren'})

    assert 'es' in response.json['language']
    #assert 4 == len(response.json['opinion_sentence'])


# def test_hello(client):
#     response = client.get('/hello')
#     assert response.data == b'Hello, World!'