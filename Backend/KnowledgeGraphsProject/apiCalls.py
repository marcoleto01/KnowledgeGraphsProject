import time

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import nerTool
import spacy
import shutil

from Project import relationExtractor, graphBuilder, graphAnalysis, GPT_Api
from neo4j import GraphDatabase

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_trf")

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "14052001"))
session = driver.session()

#Define the variable for the GraphAnalysis
G = graphAnalysis.getGraph()


@app.route('/createProject', methods=['POST'])
def create_new_project():
    projectName = request.args.get('projectName')
    #Create a new folder into NER/Projects with the name of the project
    try:
        print(projectName)
        os.mkdir("Projects/" + projectName)
        return jsonify({"status": "success"})
    except OSError:
        return jsonify({"status": " Creation of the directory %s failed" % projectName})


@app.route('/deleteProject', methods=['DELETE'])
def delete_project():
    projectName = request.args.get('projectName')
    project_path = os.path.join("Projects", projectName)

    # Verifica se la cartella esiste e rimuovi tutto il contenuto
    if os.path.isdir(project_path):
        try:
            shutil.rmtree(project_path)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": f"Deletion of the directory {projectName} failed", "error": str(e)})
    else:
        return jsonify({"status": "Directory not found", "projectName": projectName})


@app.route('/addTextToProject', methods=['POST'])
def add_text_to_project():
    fileName = request.args.get('fileName')
    projectName = request.args.get('projectName')
    text = request.data.decode('utf-8')
    project_path = f"Projects/{projectName}"

    if not os.path.isdir(project_path):
        return jsonify({'error': f'La cartella del progetto "{projectName}" non esiste'}), 400

    textfile_directory = os.path.join(project_path, 'textFile')
    if not os.path.exists(textfile_directory):
        os.makedirs(textfile_directory)

    file_path = os.path.join(textfile_directory, f'{fileName}')

    try:
        with open(file_path, 'w', encoding="utf-8") as file:
            file.write(text)

        doc = nlp(text)
        nerTool.saveNerModel(f"Projects/{projectName}", doc)

        return jsonify({'message': f'File "{fileName}" creato con successo in "{textfile_directory}"'}), 200
    except Exception as e:
        return jsonify({'error': f'Errore durante la creazione del file: {str(e)}'}), 500


@app.route('/addCustomTerm', methods=['POST'])
def add_custom_terms():
    project_name = request.args.get('projectName')
    label = request.args.get('label')
    pattern = request.args.get('pattern')

    # Percorso della cartella del progetto
    project_path = f"Projects/{project_name}"

    # Verifica se la cartella del progetto esiste
    if not os.path.isdir(project_path):
        return jsonify({'error': f'La cartella del progetto "{project_name}" non esiste'}), 400

    # Percorso del file custom_terms.txt
    file_path = os.path.join(project_path, 'customTerms.txt')

    #Verifica se il file customTerms.txt esiste
    if not os.path.exists(file_path):
        #crea il file customTerms.txt
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('custom_terms = []')

    # Inizializza una lista di termini personalizzati
    custom_terms = []

    # Prova a leggere il file esistente
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Converte il contenuto in una lista di dizionari
                custom_terms = json.loads(content.replace('custom_terms = ', '').strip())
        except Exception as e:
            return jsonify({'error': f'Errore durante la lettura del file: {str(e)}'}), 500

    # Aggiungi il nuovo termine
    new_term = {"label": label, "pattern": pattern}
    custom_terms.append(new_term)

    # Scrivi nuovamente la lista aggiornata nel file
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('custom_terms = ')
            file.write(json.dumps(custom_terms, indent=4))

        doc = nerTool.loadNerModel(f"Projects/{project_name}", nlp)
        doc = nerTool.add_custom_entity(doc, pattern, label)
        nerTool.saveNerModel(f"Projects/{project_name}", doc)

        return jsonify({'message': f'Termini personalizzati aggiunti con successo in "{file_path}"'}), 200
    except Exception as e:
        return jsonify({'error': f'Errore durante la scrittura nel file: {str(e)}'}), 500


@app.route('/getCustomTerms', methods=['GET'])
def get_custom_terms_from_project():
    file_name = 'customTerms.txt'
    project_name = request.args.get('projectName')

    # Percorso della cartella del progetto
    project_path = f"Projects/{project_name}"

    # Verifica se la cartella del progetto esiste
    if not os.path.isdir(project_path):
        return jsonify({'error': f'La cartella del progetto "{project_name}" non esiste'}), 400

    # Percorso del file custom_terms.txt
    file_path = os.path.join(project_path, file_name)

    # Prova a leggere il file esistente
    if not os.path.exists(file_path):
        return jsonify({'error': f'Il file "{file_name}" non esiste nel progetto "{project_name}"'}), 404

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Converte il contenuto in una lista di dizionari
            custom_terms = json.loads(content.replace('custom_terms = ', '').strip())

        return jsonify({'custom_terms': custom_terms}), 200
    except Exception as e:
        return jsonify({'error': f'Errore durante la lettura del file: {str(e)}'}), 500


@app.route('/getAllProjects', methods=['GET'])
def get_all_projects():
    # Ottieni tutte le cartelle in Projects/ con nome progetto e data di creazione
    project_list = []
    for folder in os.listdir("Projects/"):
        folder_path = os.path.join("Projects/", folder)
        if os.path.isdir(folder_path):  # Verifica se è una cartella
            creation_time = os.path.getctime(folder_path)
            project_list.append({
                "projectName": folder,
                "date": time.ctime(creation_time)
            })
            print(type(creation_time))
    return jsonify({"status": "success", "data": project_list})


@app.route('/getTextFromProject', methods=['GET'])
def get_text_from_project():
    projectName = request.args.get('projectName')
    project_path = projectName
    file_path = f"{projectName}/textFile/"

    # Controlla se la cartella del progetto esiste
    if not os.path.isdir(project_path):
        return jsonify({"status": "error", "message": "Project not found"}), 404

    # Controlla se la cartella dei file di testo esiste
    if os.path.exists(file_path):
        # Ottieni una lista di tutti i file .txt nella cartella
        txt_files = [f for f in os.listdir(file_path) if f.endswith('.txt')]

        print(txt_files)

        if txt_files:
            # Prendi il primo file .txt trovato
            txt_file_path = os.path.join(file_path, txt_files[0])
            # Ottieni il testo dal file
            text = ""
            with open(txt_file_path, "r", encoding="utf-8") as text_file:
                text += text_file.read()
            # Ritorna a dictionary with text and fileName
            return jsonify({"status": "success", "data": text, "fileName": txt_files[0]})

        else:
            return jsonify({"status": "error", "message": "No .txt files found"}), 404
    else:
        return jsonify({"status": "error", "message": "File directory not found"}), 404


@app.route('/removeCustomTerm', methods=['DELETE'])
def remove_custom_term():
    project_name = request.args.get('projectName')
    pattern = request.args.get('pattern')

    # Percorso della cartella del progetto
    project_path = f"Projects/{project_name}"

    if not os.path.isdir(project_path):
        return jsonify({"status": "error", "message": "Project not found"}), 404

    file_name = 'customTerms.txt'

    # Percorso del file custom_terms.txt
    file_path = os.path.join(project_path, file_name)

    # Prova a leggere il file esistente
    if not os.path.exists(file_path):
        return jsonify({'error': f'Il file "{file_name}" non esiste nel progetto "{project_name}"'}), 404

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Converte il contenuto in una lista di dizionari
            custom_terms = json.loads(content.replace('custom_terms = ', '').strip())

        # Rimuovi il termine con il pattern specificato
        custom_terms = [term for term in custom_terms if term['pattern'] != pattern]

        # Scrivi nuovamente la lista aggiornata nel file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('custom_terms = ')
            file.write(json.dumps(custom_terms, indent=4))

        doc = nerTool.loadNerModel(f"Projects/{project_name}", nlp)
        doc = nerTool.remove_word_from_entity(doc, pattern)
        nerTool.saveNerModel(f"Projects/{project_name}", doc)

        return jsonify({'message': f'Termine "{pattern}" rimosso con successo da "{file_path}"'}), 200

    except Exception as e:
        return jsonify({'error': f'Errore durante la lettura/scrittura del file: {str(e)}'}), 500


@app.route('/loadNerModel', methods=['GET'])
def loadNerModel():
    projectName = request.args.get('projectName')
    # Open the document "ner.spacy" in the projectName folder
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    return jsonify({"status": "success"})


@app.route('/getCommonWords', methods=['GET'])
def getCommonWords():
    projectName = request.args.get('projectName')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    stop_words = nlp.Defaults.stop_words
    common_words = nerTool.get_common_words(doc, stop_words)
    return jsonify({"commonWords": common_words})


@app.route('/getWordAnalysis', methods=['GET'])
def getWordAnalysis():
    projectName = request.args.get('projectName')
    word = request.args.get('word')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    word_analysis = nerTool.word_analysis(doc, word)
    return jsonify({"wordAnalysis": word_analysis})


@app.route('/getEntities', methods=['GET'])
def getEntitiesFromText():
    projectName = request.args.get('projectName')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    entities = nerTool.getEntities(doc)
    return jsonify({"entities": entities})


@app.route('/getSentenceHTML', methods=['GET'])
def getSentenceHTML():
    projectName = request.args.get('projectName')
    sentence = request.args.get('sentence')
    doc = nerTool.loadNerModel(f"{projectName}", nlp)
    html = nerTool.getSentenceHTML(sentence, doc)
    return jsonify({"SentenceHtml": html})


@app.route('/addEntity', methods=['POST'])
def addEntity():
    projectName = request.args.get('projectName')
    pattern = request.args.get('pattern')
    label = request.args.get('label')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    doc = nerTool.add_custom_entity(doc, pattern, label)
    nerTool.saveNerModel(f"Projects/{projectName}", doc)
    return jsonify({"status": "success"})


@app.route('/getNumberOfEntities', methods=['GET'])
def getNumberOfEntities():
    projectName = request.args.get('projectName')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    entitiesNumber = nerTool.getEntitiesNumber(doc)
    return jsonify({"entitiesNumber": entitiesNumber})


@app.route('/addPriorEntity', methods=['POST'])
def addPriorEntity():
    projectName = request.args.get('projectName')
    pattern = request.args.get('pattern')
    label = request.args.get('label')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    doc = nerTool.add_prior_entity(doc, pattern, label)
    nerTool.saveNerModel(f"Projects/{projectName}", doc)
    return jsonify({"status": "success"})


@app.route('/addPriorEntityArray', methods=['POST'])
def addPriorEntityArray():
    projectName = request.args.get('projectName')
    pattern_array = request.json.get('patterns', [])
    pattern = request.args.get('pattern')
    label = request.args.get('label')
    print(pattern_array)

    pattern_array.append(pattern)
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    for pattern in pattern_array:
        doc = nerTool.add_prior_entity(doc, pattern, label)
    nerTool.saveNerModel(f"Projects/{projectName}", doc)

    return jsonify({"status": "success"})


@app.route('/copyEntity', methods=['POST'])
def copyEntity():
    projectName = request.args.get('projectName')
    sourceProjectName = request.args.get('sourceProject')
    docProject = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    docSource = nerTool.loadNerModel(f"Projects/{sourceProjectName}", nlp)
    docProject = nerTool.copy_entity_from_doc(docProject, docSource)
    nerTool.saveNerModel(f"Projects/{projectName}", docProject)
    return jsonify({"status": "success"})


@app.route('/addAllCustomTerms', methods=['POST'])
def addAllCustomTerm():
    projectName = request.args.get('projectName')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    doc = nerTool.add_all_custom_terms(projectName, doc)
    nerTool.saveNerModel(f"Projects/{projectName}", doc)
    return jsonify({"status": "success"})


@app.route('/getEntitiesCorrelatedToCustomTerms', methods=['GET'])
def getEntitiesCorrelatedToCustomTerms():
    projectName = request.args.get('projectName')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    entities = nerTool.get_entities_correlated_to_custom_terms(doc, projectName)
    return jsonify({"entities": entities})


@app.route('/addEnergyValueMeasurement', methods=['POST'])
def addEnergyValueMeasurement():
    projectName = request.args.get('projectName')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    doc = nerTool.include_measure_unit_in_label(doc)
    nerTool.saveNerModel(f"Projects/{projectName}", doc)
    return jsonify({"status": "success"})


@app.route('/removeWordFromEntities', methods=['DELETE'])
def RemoveWordFromEntities():
    projectName = request.args.get('projectName')
    word = request.args.get('word')
    entity = request.args.get('entity')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    doc = nerTool.remove_word_from_entity_2(doc, word, entity)
    nerTool.saveNerModel(f"Projects/{projectName}", doc)
    return jsonify({"status": "success"})


@app.route('/removeEntity', methods=['DELETE'])
def RemoveEntity():
    projectName = request.args.get('projectName')
    entity = request.args.get('entity')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    doc = nerTool.remove_entity(doc, entity)
    nerTool.saveNerModel(f"Projects/{projectName}", doc)
    return jsonify({"status": "success"})


@app.route('/getWordsFromEntity', methods=['GET'])
def getWordFromEntity():
    projectName = request.args.get('projectName')
    entity = request.args.get('entity')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    words = nerTool.get_words_from_entities(doc, entity)
    return jsonify({"words": words})


@app.route('/changeEntityLabel', methods=['POST'])
def changeEntityLabel():
    projectName = request.args.get('projectName')
    oldLabel = request.args.get('oldLabel')
    newLabel = request.args.get('newLabel')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    doc = nerTool.change_entity_label(doc, oldLabel, newLabel)
    nerTool.saveNerModel(f"Projects/{projectName}", doc)
    return jsonify({"status": "success"})


@app.route('/getEntitiesFromDoc', methods=['GET'])
def getEntitiesFromDoc():
    projectName = request.args.get('projectName')
    #Take dox from the project
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    entities, percentage = nerTool.get_entities_term(doc)
    return jsonify({"entities": entities, "percentage": percentage})


@app.route('/getHtmlByTerm', methods=['GET'])
def getHtmlByTerm():
    projectName = request.args.get('projectName')
    term = request.args.get('term')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    html = nerTool.get_html_by_term(doc, term)
    return jsonify({"html": html})


@app.route('/getModelNames', methods=['GET'])
def getModelNames():
    # Return the names of the models in the folder
    model_list = []
    for folder in os.listdir("Models/"):
        folder_path = os.path.join("Models/", folder)
        if os.path.isdir(folder_path):  # Verifica se è una cartella
            creation_time = os.path.getctime(folder_path)
            model_list.append(folder)
            print(type(creation_time))
    return jsonify({"status": "success", "data": model_list})


@app.route('/getTaggerProjects', methods=['GET'])
def getTaggerProjects():
    # Return the names of the models in the folder
    tagger_list = []
    for folder in os.listdir("Tagging/"):
        folder_path = os.path.join("Tagging/", folder)
        if os.path.isdir(folder_path):
            creation_time = os.path.getctime(folder_path)
            tagger_list.append({
                "projectName": folder,
                "date": time.ctime(creation_time)
            })

    return jsonify({"status": "success", "data": tagger_list})


@app.route('/deleteTaggerProject', methods=['DELETE'])
def deleteTaggerProject():
    projectName = request.args.get('projectName')
    project_path = os.path.join("Tagging", projectName)

    # Verifica se la cartella esiste e rimuovi tutto il contenuto
    if os.path.isdir(project_path):
        try:
            shutil.rmtree(project_path)
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": f"Deletion of the directory {projectName} failed", "error": str(e)})
    else:
        return jsonify({"status": "Directory not found", "projectName": projectName})


@app.route('/tagText', methods=['POST'])
def tagText():
    fileName = request.args.get('fileName')
    #split fileName with . and take the first element
    fileName = fileName.split(".")[0]
    modelName = request.args.get('modelName')
    text = request.data.decode('utf-8')
    #Create a new folder into NER/Tagging with the name of fileName
    try:
        os.mkdir("Tagging/" + fileName)
    except OSError:
        return jsonify({"status": " Creation of the directory %s failed" % fileName})

    #Create a new folder into NER/Tagging with the name of textFile
    try:
        os.mkdir(f"Tagging/{fileName}/textFile")
    except OSError:
        return jsonify({"status": " Creation of the directory %s failed" % fileName})

    #Save the text in the folder in a .txt file with the name of the fileName
    with open(f"Tagging/{fileName}/textFile/{fileName}.txt", 'w', encoding="utf-8") as file:
        file.write(text)

    #Tag the text with the model
    doc = nerTool.useTaggingModel(modelName, text)
    doc = nerTool.remove_leading_articles(doc)
    doc = nerTool.enhance_entities(doc)

    #Include the entities from the trf model to the doc
    doc = nerTool.include_base_entities(doc, text, nlp)

    #Save the doc in the folder
    nerTool.saveNerModel(f"Tagging/{fileName}", doc)

    return jsonify({"status": "success"})


@app.route('/getRelationsFromProject', methods=['GET'])
def getRelationsFromProject():
    projectName = request.args.get('projectName')
    doc = nerTool.loadNerModel(f"Projects/{projectName}", nlp)
    relations = relationExtractor.extract_relations(doc)
    print(relations)
    return jsonify({"relations": relations})


import json


@app.route('/buildKnowledgeGraphFromProject', methods=['POST'])
def buildKnowledgeGraphFromProject():
    projectName = request.args.get('projectName')
    query = "SHOW DATABASES"
    result = session.run(query)
    databases = [record["name"] for record in result if record["name"] != "system"]
    if projectName in databases:
        return jsonify({"status": "error", "message": "Project already exists in the database"}), 400
    session.run(f"CREATE DATABASE {projectName}")
    project_session = driver.session(database=projectName)
    data = request.json
    array = data.get('doc', [])
    for document in array:
        print("Processing document:", document)
        doc = nerTool.loadNerModel(f"Tagging/{document}", nlp)
        relations = relationExtractor.extract_relations(doc, document)
        print(relations)
        graphBuilder.build_graph(relations, doc, project_session, driver, document)
    # Add the doc to 'graphFilesMapping.json'
    with open('graphFilesMapping.json', 'r') as file:
        data = json.load(file)
        data[projectName] = array
    with open('graphFilesMapping.json', 'w') as file:
        json.dump(data, file)

    return jsonify({"status": "success"})


@app.route('/processQuery', methods=['GET'])
def processQuery():
    query = request.args.get('query')
    projectName = request.args.get('projectName')
    print(f"Query: {query}, Project: {projectName}")
    session = driver.session(database=projectName)
    print("Ciao")
    graph_data = graphBuilder.execute_query(session, query)

    print("GraphData:", graph_data)

    if graph_data != None:
        print(graph_data["nodes"])
        print(graph_data["edges"])
        return jsonify({"nodes": graph_data["nodes"], "edges": graph_data["edges"],
                        "relation_texts": graph_data["relation_texts"]})

    return jsonify({"nodes": [], "edges": [], "relation_texts": []})


@app.route('/processQueryByForm', methods=['POST'])
def processQueryByForm():
    projectName = request.args.get('projectName')
    data = request.get_json()
    queryData = data.get('query', [])
    queryDates = data.get('dates', [])
    query = graphBuilder.createQueryByData(queryData, queryDates)
    print(query)
    session = driver.session(database=projectName)
    graph_data = graphBuilder.execute_query(session, query)
    return jsonify(
        {"nodes": graph_data["nodes"], "edges": graph_data["edges"], "relation_texts": graph_data["relation_texts"],
         "query": query, })


@app.route('/getQueryChoose', methods=['GET'])
def get_query_options():
    projectName = request.args.get('projectName')
    # Open graphFilesMapping.json and take the documents in the projectName
    if not os.path.exists('graphFilesMapping.json'):
        return jsonify({"status": "error", "message": "graphFilesMapping.json not found"}), 404

    with open('graphFilesMapping.json', 'r') as file:
        data = json.load(file)
        array = data.get(projectName, [])

    return_dict = {}
    return_dates = []

    for document in array:
        # Take doc.spacy in Tagging/document
        doc_path = f"Tagging/{document}"
        if not os.path.exists(doc_path):
            continue
        doc = nerTool.loadNerModel(doc_path, nlp)
        query_dict, dates = graphBuilder.getQueryChoose(doc, return_dict, return_dates)
        for key, values in query_dict.items():
            if key in return_dict:
                return_dict[key].extend([value for value in values if value not in return_dict[key]])
            else:
                return_dict[key] = values
        return_dates.extend([date for date in dates if date not in return_dates])

    return jsonify({"query": return_dict, "dates": return_dates})


@app.route('/getGraphProjects', methods=['GET'])
def getGraphProjects():
    query = "SHOW DATABASES"
    result = session.run(query)
    databases = [record["name"] for record in result if record["name"] != "system"]
    return jsonify({"projects": databases})


@app.route('/removeGraphProject', methods=['DELETE'])
def removeGraphProject():
    projectName = request.args.get('projectName')
    query = f"DROP DATABASE {projectName}"
    session.run(query)
    return jsonify({"status": "success"})


@app.route('/getTaggedTextFiles', methods=['GET'])
def getTaggedTextFiles():
    # Take the name of directories in /Tagging
    tagging_dir = "Tagging"
    if not os.path.isdir(tagging_dir):
        return jsonify({"status": "error", "message": "Tagging directory not found"}), 404

    tagged_files = []
    for folder in os.listdir(tagging_dir):
        folder_path = os.path.join(tagging_dir, folder)
        if os.path.isdir(folder_path):
            tagged_files.append(folder)

    return jsonify({"status": "success", "taggedFiles": tagged_files})


@app.route('/getGraphInformation', methods=['GET'])
def getGraphInformation():
    info = graphAnalysis.getGraphInformation(G)
    return jsonify({"status": "success", "data": info})


@app.route('/getNodeInformation', methods=['GET'])
def getNodeInformation():
    nodeString = request.args.get('nodeText')
    info = graphAnalysis.getNodeInformation(G, nodeString)
    return jsonify({"status": "success", "data": info})


@app.route('/getGraphInformationByCommunity', methods=['GET'])
def getGraphInformationByCommunity():
    community_id = request.args.get('communityId')
    info = graphAnalysis.getGraphInformationByCommunity(community_id)
    return jsonify({"status": "success", "data": info})


@app.route('/getLouvainCommunities', methods=['GET'])
def getLouvainCommunities():
    communities = graphAnalysis.getLouvainCommunities()
    return jsonify({"status": "success", "data": communities})


@app.route('/getLouvainCommunityInfo', methods=['GET'])
def getLouvainCommunityInfo():
    community_id = request.args.get('communityId')
    info = graphAnalysis.getLouvainCommunityInfo(community_id)
    return jsonify({"status": "success", "data": info})


@app.route('/getNodesInfoByGraph', methods=['GET'])
def getNodesInfoByGraph():
    community_id = request.args.get('communityId')
    session = driver.session(database='cleanedgraph2')
    query = graphAnalysis.getAllCommunityNodesQuery(community_id)
    print("Query:", query)
    graph_data = graphBuilder.execute_query_without_text(session, query)
    print("GraphData:", graph_data)
    if graph_data != None:
        return jsonify({"nodes": graph_data["nodes"], "edges": graph_data["edges"]})

    return jsonify({"nodes": [], "edges": [], "relation_texts": []})


@app.route('/getAllModels', methods=['GET'])
def getAllModels():
    # Return the names of the models in the folder
    model_list = []
    for file in os.listdir("GraphAnalysis/energyReportsGraph/Models/"):
        model_list.append(file)

    return jsonify({"status": "success", "data": model_list})


@app.route('/getAllPossibleLinkQueryChoose', methods=['GET'])
def getAllPossibleLinkQueryChoose():
    possibleChoose, entity_families = graphAnalysis.getAllPossibleLinkQueryChoose()
    return jsonify({"status": "success", "possibleChoose": possibleChoose, "entity_families": entity_families})


@app.route('/createCustomStrategy', methods=['POST'])
def createCustomStrategy():
    #create_new_custom_model(uniformWeight=0.25, hardWeight=0.25, centralityWeight=0.25)
    global creatingModel
    creatingModel = True
    uniformWeight = request.args.get('uniformWeight')
    hardWeight = request.args.get('hardWeight')
    centralityWeight = request.args.get('centralityWeight')

    graphAnalysis.create_new_custom_model(uniformWeight, hardWeight, centralityWeight)

    creatingModel = False

    return jsonify({"status": "success"})


@app.route('/getCreatingModel', methods=['GET'])
def getCreatingModel():
    return jsonify({"status": "success", "data": creatingModel})


@app.route('/getNodePrediction', methods=['POST'])
def getNodePrediction():
    global searchingLinks
    model = request.args.get('model')
    threshold = request.args.get('threshold')
    body = request.get_json()
    nodes = body.get('nodes')

    print("Threshold:", threshold)
    print("Model:", model)

    print(f"Entro in getNodePrediction con model: {model} e nodes: {nodes}")
    searchingLinks = True

    prediction = graphAnalysis.node_prediction(model, nodes, threshold)
    #print("Prediction:", prediction)
    prediction_text = []
    for relation in prediction:
        prediction_text.append(relation['relation'])
    if len(prediction_text) > 0:
        #Concatenate element in prediction_text until the length is less than 19500
        prediction_API_input = ""
        #print("Prediction text:", prediction_text)
        for element in prediction_text:
            if len(prediction_API_input) + len(element) < 19500:
                prediction_API_input += element
            else:
                break
        elaborate_prediction = GPT_Api.elaborate_prediction(prediction_API_input)
        print("Elaborate prediction:", elaborate_prediction)
    else:
        elaborate_prediction = ""
    #elaborate_prediction = GPT_Api.elaborate_prediction(prediction)
    searchingLinks = False

    return jsonify({"status": "success", "data": prediction, "elaborate_prediction": elaborate_prediction})


@app.route('/getSearchingLinks', methods=['GET'])
def getSearchingLinks():
    return jsonify({"status": "success", "data": searchingLinks})


#Variable to check when a model is ready
creatingModel = False

#Searching link
searchingLinks = False


@app.route('/getReportByCommunity', methods=['GET'])
def getReportByCommunity():
    community_id = request.args.get('communityId')
    session = driver.session(database='cleanedgraph2')
    texts = graphAnalysis.getCommunityReport(community_id, session)
    text_to_chat = ""
    for text in texts:
        text_to_chat += text + "\n"
    print("Text to chat: ",text_to_chat)
    info = GPT_Api.elaborate_community_report_text(texts)
    return jsonify({"status": "success", "data": info})
