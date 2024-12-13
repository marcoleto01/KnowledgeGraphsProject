from neo4j import GraphDatabase
import string


not_entity_label = ["PERCENT", "DATE", "TIME", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "WORK_OF_ART"]

'''
Il metodo get_entities_from_subject prende in input un soggetto e restituisce un dizionario che 
mappa tutte le entities che contiene.
'''
def get_entities_from_subject(subject):
    entity_text = ""
    entity_label = ""

    labels = []

    for token in subject:
        if token.ent_iob_ != "O":
            if token.ent_type_ == entity_label or entity_label == "":
                entity_text += token.text + " "
                entity_label = token.ent_type_
            elif entity_label != "" and token.ent_type_ != entity_label:
                labels.append({"entity": entity_text, "label": entity_label})
                entity_text = token.text + " "
                entity_label = token.ent_type_
        elif entity_text != "":
            labels.append({"entity": entity_text, "label": entity_label})
            entity_text = ""
            entity_label = ""

    # Aggiunta ultima entità trovata
    if entity_text != "":
        labels.append({"entity": entity_text, "label": entity_label})

    subject_text = ""
    for token in subject:
        subject_text += token.text + " "

    return {"subject": subject_text, "entities": labels}



'''
Il metodo get_entities_from_adp prende in input una lista di preposizioni e restituisce un dizionario che
mappa tutte le entities che contiene.
'''
def get_entities_from_adp(adps):
    labels = []
    for adp in adps:
        adp_text = ""
        adp_label = ""
        adp_prop = ""
        i = 0
        while i < len(adp):
            if i == 0:
                if adp[i].pos_ == "ADP" or adp[i].pos_ == "SCONJ":
                    adp_prop += adp[i].text
                i += 1
                while i < len(adp) and (adp[i].pos_ != "ADP" or adp[i].pos_ != "SCONJ"):

                    adp_text += adp[i].text + " "
                    if adp[i].ent_iob_ != "O":
                        adp_label = adp[i].ent_type_
                    i += 1
                # Add the adp properties
                if adp_label != "":
                    adp_prop += adp_label
                same_prop = 1
                for label in labels:
                    if label["properties"] == adp_prop:
                        same_prop += 1
                if same_prop != 1:
                    # Add f"{same_prop}°" in the start of the adp properties
                    adp_prop = f"{adp_prop}n{same_prop}"
                if (adp_text != ""):
                    if " " in adp_prop:
                        adp_prop = adp_prop.replace(" ", "_")
                    labels.append({"adp_text": adp_text, "label": adp_label, "properties": adp_prop})
                    adp_text = ""
                    adp_label = ""
                    adp_prop = ""

    return labels

"""Remove punctuation and replace with spaces."""
def clean_text(text):
    return ''.join('' if char in string.punctuation and char != "%" else char for char in text)

"""
Clean property key to make it valid in Neo4j. Convert special characters to descriptive text while keeping numbers.
    """
def clean_property_key(key):

    # Special character mappings
    char_map = {
        '/': '_per_',
        '\\': '_backslash_',
        '.': '_dot_',
        '-': '_dash_',
        '=': '_equals_',
        '@': '_at_',
        '#': '_hash_',
        '^': '_caret_',
        '&': '_and_',
        '*': '_star_',
        '(': '_leftparen_',
        ')': '_rightparen_',
        '[': '_leftbracket_',
        ']': '_rightbracket_',
        '{': '_leftbrace_',
        '}': '_rightbrace_',
        '|': '_pipe_',
        '?': '_question_',
        '!': '_exclamation_',
        '<': '_less_',
        '>': '_greater_',
        ',': '_comma_',
        ':': '_colon_',
        ';': '_semicolon_',
        '`': '_backtick_',
        '~': '_tilde_',
        "'": '_quote_',
        '"': '_doublequote_',
    }

    result = key
    for char, replacement in char_map.items():
        result = result.replace(char, replacement)

    # Ensure the key starts with a letter (Neo4j requirement)
    if result and not result[0].isalpha():
        result = 'prop_' + result

    return result



def standardize_entity_text(dict, isAdp):
    # print("Inizio standardizzazione")
    preserve_case_labels = {'FAC', 'GPE', 'ORG', 'WORK_OF_ART', 'LOC', 'NORP'}
    if not isAdp:
        for entity in dict["entities"]:
            if entity["label"] not in preserve_case_labels:
                text = entity["entity"]
                text_lower = text.lower()
                entity["entity"] = text_lower
                dict["subject"] = dict["subject"].replace(text, text_lower)
    else:
        for entity in dict:
            if entity["label"] not in preserve_case_labels:
                text = entity["adp_text"]
                if isinstance(text, str):
                    text_lower = text.lower()
                elif isinstance(text, int):
                    text_lower = str(text)
                entity["adp_text"] = text_lower

    return dict


'''
Il metodo build_Cypher_query_adps prende in input un soggetto, un verbo, un oggetto e una lista di preposizioni e restituisce
una lista di query Cypher per creare i nodi e le relazioni nel database di Neo4j.
'''
def build_Cypher_query_adps(subject, verb, obj, adps):
    print(f"Subject: {subject}, Verb: {verb}, Object: {obj}, Adps: {adps}")

    # print("Inizio query")
    queries = []

    # Prepare subject labels
    subject_labels = ['Node'] + [entity['label'].strip() for entity in subject['entities']]
    subject_labels_str = ':' + ':'.join(subject_labels)

    # Create subject node with cleaned text
    subject_text = clean_text(subject['subject'].strip())
    subject_query = f"MERGE (s{subject_labels_str} {{text: '{subject_text}'}})"
    queries.append(subject_query)

    # Handle object nodes and relationships
    if obj["subject"]:
        object_query = ""
        object_text = clean_text(obj["subject"].strip())
        # Create object node with cleaned text

        if len(obj["entities"]) != 0:
            object_label = obj["entities"][0]["label"].strip()
            object_query = f"MERGE (o:Node:{object_label} {{text: '{object_text}'}})"
        else:
            object_query = f"MERGE (o:Node {{text: '{object_text}'}})"
        queries.append(object_query)

        # Create relationship with cleaned properties from adps
        properties = {clean_property_key(adp["properties"]): clean_text(adp["adp_text"].strip())
                      for adp in adps if adp["properties"].strip()}  # Skip empty property keys
        props_str = ", ".join([f"{k}: '{v}'" for k, v in properties.items()])

        if len(obj["entities"]) != 0:
            object_label = obj["entities"][0]["label"].strip()
            relationship_query = f"""MATCH (s{subject_labels_str} {{text: '{subject_text}'}})
    MATCH (o:Node:{object_label} {{text: '{object_text}'}})
    MERGE (s)-[r:{verb} {{{props_str}}}]->(o)"""
            queries.append(relationship_query)
        else:
            relationship_query = f"""MATCH (s{subject_labels_str} {{text: '{subject_text}'}})
    MATCH (o:Node {{text: '{object_text}'}})
    MERGE (s)-[r:{verb} {{{props_str}}}]->(o)"""
            queries.append(relationship_query)

    else:
        # Reflexive relationship with cleaned properties
        properties = {clean_property_key(adp["properties"]): clean_text(adp["adp_text"].strip())
                      for adp in adps if adp["properties"].strip()}  # Skip empty property keys
        props_str = ", ".join([f"{k}: '{v}'" for k, v in properties.items()])

        relationship_query = f"""MATCH (s{subject_labels_str} {{text: '{subject_text}'}})
MERGE (s)-[r:{verb} {{{props_str}}}]->(s)"""
        queries.append(relationship_query)

    return queries

'''
METODI DI TEST
'''

def get_all_node_names(session):
    query = """
    MATCH (n)
    WHERE n.text IS NOT NULL
    RETURN n.text AS text, labels(n) AS labels
    """

    result = session.run(query)
    node_dict = {record["text"]: record["labels"] for record in result}
    return node_dict


def check_database_nodes(session):
    """
    Verifica la presenza di nodi nel database
    """
    query = """
    MATCH (n)
    RETURN COUNT(n) as count
    """
    result = session.run(query)
    count = result.single()["count"]
    print(f"Numero totale di nodi: {count}")

    # Vediamo anche i tipi di nodi presenti
    query_labels = """
    MATCH (n)
    RETURN DISTINCT labels(n) as labels
    """
    result = session.run(query_labels)
    labels = [record["labels"] for record in result]
    print(f"Tipi di nodi presenti: {labels}")


def check_node_properties(session):
    """
    Verifica le proprietà presenti nei nodi
    """
    query = """
    MATCH (n)
    WITH DISTINCT keys(n) as props
    RETURN props
    """
    result = session.run(query)
    properties = [record["props"] for record in result]
    print(f"Proprietà presenti nei nodi: {properties}")


def check_names_variations(session):
    """
    Controlla diverse varianti della proprietà name
    """
    query = """
    MATCH (n)
    WHERE n.text IS NOT NULL 
    RETURN n.text as name1
    LIMIT 5
    """
    result = session.run(query)
    return [dict(record) for record in result]


def get_node_labels(session, node_id):
    query = """
    MATCH (n)
    WHERE id(n) = $node_id
    RETURN labels(n) AS labels
    """
    result = session.run(query, node_id=node_id).single()
    return result["labels"] if result else None  # Restituisce le labels o None


"""
Genera una query Cypher per collegare due nodi che condividono una stessa entità con una relazione 'IS_EXTENDED'.
"""
def crea_query_collega_nodi(s1, s2):
    query = f"""
    MATCH (n1 {{text: '{s1}'}}), (n2 {{text: '{s2}'}})
    MERGE (n1)-[:IS_EXTENDED]->(n2)
    """
    return query.strip()


"""
Il metodo get_extended_relations prende in input un dizionario di nodi e un array di entità e restituisce
il numero di relazioni 'IS_EXTENDED' create.
"""
def get_extended_relations(session, node_dict, entity_array):
    count = 0

    for node_name in node_dict:
        if node_name in entity_array:
            for contain_node_name in node_dict:
                if node_name in contain_node_name and node_name != contain_node_name and all(
                        elem in node_dict[contain_node_name] for elem in node_dict[node_name]):
                    query = crea_query_collega_nodi(node_name, contain_node_name)
                    session.run(query)
                    count += 1




'''
Il metodo build_graph prende come parametri:
 -un vettore di relazioni precedentemente estratte da un testo
 -il file doc.spacy che contiene le entità riconosciute dal modello NER
 -la sessione di Neo4j
 -il driver di Neo4j
 -il nome del progetto, che identifica il database in cui inserire i nodi e le relazioni
 
e si occupa di generare partendo dalle relazioni le diverse query Cypher per creare i nodi e le relazioni all'interno del database
'''
def build_graph(relations, doc, session, driver, projectName):
    for relations_group in relations:
        for relation in relations_group:
            subjects = relation[0]
            entities_subject = get_entities_from_subject(subjects)
            if len(entities_subject["entities"]) == 0:
                continue
            else:
                for label in entities_subject["entities"]:
                    if label["label"] not in not_entity_label:
                        import_label = True
            verb_text = ""
            verb = relation[1]
            if isinstance(verb, list):
                for v in verb:
                    verb_text += v.text + " "
            else:
                verb_text = verb

            # Remove last empty space in
            verb_text = verb_text.strip()
            verb_text = verb_text.replace(" ", "_")
            verb_text = verb_text.replace("/", "")

            objects = relation[2]
            entities_obj = get_entities_from_subject(objects)

            adp_relations = get_entities_from_adp(relation[3])

            adp_relations.append({"adp_text": relation[4], "label": "", "properties": "sentencesIndex"})

            # Method to unify same word syntax
            entities_subject = standardize_entity_text(entities_subject, False)
            entities_obj = standardize_entity_text(entities_obj, False)
            adp_relations = standardize_entity_text(adp_relations, True)

            adp_relations.append({"adp_text": projectName, "label": "", "properties": "documentName"})

            print(f"Adp relations: {adp_relations}")

            queries = build_Cypher_query_adps(entities_subject, verb_text, entities_obj, adp_relations)

            for q in queries:
                result = session.run(q)

    entity_array = []
    for ent in doc.ents:
        if ent.label_ not in ["TIME", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "PERCENT"]:
            entity_array.append(ent.text)

    node_dict = get_all_node_names(session)
    get_extended_relations(session, node_dict, entity_array)


import spacy
import nerTool
import json


'''
Il metodo getRelationTextFromDoc prende in input il nome del documento e l'indice della fase e restituisce il testo correlato
'''
def getRelationTextFromDoc(document_name, phase_index):
    #Open phasesMapping.json from Tagging/document_name
    with open(f"Tagging/{document_name}/phasesMapping.json") as f:
        data = json.load(f)

    return data[phase_index]


'''
Il metodo execute query, esegue la query passata come parametro e parsa il risultato in due dizionari,
formato che viene poi passato e gestito dal frontend per la corretta visualizazione del grafo.
'''
def execute_query(session, query):
    try:
        result = session.run(query)
        # Creare liste per nodi e archi
        nodes = {}
        edges = []
        relation_texts = []

        phase_already_taken = []

        for record in result:
            relation_dict = {}
            relation = {}
            phase_index = -1
            document_name = ""

            #print(record)

            # Itera su tutti gli elementi nel record
            for value in record.values():

                # Gestisce i nodi
                if hasattr(value, 'labels'):
                    node_id = value.id  # ID univoco del nodo
                    if node_id not in nodes:
                        nodes[node_id] = {
                            'id': node_id,
                            'labels': list(value.labels),
                            'properties': dict(value)
                        }

                # Gestisce le relazioni
                elif hasattr(value, 'type'):
                    # Ottiene gli ID dei nodi di origine e destinazione
                    start_node = value.start_node
                    end_node = value.end_node

                    relation['subject'] = start_node['text']
                    if (relation['subject'] != end_node['text']):
                        relation['object'] = end_node['text']

                    # Aggiunge i nodi se non esistono già
                    if start_node.id not in nodes:
                        nodes[start_node.id] = {
                            'id': start_node.id,
                            'labels': list(start_node.labels),
                            'properties': dict(start_node)
                        }

                    if end_node.id not in nodes:
                        nodes[end_node.id] = {
                            'id': end_node.id,
                            'labels': list(end_node.labels),
                            'properties': dict(end_node)
                        }
                    new_value_dict = {}
                    for key in value:
                        if key == 'documentName':
                            document_name = value[key]
                        elif key == 'sentencesIndex':
                            phase_index = value[key]
                        else:
                            new_value_dict[key] = value[key]

                    # Crea l'arco
                    edge = {
                        'id': value.id,
                        'source': start_node.id,
                        'target': end_node.id,
                        'type': value.type,
                        'properties': dict(new_value_dict)
                    }
                    edges.append(edge)

                    relation['verb'] = value.type
                    relation['properties'] = dict(value)

            relation_dict['relation'] = relation
            #Get the relation text
            if document_name != "" and phase_index != -1:
                if phase_index in phase_already_taken:
                    relation_dict['text'] = ""
                else:
                    text = getRelationTextFromDoc(document_name, phase_index)
                    relation_dict['text'] = text
                    phase_already_taken.append(phase_index)

            relation_texts.append(relation_dict)

            print(f"Relation dict: {relation_dict}")

        # Restituire i nodi e gli archi come liste
        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            'relation_texts': relation_texts
        }

    except Exception as e:
        print(f"Errore durante l'esecuzione della query: {e}")
        return None

'''
Metodo alla quale vengono passati una serie di tags e date, e restituisce la query Cypher da eseguire.
Implementato per permettere nel frontend la creazione di query attraverso un form.
'''
def createQueryByData(tags, dates):
    tag_conditions = []
    tag_list = ''

    # Process tags to build conditions
    for tag in tags:
        if tag['tag'] != '':
            tag_list += ":" + tag['tag']
            entities = [entity for entity in tag.get('entities', [])]
            for entity in entities:
                # Add conditions for both n and m with dynamic label based on the tag
                tag_conditions.append(
                    f"((n:{tag['tag']} AND ANY(key IN keys(n) WHERE n[key] CONTAINS '{entity}')) "
                    f"OR (m:{tag['tag']} AND ANY(key IN keys(m) WHERE m[key] CONTAINS '{entity}')))"
                )

    # Build the WHERE clause for dates
    date_conditions = []
    for date in dates:
        if date != '':
            date_conditions.append(
                f"((ANY(key IN keys(n) WHERE n[key] CONTAINS '{date}')) "
                f"OR (ANY(key IN keys(r) WHERE r[key] CONTAINS '{date}' AND NOT key = 'sentencesIndex')) AND type(r) <> \"IS_EXTENDED\")"
            )

    # Combine all conditions
    combined_conditions = " AND ".join(tag_conditions + date_conditions)

    # Build and return the query
    query = f"""
    MATCH (n)-[r]-(m)
    WHERE ({combined_conditions})
    RETURN n, r, m
    """
    return query



'''
Il metodo getQueryChoose prende in input il documento e restituisce un dizionario con le entità e le date riconosciute,
per permettere di visualizzare le selezioni nel frontend
'''

def getQueryChoose(doc, return_dict, return_dates):
    for ent in doc.ents:
        if ent.label_ not in ["TIME", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "PERCENT"]:
            if ent.label_ == 'DATE':
                date_text = ent.text.strip()
                if not date_text in return_dates and date_text.isdigit():
                    return_dates.append(date_text)
            else:
                if ent.label_ not in return_dict:
                    return_dict[ent.label_] = []
                if ent.text not in return_dict[ent.label_]:
                    return_dict[ent.label_].append(ent.text)



    return return_dict, return_dates
