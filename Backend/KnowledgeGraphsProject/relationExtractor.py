import json
import os

import spacy
import string


# CONVERSIONE VERBO PASSIVO IN ATTIVO
def passive_to_active_verb(doc):
    aux_be = None
    aux_have = None
    been = None
    main_verb = None

    # identificare verbi e ausiliari
    for token in doc:
        if token.lemma_ == "be" and token.pos_ == "AUX":
            aux_be = token
        elif token.lemma_ == "have" and token.pos_ == "AUX":
            aux_have = token
        elif token.lower_ == "been":
            been = token
        elif token.pos_ == "VERB" and token.tag_ == "VBN":
            main_verb = token

    if not main_verb:
        return doc.text

    # forma base del verbo principale
    base_form = main_verb.lemma_

    # Regole per la conversione passivo-attivo
    if aux_have:
        if aux_have.text.lower() in ["has", "had"]:
            return base_form + "s"
        else:
            return base_form
    elif aux_be:
        if aux_be.text.lower() in ["is", "are"]:
            return base_form + "s"
        elif aux_be.text.lower() == "am":
            return base_form
        else:
            return base_form
    elif been:
        return base_form + "ing"

    return base_form

#Verifica se un token non è un attributo rispetto al suo nodo principale
def notFromAttr(token):
    for t in token.head.children:
        if t.dep_ == 'attr' and t == token:
            return False
    return True

#Ricorsivamente identifica il soggetto completo a partire da un token
def get_full_subject_recursive(token, start, end):
    if not token.rights and not token.lefts:
        return token, start, end

    token_children_right = []
    token_children_left = []
    for t in token.children:
        if t in token.rights:
            token_children_right.append(t)
        else:
            token_children_left.append(t)

    for t in token_children_left:
        if t.dep_ in ['det', 'amod', 'compound', 'nummod', 'quantmod', 'nmod', "npadvmod"] or t.pos_ in ['DET', 'ADJ',
                                                                                                         'NUM']:
            start = t if t.i < start.i else start
            _, start, end = get_full_subject_recursive(t, start, end)
            # Ricorsione sui figli sinistra

    for t in token_children_right:
        if t.dep_ in ['prep', 'compound', 'amod']:
            if t.dep_ == 'prep' and notFromAttr(token):
                for pobj in t.children:
                    if pobj.dep_ == 'pobj':
                        end = max(end, pobj, key=lambda x: x.i)
                        _, start, end = get_full_subject_recursive(pobj, start, end)
                        # Ricorsione sui figli destri

    return token, start, end

#Ottiene il soggetto completo associato a un token, attiavando la ricorsione
def get_full_subject(token):
    _, start, end = get_full_subject_recursive(token, token, token)
    return token.doc[start.i:end.i + 1]


# Used to save information about the subject --> [high demand in China] and [India] become [high demand in China] and [high demand in India]

#Recupera tutti i soggetti collegati a un token verbo collegati da congiunzione, dunque li separe per formare diversi soggetti
def get_all_subjects(token, old_subject):
    subjects = []

    main_subject = get_full_subject(token)
    if old_subject is None:
        subjects.append(main_subject)

    for child in main_subject[-1].children:
        if child.dep_ == "conj":
            if old_subject is None:
                main_full_subject = get_full_subject(main_subject[-1])
                complete_child_subject = [token for token in main_subject if token not in main_full_subject]
            else:
                old_full_subject = get_full_subject(old_subject[-1])
                complete_child_subject = [token for token in old_subject if token not in old_full_subject]
            child_subject = get_full_subject(child)
            for el in child_subject:
                complete_child_subject.append(el)
            # order the list of token
            complete_child_subject.sort(key=lambda x: x.i)
            subjects.append(complete_child_subject)
            child_other_subjects = get_all_subjects(child, main_subject)
            for sub in child_other_subjects:
                if sub not in subjects:
                    subjects.append(sub)

    return subjects

#Raccoglie tutti i verbi associati a un verbo principale, inclusi ausiliari e modificatori
def get_all_verbs(verb):
    verb_tokens = []
    # if verb is istance of a list
    if isinstance(verb, list):
        for v in verb:
            verb_tokens.append(v)
    else:
        verb_tokens.append(verb)

    for token in verb.children:
        if token.dep_ == 'xcomp' or token.dep_ == 'acomp' or token.dep_ == 'advmod' or token.dep_ == 'aux' or token.dep_ == 'auxpass':
            verb_tokens.append(token)
            verb_list = get_all_verbs(token)
            for v in verb_list:
                if v not in verb_tokens:
                    verb_tokens.append(v)

    # order verb tokens for token.i
    verb_tokens.sort(key=lambda x: x.i)

    return verb_tokens

#Ricorsivamente identifica l'oggetto completo a partire da un token
def get_full_object_recursion(token, start, end):
    if not token.rights and not token.lefts:
        return token, start, end

    # Processa i modificatori a sinistra
    for t in token.lefts:
        if t.dep_ in ['prep', 'pobj', 'compound', 'nummod', 'quantmod', 'nmod', 'amod', 'advmod', ] or t.pos_ in ['DET',
                                                                             'ADJ',                                                                                                                  'NUM']:
            # Aggiorna `start` solo se necessario
            start = t if t.i < start.i else start
            _, start, end = get_full_object_recursion(t, start, end)  # Ricorsione sui figli sinistri

    # Processa i modificatori a destra
    for t in token.rights:
        if t.dep_ in ['prep', 'pobj', 'compound', 'nummod', 'quantmod', 'nmod', 'amod', 'advmod', 'advcl', 'npadvmod',
                      'pcomp'] or t.pos_ in ['DET', 'ADJ', 'NUM']:
            # Aggiorna `end` solo se necessario
            end = t if t.i > end.i else end
            _, start, end = get_full_object_recursion(t, start, end)  # Ricorsione sui figli destri
        elif t.dep_ in ['acl']:
            end = t if t.i > end.i else end
            _, start, end = get_full_object_recursion_by_verbs(t, start, end)
        elif t.dep == "relcl" and (
                t.pos_ != "VERB" and any(child.dep_ in ["nsubj", "nsubjpass"] for child in t.children)):
            end = t if t.i > end.i else end
            _, start, end = get_full_object_recursion_by_verbs(t, start, end)

    return token, start, end


def get_full_object_recursion_by_verbs(token, start, end):
    if not token.rights and not token.lefts:
        return token, start, end

    for t in token.rights:
        if t.dep_ in ["prep", "pobj", "dobj", "advmod", 'npadvmod']:
            end = t if t.i > end.i else end
            _, start, end = get_full_object_recursion(t, start, end)

    return token, start, end

#Ottiene l'oggetto completo associato a un token
def get_full_object(token):
    _, start, end = get_full_object_recursion(token, token, token)
    return token.doc[start.i:end.i + 1]

#Recupera tutti gli oggetti collegati a un oggetto principale, incluse congiunzioni
def get_all_objects(obj):
    objects = []

    main_object = get_full_object(obj)
    objects.append(main_object)

    for child in obj.children:
        if child.dep_ == "conj":
            new_objects = get_all_objects(child)
            for obj in new_objects:
                if obj not in objects:
                    objects.append(obj)

    return objects


def is_contained(span1, span2):
    print(f"Span1: {span1}; Span2: {span2}")
    return span1.start >= span2.start and span1.end <= span2.end

#elimina oggetti ridondanti o contenuti in altri oggetti, separa preposizioni (ADP) e congiunzioni subordinanti (SCONJ), e combina oggetti con le loro preposizioni
def process_prep(objects, verb):
    new_objects = []

    new_spans = []
    # Delete all the objects that are contained in other objects
    for span in objects:
        if not any(is_contained(span, other_span) and span != other_span for other_span in new_spans):
            new_spans.append(span)
            el = [token for token in span if token not in verb]
            if len(el) > 0:
                new_objects.append(el)

    objs = []
    adp = []

    for obj in new_objects:
        if obj[0].pos_ == 'ADP' or obj[0].pos_ == 'SCONJ':
            adp.append(obj)
        else:
            objs.append(obj)

    final_objects = []
    if len(objs) != 0:
        for o in objs:
            el = [token for token in o]
            for a in adp:
                for token in a:
                    el.append(token)
            # sort the list of token
            el.sort(key=lambda x: x.i)
            final_objects.append(el)
    else:
        el = []
        for a in adp:
            for token in a:
                el.append(token)
        el.sort(key=lambda x: x.i)
        final_objects.append(el)

    return final_objects, objs, adp

# Implementa la risoluzione delle anafore
def process_sub(sub, token):
    print(f"Entro in process_sub con sub: {sub} e token: {token}")

    pron = ["this", "that", "it", "those", "they", "which"]

    new_sub = None
    # If not any pronoun in sub
    if not any(word.text.lower() in pron for word in sub):
        return sub

    subj = " ".join(v.text for v in sub) if isinstance(sub, list) else sub.text
    # Caso in cui il token è collegato ad un VERB che ha la relazione di nsuj o nsubjpass con un il soggetto di riferimento
    print(f"Subj: {subj}; Type: {type(subj)}")

    if (any(s in subj for s in pron)) and token.dep_ in ["relcl", "advcl", "conj", "ccomp",
                                                         "xcomp"] and token != token.head:

        new_sub = token.head
        if not new_sub.pos_ == "VERB":
            if new_sub.dep_ == "npadvmod":
                verb = new_sub.head
                for verb_child in verb.children:
                    if verb_child.dep_ == "nsubj" or verb_child.dep_ == "nsubjpass":
                        new_sub = verb_child
            elif new_sub.dep_ == "appos":
                new_sub = new_sub.head
            else:
                for token_child in new_sub.children:
                    if token_child.dep_ == "attr":
                        new_sub = token_child
        else:
            for token_child in new_sub.children:
                if token_child.dep_ == "nsubj" or token_child.dep_ == "nsubjpass":
                    new_sub = token_child

    # Caso in cui il token non ha padre e il riferimento al verbo del soggetto che si cerca è un figlio
    if any(s in subj for s in pron):
        for child in token.children:
            if child.pos_ == 'VERB' and child.i < token.i and child.dep_ in ['ccomp', 'conj', 'xcomp', 'advcl']:
                for verb_child in child.children:
                    if verb_child.dep_ in ["nsubj", 'nsubjpass']:
                        new_sub = verb_child

    if new_sub:
        ret_subject = get_full_subject(new_sub)
        ret = []
        for el in sub:
            if el.text in pron:
                for token in ret_subject:
                    ret.append(token)
            else:
                ret.append(el)
        return ret

    return sub

# Estrae il soggetto, verbo e oggetti principali e preposizioni da un token
def get_component_from_token(token):
    # Find all the verb

    verb = get_all_verbs(token)

    subjects = []
    objects = []
    new_verb = ""
    forma_passiva=False

    if any(child.dep_ == "auxpass" for child in token.children):  # FORMA PASSIVA

        prep_sub_token = []
        forma_passiva=True
        for v in verb:
            for v_child in v.children:
                if v_child.dep_ == "agent":
                    agent = v_child
                    for agent_child in agent.children:
                        if agent_child.dep_ == "pobj":
                            new_subjects = get_all_subjects(agent_child, None)
                            for sub in new_subjects:
                                if sub not in subjects:
                                    sub_p = process_sub(sub, v_child)
                                    subjects.append(sub_p)

                elif v_child.dep_ == "oprd" or v_child.dep_ == "dative":
                    new_subjects = get_all_subjects(v_child, None)
                    for sub in new_subjects:
                        if sub not in objects:
                            subjects.append(sub)
                elif v_child.dep_ == "prep" and v_child.pos_ == "ADP":
                    prep_sub_token.append(v_child)

        prep_obj_token = []

        if len(subjects) == 0:
            for prep in prep_sub_token:
                prep_obj_token.append(prep)
                for token_child in prep.children:
                    new_subjects = get_all_subjects(token_child, None)
                    for sub in new_subjects:
                        if sub not in objects:
                            subjects.append(sub)

        for token_child in token.children:
            if token_child.dep_ == "nsubjpass":
                new_objects = get_all_objects(token_child)
                for obj in new_objects:
                    if obj not in objects:
                        obj_p = process_sub(obj, token)
                        objects.append(obj_p)
            # New code
            if token_child.dep_ in ["prep", "pobj", "dobj", "advmod", 'npadvmod'] and token_child not in prep_obj_token:
                new_objects = get_all_objects(token_child)
                for obj in new_objects:
                    if obj not in objects:
                        objects.append(obj)

        new_verb = passive_to_active_verb(verb)

    else:  # FORMA ATTIVA
        for child in token.children:
            if child.dep_ == "nsubj" or (
                    child.dep_ == "attr" and not any(child.dep_ == "nsubj" for child in child.head.children)):
                # Un soggetto è stato trovato
                new_subjects = get_all_subjects(child, None)
                for sub in new_subjects:
                    if sub not in subjects:
                        sub_p = process_sub(sub, token)
                        subjects.append(sub_p)
                        # THe subject is not found
        if len(subjects) == 0 and token.dep_ == "advcl":
            verb_parent = token.head
            for verb_child in verb_parent.children:
                if verb_child.dep_ == "nsubj":
                    new_subjects = get_all_subjects(verb_child, None)
                    for sub in new_subjects:
                        if sub not in subjects:
                            sub_p = process_sub(sub, verb_parent)
                            subjects.append(sub_p)

        for v in verb:

            for token in v.children:
                if token.dep_ in ["prep", "pobj", "dobj", "advmod", 'npadvmod']:
                    new_objects = get_all_objects(token)
                    for obj in new_objects:
                        if obj not in objects:
                            objects.append(obj)
                if token.dep_ == "attr":
                    new_focus = token
                    if any(child.dep_ == "nsubj" for child in new_focus.head.children):
                        new_objects = get_all_objects(new_focus)
                        for obj in new_objects:
                            if obj not in objects:
                                objects.append(obj)
                    else:
                        for focus_child in new_focus.children:
                            if focus_child.dep_ in ["prep", "pobj", "dobj", "npadvmod", "compound"]:
                                new_objects = get_all_objects(focus_child)
                                for obj in new_objects:
                                    if obj not in objects:
                                        objects.append(obj)

    objs = []
    adps = []

    if not forma_passiva:
        if len(objects) != 0:
            objects, objs, adps = process_prep(objects, verb)
            '''
            print("Objects: ", objects)
            print("Objs: ", objs)
            print("Adps: ", adps)     
            '''

    # Prendi punteggiatura in un vettore
    punctuation = string.punctuation
    to_remove = []
    for adp in adps:
        if len(adp) == 1 and adp[0].text in punctuation:
            to_remove.append(adp)
    for el in to_remove:
        adps.remove(el)

    if (new_verb):
        verb = new_verb

    return subjects, verb, objs, adps

# Verifica se un ausiliare non è collegato a un verbo
def auxiliar_without_verb(token):
    if token.pos_ == "AUX":
        if token.head != token:
            head_child = token.head.children
            for child in head_child:
                if child.dep_ == "aux" or child.dep_ == "auxpass":
                    return False
        return True
    return False

# Crea relazioni tra soggetto, verbo e oggetti e preposizioni
def get_all_relations(subject, verb, objs, adps):
    relations = []

    if len(objs) != 0 and len(adps) != 0:
        for obj in objs:
            for sub in subject:
                # Add al the adps
                relations.append((sub, verb, obj, adps))
    elif len(adps) != 0:
        for sub in subject:
            relations.append((sub, verb, [], adps))
    elif len(adps) == 0 and len(objs) != 0:
        for obj in objs:
            for sub in subject:
                relations.append((sub, verb, obj, []))

    return relations

# Restituisce l'identificativo di un'entità a cui appartiene un token
def return_entity_id(token, entity_dict):
    for ent in entity_dict:
        # If the token pos is betweeen start or stop of the entity return the entity id
        if token.i >= entity_dict[ent]["start"] and token.i <= entity_dict[ent]["end"]:
            return ent

# Determina se due insiemi di token appartengono alla stessa entità
def same_entity(entity1, entity2, entity_dict):
    ent_id_1 = {}
    ent_id_2 = {}
    for token in entity1:
        if token.ent_iob_ != "O":
            entity_id = return_entity_id(token, entity_dict)
            if entity_id in ent_id_1:
                ent_id_1[entity_id].append(token)
            else:
                ent_id_1[entity_id] = [token]
    for token in entity2:
        if token.ent_iob_ != "O":
            entity_id = return_entity_id(token, entity_dict)
            if entity_id in ent_id_2:
                ent_id_2[entity_id].append(token)
            else:
                ent_id_2[entity_id] = [token]
    # If there are two same id in the dictonary and the tokens are the different, return True

    for id1, tokens1 in ent_id_1.items():
        if id1 in ent_id_2:
            tokens2 = ent_id_2[id1]
            if set(tokens1) != set(tokens2):
                return True
    return False

# Concatena i testi di un array di token in una stringa
def get_array_string(array):
    return " ".join(p.text if hasattr(p, 'text') else str(p) for p in array)


from itertools import combinations

# Unifica relazioni con lo stesso soggetto e predicato combinando gli oggetti
def unify_same_relation_objects(sentence_relations, entity_dict):
    if len(sentence_relations) == 1:
        return sentence_relations

    relations = []
    different_objects = {}

    # Raggruppa relazioni con la stessa coppia soggetto-predicato
    for rel in sentence_relations:
        string_id = get_array_string(rel[0]) + get_array_string(rel[1])
        if len(rel[2]) == 0:  # Relazioni senza oggetti
            relations.append(rel)
        elif string_id not in different_objects:
            different_objects[string_id] = [rel]
        else:
            different_objects[string_id].append(rel)

    # Unifica gli oggetti per le stesse coppie soggetto-predicato
    object_to_unify = {}

    for objects_key, rel_list in different_objects.items():
        for rel1, rel2 in combinations(rel_list, 2):
            elemento1, elemento2 = rel1[2], rel2[2]
            if same_entity(elemento1, elemento2, entity_dict):
                if objects_key not in object_to_unify:
                    object_to_unify[objects_key] = [rel1, rel2]
                else:
                    if rel1 not in object_to_unify[objects_key]:
                        object_to_unify[objects_key].append(rel1)
                    if rel2 not in object_to_unify[objects_key]:
                        object_to_unify[objects_key].append(rel2)

        # Aggiungi le relazioni non unificate
        for rel in rel_list:
            if objects_key in object_to_unify and rel in object_to_unify[objects_key]:
                continue
            else:
                relations.append(rel)

    # Crea nuove relazioni con oggetti unificati
    for objects_key, rel_list in object_to_unify.items():
        new_obj = []
        subject, predicate, _, additional_info = rel_list[0]  # Mantieni soggetto, predicato e info extra

        for rel in rel_list:
            for token in rel[2]:  # Oggetti della relazione
                if token not in new_obj:
                    new_obj.append(token)

        # Ordina i nuovi oggetti
        new_obj.sort(key=lambda x: x.i)

        # Aggiungi la nuova relazione unificata se non esiste già
        if (subject, predicate, new_obj, additional_info) not in relations:
            relations.append((subject, predicate, new_obj, additional_info))

    return relations


import spacy

# Rimuove oggetti numerici o relativi a date, quantità, ecc., dalle relazioni
def remove_numerical_objects(relations):
    # Take spacy stop words
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

    for relation in relations:

        ent_id = 0
        uniqueEnt = True
        if len(relation[2]) == 1:
            removeObj = False
            if relation[2][0].ent_type_ in ["DATE", "TIME", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "PERCENT"]:
                removeObj = True
            if removeObj:
                objToremove = relation[2][0]
                relation[2].remove(objToremove)
                relation[3].append([objToremove])
        else:
            for word in relation[2]:
                if word.ent_type_ in ["DATE", "TIME", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "PERCENT"]:
                    if ent_id == 0:
                        ent_id = word.ent_id_
                    else:
                        if word.ent_id_ != ent_id:
                            uniqueEnt = False
                elif word.text.lower not in spacy_stopwords:
                    uniqueEnt = False

            if uniqueEnt and ent_id != 0:
                new_adp = []
                for word in relation[2]:
                    new_adp.append(word)
                for word in new_adp:
                    relation[2].remove(word)

                relation[3].append(new_adp)

    return relations

# Estrae relazioni da un documento filtrando per oggetti ritenuti rilevanti
def extract_filtered_relations(doc):
    relations = []
    last_relation = None
    # entity_dict_contains start,end text and label
    entity_dict = {}
    ent_id = 0
    for ent in doc.ents:
        entity_dict[ent_id] = {"start": ent.start, "end": ent.end, "text": ent.text, "label": ent.label_}
        ent_id += 1

    for sent in doc.sents:

        for token in sent:
            if token.pos_ == 'VERB' or auxiliar_without_verb(token):
                # Gestione attiva della frase
                subject, verb, objs, adps = get_component_from_token(token)
                if subject and verb and (objs or adps):
                    subject_relations = get_all_relations(subject, verb, objs, adps)
                    relations.extend(unify_same_relation_objects(subject_relations, entity_dict))

    relations = remove_numerical_objects(relations)

    return relations

# Rimuove relazioni che hanno cone soggetto pronomi relativi come "this", "that", ecc
def remove_relative_pronouns(relations):
    relation_to_remove = []
    for relation in relations:

        if isinstance(relation[2], list):
            obj = " ".join(token.text for token in relation[2])
        else:
            obj = relation[2]

        if any(word == relation[0] for word in ["this", "that", "it", "those", "they", "which"]) or any(
                word == obj for word in ["this", "that", "it", "those", "they", "which"]):
            relation_to_remove.append(relation)

    return [relation for relation in relations if relation not in relation_to_remove]

#  Estrae e processa relazioni da un documento
def extract_relations(doc, document):
    relations = []
    directory_path = os.path.join('Tagging', document)
    file_path = os.path.join(directory_path, 'phasesMapping.json')

    mapFile= False
    # Controlla se la directory esiste, se no la crea
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Creata directory: {directory_path}")

    # Controlla se il file esiste
    if not os.path.exists(file_path):
        mapFile = True
        docMapping={}

    for index, sent in enumerate(doc.sents):
        print("Sentence: ", sent.text)
        if mapFile:
            docMapping[index] = sent.text
        # Extract relations
        phase_relations = []
        sent_relation = extract_filtered_relations(sent)
        print("Sent relations: ", sent_relation)
        for el in sent_relation:
            #add the sentence index to el tuple
            new_el = el + (index,)

            phase_relations.append(new_el)
            print("Relation: ", new_el)

        relations.append(phase_relations)

    relations = [remove_relative_pronouns(rel) for rel in relations]

    if mapFile:
        with open(file_path, 'w') as f:
            json.dump(docMapping, f)

    return relations
