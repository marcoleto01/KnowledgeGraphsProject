import spacy
from spacy.tokens import Span
from spacy import displacy
import json
import pandas as pd


'''
Metodi per carimento e salvataggio del file .spacy
'''
def saveNerModel(projectName, doc):
    #In the projectName folder, open a document "ner.spacy"
    with open(f"{projectName}/doc.spacy", 'wb') as f:
        f.write(doc.to_bytes())


def loadNerModel(projectName, nlp):
    #Open the document "ner.spacy" in the projectName folder
    doc = spacy.tokens.Doc(nlp.vocab)
    with open(f"{projectName}/doc.spacy", 'rb') as f:
        doc.from_bytes(f.read())

    return doc


'''
Il metodo get_common_words, prendeun file spacy e restituisce la lista delle parole comuni che sono contenute al suo interno
'''

def get_common_words(doc, stop_words):
    common_words = Counter()

    valid_tags = ['NN', 'NNS']  # Only singular and plural nouns

    for sent in doc.sents:
        for token in sent:

            if token.tag_ in valid_tags and token.text.lower() not in stop_words and token.ent_iob_ == 'O':
                common_words[token.text.lower()] += 1

    # Order the words by frequency
    common_words = dict(sorted(common_words.items(), key=lambda item: item[1], reverse=True))
    ret = []
    for word in common_words:
        el = {
            "word": word,
            "frequency": common_words[word]
        }
        ret.append(el)
    return ret


'''
Il metodo word_analysis, data una parola target ed un file .spacy, 
restituisce una lista di parole composte che contengono la parola target e rispettano le regole di composizione
'''
def word_analysis(doc, target_word):
    valid_tags = ['NN', 'NNS']  # Solo nomi singolari e plurali

    recognized_entities = set()
    for ent in doc.ents:
        if target_word in ent.text.lower():
            recognized_entities.add(ent.text.lower())

    target_word = target_word.lower()  # Normalizza la parola target in minuscolo

    combinations_counter = Counter()

    def has_compound_relation_with_span(token, span_tokens):
        """
        Verifica se un token ha una relazione compound con qualsiasi token nello span.
        """
        # Verifica se il token è compound del suo head e se il suo head è nello span
        if token.dep_ == "compound" and token.head in span_tokens:
            return True

        # Verifica se qualche token nello span è compound e ha come head il token corrente
        for span_token in span_tokens:
            if span_token.dep_ == "compound" and span_token.head == token:
                return True

        return False

    # Itera sui token del documento
    for i, token in enumerate(doc):
        # Controlla se il token corrente è la parola target e non appartiene a un'entità
        if token.text.lower() == target_word and token.tag_ in valid_tags:

            # Inizializza la parola composta con la parola target
            word_combination = [token.text.lower()]

            # Espansione a sinistra considerando il compound
            left_idx = i - 1
            while left_idx >= 0:
                current_token = doc[left_idx]
                if (current_token.tag_ in valid_tags and
                        has_compound_relation_with_span(current_token, doc[left_idx + 1:i + 1])):
                    word_combination.insert(0, current_token.text.lower())  # Aggiungi all'inizio
                    left_idx -= 1
                else:
                    break

            # Espansione a destra considerando il compound
            right_idx = i + 1
            while right_idx < len(doc):
                current_token = doc[right_idx]
                if (current_token.tag_ in valid_tags and
                        has_compound_relation_with_span(current_token, doc[i:right_idx])):
                    word_combination.append(current_token.text.lower())  # Aggiungi alla fine
                    right_idx += 1
                else:
                    break

            # Unisci le parole per formare la combinazione finale
            final_combination = ' '.join(word_combination)

            aggiungi = True
            for ent in recognized_entities:
                if final_combination in ent:
                    print(f"Entità riconosciuta: {final_combination} in {ent}")
                    aggiungi = False
                    break

            if (aggiungi):
                # Aggiungi la combinazione alla lista
                combinations_counter[final_combination] += 1

    # Creazione della lista finale di elementi ordinati per frequenza
    el_list = [{"word": word, "frequency": freq} for word, freq in combinations_counter.items()]
    el_list = sorted(el_list, key=lambda x: x['frequency'], reverse=True)

    print(el_list)

    return el_list



'''
Il metodo getEntities, prende un file .spacy e restituisce la lista delle entità riconosciute
'''
def getEntities(doc):
    entities = []
    for ent in doc.ents:
        entities.append(ent.label_)
    #take just the unique values
    entities = list(set(entities))
    return entities


'''
Il metodo getEntitiesNumber, prende un file .spacy e restituisce il numero di entità riconosciute
'''
def getEntitiesNumber(doc):
    ret = 0
    for ent in doc.ents:
        ret += 1
    return ret


'''
Il metodo getSentences HTML prende un file .spacy e restituisce una stringa HTML con le entità riconosciute
'''
def getSentenceHTML(sentence, doc):
    start = doc.text.find(sentence)
    end = start + len(sentence)
    sentence_span = doc.char_span(start, end)

    if sentence_span:
        html = displacy.render(sentence_span, style="ent", jupyter=False)
    else:
        html = displacy.render(doc, style="ent", jupyter=False)

    return html



'''
Il metodo add_custom_entity, prende un file .spacy, una frase e una label e aggiunge un'entità personalizzata 
'''
def add_custom_entity(doc, phrase, label):
    print(f"Aggiunta entità personalizzata: {phrase} con label {label}")
    # Converti la frase da cercare in minuscolo
    phrase_lower = phrase.lower()
    valid_tags = ['NN', 'NNS']  # Only singular and plural nouns

    # Trova tutte le occorrenze della frase nel doc
    spans = []
    for token in doc:
        # Confronta il token con la prima parola della frase, entrambi in minuscolo
        if token.text.lower() == phrase_lower.split()[0].lower():  # Inizia con la prima parola
            # Controlla se la frase completa corrisponde (case-insensitive)
            end_candidate = token.i + len(phrase.split())
            if end_candidate <= len(doc) and doc[token.i:end_candidate].text.lower() == phrase_lower:
                # Aggiungi la nuova entità con il testo originale
                spans.append(Span(doc, token.i, end_candidate, label=label))

    print(f"Aggiunte {len(spans)} nuove entità")
    print(spans)

    # Mantieni solo le nuove entità che non si sovrappongono con le entità esistenti
    non_overlapping_spans = []
    existing_spans = set((ent.start, ent.end) for ent in doc.ents)

    for span in spans:
        if not any((span.start < end and span.end > start) for start, end in existing_spans):
            non_overlapping_spans.append(span)
        else:
            print(f"Entità sovrapposta: {span.text}")

    # Aggiungi le nuove entità alla lista esistente di entità
    doc.ents = list(doc.ents) + non_overlapping_spans
    print(f"Agggiunti {len(non_overlapping_spans)} nuove entità")

    return doc


def remove_custom_entity(doc, label):
    # Mantieni solo le entità che non corrispondono alla label
    doc.ents = [ent for ent in doc.ents if ent.label_ != label]

    return doc


def remove_word_from_entity(doc, word):
    # Mantieni solo le entità che non contengono la parola
    doc.ents = [ent for ent in doc.ents if word not in ent.text]

    return doc

'''
Il metodo add_prior_entity, prende un file .spacy, una frase e una label e aggiunge un'entità personalizzata dandogli
priorità quindi sostituendola con quella gia esistente in caso di sovrapposizione
'''
def add_prior_entity(doc, phrase, label):
    print(f"Aggiunta entità personalizzata: {phrase} con label {label}")

    # Converti la frase da cercare in minuscolo
    phrase_lower = phrase.lower()

    # Trova tutte le occorrenze della frase nel doc
    spans = []
    for token in doc:
        # Confronta il token con la prima parola della frase, entrambi in minuscolo
        if token.text.lower() == phrase_lower.split()[0]:
            # Controlla se la frase completa corrisponde (case-insensitive)
            end_candidate = token.i + len(phrase.split())
            if end_candidate <= len(doc):
                span_candidate = doc[token.i:end_candidate]
                if span_candidate.text.lower() == phrase_lower:
                    # Aggiungi la nuova entità con il testo originale
                    spans.append(Span(doc, token.i, end_candidate, label=label))

    # Lista delle nuove entità che non si sovrappongono o prevalgono
    new_spans = []

    # Itera su tutte le nuove entità identificate
    for span in spans:
        sovrapposta = False
        start = span.start
        end = span.end

        # Verifica sovrapposizione con le entità esistenti
        ents_to_remove = []
        for ent in doc.ents:
            # Se l'entità esistente è contenuta nella nuova entità, la segniamo per rimozione
            if span.start <= ent.start and span.end >= ent.end:
                ents_to_remove.append(ent)
            # Se la nuova entità è completamente contenuta in un'entità esistente, scartala
            elif ent.start <= span.start and ent.end >= span.end:
                sovrapposta = True
                break
            # Estendi i limiti di `span` per includere entità parzialmente sovrapposte
            elif ent.start < span.start < ent.end:
                start = ent.start
                ents_to_remove.append(ent)
            elif ent.start < span.end < ent.end:
                end = ent.end
                ents_to_remove.append(ent)

        # Crea un nuovo span con i limiti estesi, se necessario
        if not sovrapposta:
            new_span = Span(doc, start, end, label=label)
            new_spans.append(new_span)

        # Rimuoviamo le entità che sono contenute o parzialmente sovrapposte
        doc.ents = [ent for ent in doc.ents if ent not in ents_to_remove]

    for span in new_spans:
        print(f"Aggiunta entità: {span.text} con label {span.label_} con start {span.start} e end {span.end}")

    print("Lunghezza entità esistenti: ", len(doc.ents))

    # Aggiungi le nuove entità non sovrapposte
    doc.ents = list(doc.ents) + new_spans

    print("Lunghezza entità esistenti con nuove: ", len(doc.ents))

    print(f"Agggiunte {len(new_spans)} nuove entità")
    return doc


from spacy.tokens import Span

def copy_entity_from_doc(doc, doc2):
    labelToIgnore = ['CARDINAL', 'NORP', 'FAC', 'LAW', 'PERCENT', 'ENERGY VALUE', 'PERSON', 'LOC', 'MONEY', 'PRODUCT',
                     'TIME', 'DATE', 'ORG', 'ORDINAL', 'GPE']
    sorted_entities = []
    ent_text = []
    for ent in doc2.ents:
        if ent.label_ not in labelToIgnore and ent.text not in ent_text:
            sorted_entities.append(ent)
            ent_text.append(ent.text)
    sorted_entities = sorted(sorted_entities, key=lambda ent: len(ent.text.split()), reverse=False)

    for ent in sorted_entities:
        add_prior_entity(doc, ent.text, ent.label_)

    return doc

'''
Il metodo add_all_custom_terms, prende un file .spacy e un progetto e aggiunge tutti i custom terms presenti nel file customTerms.txt
al file .spacy in modo da riconoscerli come entità personalizzate
'''
def add_all_custom_terms(projectName, doc):
    file_path = f"Projects/{projectName}/customTerms.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        custom_terms = json.loads(content.replace('custom_terms = ', '').strip())

    for term in custom_terms:
        doc = add_custom_entity(doc, term['pattern'], term['label'])
    return doc


'''
Il metodo get_custom_terms, prende un progetto e restituisce la lista dei custom terms presenti nel file customTerms.txt
associato al projectName
'''

def get_custom_terms(projectName):
    file_path = f"Projects/{projectName}/customTerms.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            custom_terms = json.loads(content.replace('custom_terms = ', '').strip())
    except FileNotFoundError:
        custom_terms = []
    return custom_terms


# Funzione di utilità per controllare se una parola è contenuta in un'entità riconosciuta
def is_contained_in_recognized_entities(start, end, recognized_entities):
    for ent_start, ent_end, _ in recognized_entities:
        # Se l'entità copre lo stesso intervallo o un intervallo maggiore, è contenuta
        if ent_start <= start and ent_end >= end:
            return True
    return False

'''
Il metodo get_entities_correlated_to_custom_terms, prende un file .spacy e un progetto e restituisce la lista delle entità correlate
ai custom terms presenti nel file customTerms.txt associato al projectName
'''


def get_entities_correlated_to_custom_terms(doc, projectName):
    custom_entities = Counter()
    custom_terms = get_custom_terms(projectName)

    recognized_entities = []
    for ent in doc.ents:
        recognized_entities.append(ent.text.lower())

    print("power sector emissions" in recognized_entities)

    def has_compound_relation_with_span(token, span_tokens):
        """
        Verifica se un token ha una relazione compound con qualsiasi token nello span.

        Args:
            token: Token da verificare
            span_tokens: Lista di token che formano l'entità corrente
        Returns:
            bool: True se esiste una relazione compound
        """
        # Verifica se il token è compound del suo head e se il suo head è nello span
        if token.dep_ == "compound" and token.head in span_tokens:
            return True

        # Verifica se qualche token nello span è compound e ha come head il token corrente
        for span_token in span_tokens:
            if span_token.dep_ == "compound" and span_token.head == token:
                return True

        return False

    for sent in doc.sents:
        for term in custom_terms:
            pattern = term['pattern'].lower()
            pattern_tokens = pattern.split()

            for i in range(len(sent) - len(pattern_tokens) + 1):
                # Verifica match pattern
                if not all(sent[i + j].text.lower() == pattern_tokens[j]
                           for j in range(len(pattern_tokens))):
                    continue

                # Tokens che formano il pattern matchato
                matched_span = sent[i:i + len(pattern_tokens)]
                entity = []
                start_idx = i
                end_idx = i + len(pattern_tokens)

                # Espansione a sinistra
                left_idx = i - 1
                while left_idx >= 0:
                    current_token = sent[left_idx]
                    if (current_token.tag_ in ['NN', 'NNS'] and
                            has_compound_relation_with_span(current_token, matched_span)):
                        entity.insert(0, current_token.text)
                        matched_span = sent[left_idx:end_idx]  # Aggiorna lo span
                        left_idx -= 1
                    else:
                        break

                # Aggiungi il pattern
                entity.extend([t.text for t in sent[i:i + len(pattern_tokens)]])

                # Espansione a destra
                right_idx = end_idx
                while right_idx < len(sent):
                    current_token = sent[right_idx]
                    if (current_token.tag_ in ['NN', 'NNS'] and
                            has_compound_relation_with_span(current_token, matched_span)):
                        entity.append(current_token.text)
                        matched_span = sent[start_idx:right_idx + 1]  # Aggiorna lo span
                        right_idx += 1
                    else:
                        break

                entity_text = ' '.join(entity).lower()

                aggiungi = True
                if (entity_text != pattern):
                    for ent in recognized_entities:
                        if entity_text in ent:
                            aggiungi = False

                    if aggiungi:
                        custom_entities[entity_text] += 1

    return [{"word": entity, "frequency": freq}
            for entity, freq in sorted(custom_entities.items(),
                                       key=lambda x: x[1],
                                       reverse=True)]


def get_words_from_entities(doc, entity):
    words = Counter()
    for ent in doc.ents:
        if ent.label_ == entity:
            words[ent.text.lower()] += 1

    ret = []
    for word in words:
        el = {
            "word": word,
            "frequency": words[word]
        }
        ret.append(el)

    ret = sorted(ret, key=lambda x: x["frequency"], reverse=True)
    return ret


def remove_word_from_entity_2(doc, word, entity_label=None):
    # Filtra le entità: rimuove quelle che contengono esattamente la parola specificata
    filtered_entities = []

    for ent in doc.ents:
        if word.lower() != ent.text.lower():
            # Mantiene l'entità se non corrisponde esattamente alla parola
            filtered_entities.append(ent)

    doc.ents = filtered_entities

    return doc


def change_entity_label(doc, old_label, new_label):
    # Cambia l'etichetta delle entità corrispondenti
    filtered_entities = []
    for ent in doc.ents:
        if ent.label_ == old_label:
            #create a new entity with the new label
            new_ent = Span(doc, ent.start, ent.end, label=new_label)
            filtered_entities.append(new_ent)
        else:
            filtered_entities.append(ent)

    doc.ents = filtered_entities

    return doc


def remove_entity(doc, entity):
    filtered_entities = []

    for ent in doc.ents:
        if ent.label_ != entity:
            filtered_entities.append(ent)

    doc.ents = filtered_entities

    return doc


def include_measure_unit_in_label(doc):
    # Lista delle unità di misura
    unit_measure = ["W", "Wh", "J", "kWh", "MWh", "GWh", "TWh", "MW", "GW", "TW"]
    new_ents = []
    seen_tokens = set()  # Per tracciare i token già inclusi in un'entità

    # Iteriamo su tutte le entità presenti nel documento
    for ent in doc.ents:
        # Verifichiamo se qualche token dell'entità è già stato utilizzato
        if any(token.i in seen_tokens for token in ent):
            continue  # Se sì, la saltiamo per evitare sovrapposizioni

        # Controlliamo se l'etichetta dell'entità è 'CARDINAL' o 'ORDINAL'
        if ent.label_ in ["CARDINAL", "ORDINAL"]:
            # Otteniamo il token successivo, se esiste
            next_token = doc[ent.end] if ent.end < len(doc) else None

            # Verifichiamo se il token successivo è un'unità di misura
            if next_token and next_token.text in unit_measure:
                # Creiamo una nuova entità combinando la quantità e l'unità di misura
                new_ent = Span(doc, ent.start, next_token.i + 1, label="ENERGY VALUE")
                new_ents.append(new_ent)  # Aggiungiamo la nuova entità alla lista
                # Segniamo i token come già usati
                seen_tokens.update(range(ent.start, next_token.i + 1))
            else:
                new_ents.append(ent)  # Se non c'è unità di misura, manteniamo l'entità originale
                seen_tokens.update(range(ent.start, ent.end))
        else:
            new_ents.append(ent)  # Manteniamo le altre entità intatte
            seen_tokens.update(range(ent.start, ent.end))

    # Aggiorniamo le entità del documento senza sovrapposizioni
    doc.ents = new_ents
    return doc


from collections import defaultdict, Counter


def get_entities_term(doc):
    base_labels = {
        "CARDINAL", "DATE", "EVENT", "GPE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON",
        "QUANTITY", "TIME", "FAC", "PRODUCT", "WORK_OF_ART"
    }

    entities = defaultdict(lambda: {"label": None, "frequency": 0})
    entityFrequency = Counter()

    for ent in doc.ents:
        #       if ent.label_ not in base_labels:
        entityFrequency[ent.label_] += 1
        if entities[ent.text]["label"] is None:
            entities[ent.text]["label"] = ent.label_
        entities[ent.text]["frequency"] += 1

    total_entities = sum(entityFrequency.values())
    label_percentage = {label: (count / total_entities) * 100 for label, count in entityFrequency.items()}

    entities_list = [{"pattern": text, "label": data["label"], "count": data["frequency"]}
                     for text, data in entities.items()]

    return entities_list, label_percentage


def get_html_by_term(doc, entity):
    for sent in doc.sents:
        for ent in sent.ents:
            if ent.text == entity:
                return displacy.render(sent, style="ent", jupyter=False)


def useTaggingModel(modelProject, text):
    trained_nlp = spacy.load(f"Models/{modelProject}/output/model-best")
    if "sentencizer" not in trained_nlp.pipe_names:
        trained_nlp.add_pipe("sentencizer")
    doc = trained_nlp(text)
    return doc


def get_possible_organization(doc):
    organizations = Counter()
    # Dummy words da ignorare all'inizio

    # Take all stop words
    stop_words = spacy.lang.en.stop_words.STOP_WORDS

    for sent in doc.sents:
        i = 0
        while i < len(sent):
            # Se è una dummy word all'inizio, salta
            if sent[i].text.lower() in stop_words:
                i += 1
                continue

            # Salta i token che sono già parte di entità riconosciute
            if sent[i].ent_type_:
                i += 1
                continue

            # Verifica se la parola inizia con maiuscola
            if sent[i].text[0].isupper():
                org_tokens = [sent[i]]
                next_idx = i + 1

                # Cerca parole consecutive che iniziano con maiuscola
                while (next_idx < len(sent) and
                       sent[next_idx].text[0].isupper() and
                       not sent[next_idx].ent_type_ and
                       not sent[next_idx].is_punct):
                    org_tokens.append(sent[next_idx])
                    next_idx += 1

                if len(org_tokens) >= 1:
                    # Costruisci il testo dell'organizzazione
                    org_text = ' '.join(token.text for token in org_tokens)

                    # Ignora se è già riconosciuta come entità di altro tipo
                    is_valid = True
                    for ent in doc.ents:
                        if org_text in ent.text and ent.label_ != 'ORG':
                            is_valid = False
                            break

                    if is_valid:
                        organizations[org_text] += 1
                        i = next_idx - 1  # Aggiorna l'indice saltando le parole usate
            i += 1

    # Formatta il risultato come lista di dizionari
    result = [
        {
            "organization": org,
            "frequency": freq
        }
        for org, freq in sorted(organizations.items(), key=lambda x: x[1], reverse=True)
    ]

    return result


def remove_leading_articles(doc):
    articles = {"the", "a", "an"}

    # Salviamo le entity da modificare per non modificare l'iteratore
    entities_to_modify = []

    # Identifichiamo le entity che iniziano con un articolo
    for ent in doc.ents:
        first_token = ent[0]
        if first_token.text.lower() in articles:
            # Salviamo le informazioni necessarie per ricreare l'entity
            entities_to_modify.append({
                'start': first_token.i + 1,  # Indice del token dopo l'articolo
                'end': ent.end,  # Indice finale originale
                'label': ent.label_  # Label originale
            })

    # Se non ci sono modifiche da fare, restituiamo il doc originale
    if not entities_to_modify:
        return doc

    # Creiamo una lista delle entity originali da mantenere
    entities_to_keep = []
    for ent in doc.ents:
        if ent[0].text.lower() not in articles:
            entities_to_keep.append(ent)

    # Creiamo le nuove entity
    new_entities = []
    for ent_info in entities_to_modify:
        span = doc[ent_info['start']:ent_info['end']]
        new_ent = spacy.tokens.Span(doc, span.start, span.end, label=ent_info['label'])
        new_entities.append(new_ent)

    # Combiniamo le entity originali da mantenere con quelle nuove
    doc.ents = entities_to_keep + new_entities

    return doc


def enhance_entities(doc):
    # Create a copy of the doc to avoid modifying the original
    doc_copy = doc.copy()

    # Get existing entities that are not QUANTITY or MONEY
    existing_ents = [ent for ent in doc_copy.ents
                     if ent.label_ not in ["QUANTITY", "MONEY"]]

    # Find all measurement units with /
    new_ents = []
    i = 0

    while i < len(doc_copy):
        # Check if current token is either QUANTITY or MONEY
        if doc_copy[i].ent_type_ in ["QUANTITY", "MONEY"]:
            start = i
            end = None
            current_type = doc_copy[i].ent_type_

            # Continue while we find tokens of the same type
            while i < len(doc_copy) and doc_copy[i].ent_type_ == current_type:
                i += 1

            if i < len(doc_copy) and doc_copy[i].text == "/":
                print(f"Found / at position {i}")
                end = i + 1
            elif i + 1 < len(doc_copy) and doc_copy[i + 1].text == "/":
                end = i + 2
            # Could happen that 15 m/ is seen as a unique token
            elif doc_copy[i - 1].text[len(doc_copy[i - 1].text) - 1] == "/":
                print(f"Found / at position {i}")
                end = i
            elif i < len(doc_copy) and doc_copy[i].text == "C":
                end = i

            print("Last token: ", doc_copy[i - 1])

            if end is not None and end < len(doc_copy):
                # Create new span with the same label as the original entity
                try:
                    new_span = Span(doc_copy, start, end + 1, label=current_type)
                    if not any(has_overlap(new_span, ent) for ent in existing_ents):
                        print(f"New span aggiunta: {new_span}")
                        new_ents.append(new_span)
                except ValueError:
                    pass  # Handle invalid span creation
                i += 1
            else:
                old_span = Span(doc_copy, start, i, label=current_type)
                if not any(has_overlap(old_span, ent) for ent in existing_ents):
                    print(f"Old span aggiunta: {old_span}")
                    new_ents.append(old_span)
        else:
            i += 1

    # Combine existing and new entities
    final_ents = existing_ents + new_ents

    # Sort entities by start position
    final_ents = sorted(final_ents, key=lambda x: x.start)

    # Set the entities in the doc copy
    doc_copy.ents = final_ents

    return doc_copy


def include_base_entities(doc, text, nlp):
    # Ottieni le entità standard dal testo di input
    doc_standard_entities = nlp(text)
    doc_standard_entities = remove_leading_articles(doc_standard_entities)
    doc_standard_entities = enhance_entities(doc_standard_entities)

    # Copia le entità esistenti
    new_ents = list(doc_standard_entities.ents)

    for ent in doc.ents:
        # Aggiungi entità di 'doc' se non c'è sovrapposizione o sovrascrivi eventuali sovrapposizioni
        overlapping_ents = [ent2 for ent2 in new_ents if has_overlap(ent, ent2)]
        for overlap in overlapping_ents:
            new_ents.remove(overlap)
        new_span = Span(doc_standard_entities, ent.start, ent.end, label=ent.label_)
        new_ents.append(new_span)

    # Ordina le entità per posizione per evitare errori di SpaCy
    new_ents = sorted(new_ents, key=lambda span: span.start)

    # Assegna le entità ordinate al documento standard
    doc_standard_entities.ents = new_ents
    return doc_standard_entities

def has_overlap(span1, span2):
    """Check if two spans overlap"""
    return (span1.start <= span2.end and span2.start <= span1.end)
