from openai import OpenAI

# Configura il client con la tua API key usando una variabile di ambiente



def elaborate_prediction(relazioni):

    prompt = f"""
            Analizza le relazioni tra le entità fornite, fornendo una panoramica qualitativa delle principali connessioni e tendenze. 
            Evita di includere valori numerici e concentrati sull’identificazione di pattern globali, cluster ricorrenti e il significato contestuale delle relazioni.
            Evidenzia gli elementi più rilevanti e le dinamiche emergenti senza riferimenti a dati specifici, ma con una sintesi chiara e informativa.

            **IMPORTANTE:** Formatta la risposta in **Markdown**, utilizzando:
            - **Grassetto** per evidenziare gli elementi chiave (es: **Entità Importante**)
            - Liste puntate se necessarie
            - Non aggiungere Introduzioni nè conclusioni vai direttamente al punto
            - Non mettere troppi spazi nel markdown in quanto viene visualizzato in uno spazio ristretto dell'html

            Ecco le relazioni da analizzare:

            '{relazioni}'
            """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in Entity Relationship Analysis."},
            {"role": "user", "content": prompt}
        ]
    )

    risposta = completion.choices[0].message.content.strip()

    return risposta


def elaborate_community_report_phase(phase):

    prompt = f"""
            Riformula il seguente testo senza alterarne il significato e mantenendo la lingua inglese, per evitare che si ripetino stessi soggetti. Dammi solo la risposta.
            {phase}            
    """


    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in Entity Relationship Analysis."},
            {"role": "user", "content": prompt}
        ]
    )

    risposta = completion.choices[0].message.content.strip()

    return risposta


def elaborate_community_report_text(texts):

    full_text =""

    for text in texts:
        full_text+=elaborate_community_report_phase(text)+"\n"

    return full_text





