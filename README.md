# Renewable Energy Knowledge Graph Builder 🌱⚡

This project enables the construction of a *knowledge graph* from PDF documents. The workflow includes entity and relationship extraction, advanced syntactic analysis, and the construction of graphs that can be visualized using *Neo4j*.
The project is designed to be adaptable to any topic. In our case, we have chosen the theme of *renewable energy*.

## Requirements
- [Python 3.12](https://www.python.org/downloads/)
- [spaCy](https://spacy.io/usage)
- [pdfplumber](https://pypi.org/project/pdfplumber/0.1.2/)
- [neo4j](https://neo4j.com/)

## Workflow
#### 1. PDF Conversion [textExtraction.pynb](textExtraction.ipynb)

- Use of *pdfplumber* to extract clean text from PDF files.
- Handling of complex layouts for accurate text extraction.
- Removal of non-textual elements such as images and headers.

#### 2. Named Entity Recognition (NER)
- Implementation of *spaCy* to identify key entities in the text.
- Development of custom categories such as energyRenewable to classify domain-specific entities.

#### 3. Relation Extraction [relationExtractor.py](NER/relationExtractor.py)
- Leveraging the *spaCy [en_core_web_trf](https://spacy.io/models/en#en_core_web_trf)* model for syntactic parsing and relationship extraction.
- Identification of *subjects, **verbs, and **objects* in sentences.
- Handling of complex sentences, including subordinates and modifiers, for context-rich relationship extraction.

#### 4. Knowledge Graph Construction [graphBuilder.py](NER/graphBuilder.py)
- Use of *Neo4j* for graph-based representation of extracted data.
- Conversion of relationships into triples in the form of (subject, verb, object).
- Standardization and normalization of entities to ensure consistency in the graph.


---

#### designed by Francesca Daniele e Marco Leto
