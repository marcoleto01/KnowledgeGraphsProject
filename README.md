# Renewable Energy Knowledge Graph: From Extraction to Advanced Analysis 🌱⚡📊

## STEP 1: Renewable Energy Knowledge Graph Builder 🌱⚡

This project enables the construction of a *knowledge graph* from PDF documents. The workflow includes entity and relationship extraction, advanced syntactic analysis, and the construction of graphs that can be visualized using *Neo4j*.
The project is designed to be adaptable to any topic. In our case, we have chosen the theme of *renewable energy*.

### Requirements
- [Python 3.12](https://www.python.org/downloads/)
- [spaCy](https://spacy.io/usage)
- [pdfplumber](https://pypi.org/project/pdfplumber/0.1.2/)
- [neo4j](https://neo4j.com/)

### Workflow
##### 1. PDF Conversion [textExtraction.pynb](textExtraction.ipynb)

- Use of *pdfplumber* to extract clean text from PDF files.
- Handling of complex layouts for accurate text extraction.
- Removal of non-textual elements such as images and headers.

##### 2. Named Entity Recognition (NER)
- Implementation of *spaCy* to identify key entities in the text.
- Development of custom categories such as energyRenewable to classify domain-specific entities.

##### 3. Relation Extraction [relationExtractor.py](NER/relationExtractor.py)
- Leveraging the *spaCy [en_core_web_trf](https://spacy.io/models/en#en_core_web_trf)* model for syntactic parsing and relationship extraction.
- Identification of *subjects, **verbs, and **objects* in sentences.
- Handling of complex sentences, including subordinates and modifiers, for context-rich relationship extraction.

##### 4. Knowledge Graph Construction [graphBuilder.py](NER/graphBuilder.py)
- Use of *Neo4j* for graph-based representation of extracted data.
- Conversion of relationships into triples in the form of (subject, verb, object).
- Standardization and normalization of entities to ensure consistency in the graph.


## STEP 2: Advanced Graph Analysis and Community Detection 📊🔍

In this second phase, we focus on further analysis, refinement, and enrichment of the knowledge graph built in Phase 1. The notebooks provided detail critical tasks to ensure quality, usability, and enhanced insights from the graph data.

### Workflow
##### 1. Graph Cleaning and Data Creation [GraphCleaningAndDataCreation.ipynb]

- Data cleaning and preprocessing steps to refine the knowledge graph.
- Resolution of inconsistencies and removal of duplicate entities.
- Standardization and normalization of entity names and relations.
- Creation of structured datasets to facilitate advanced analytics.

##### 2. Community Extraction and Model Training [CommunityExtractionAndModelTraining.ipynb]
- Application of graph community detection algorithms (e.g., Louvain method) to identify clusters or communities within the graph.
- Analysis of community structures to uncover insights and patterns related to renewable energy topics.
- Training machine learning models leveraging graph-based features to predict relationships or classify entities within the renewable energy domain.

##### 3. Graph Analysis [graphAnalysis.py]
- Advanced statistical analysis of the knowledge graph.
- Calculation of centrality measures (e.g., degree, betweenness, closeness) to identify influential nodes and critical connections.
- Visualization of analytical results for intuitive interpretation and effective communication of findings.



---

##### This project has been developed as a coursework assigned for the class "Information Retrieval and Natural Language Processing", Fall 2024 (Instructor: prof. [Andrea Tagarelli](https://mlnteam-unical.github.io/)) at the DIMES Department, University of Calabria, Italy 

#### Designed by Francesca Daniele e Marco Leto 
