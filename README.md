# Topic Modelling
Topic Modelling of abstracts and titles using PubMed

I worked on this project as a part of my Master's Paper Thesis. As a part of bigger research I worked on creating a dataset of abstracts and titles from PubMed for certain medical queries.
- Python Packages Used:
- Biopython
- Pandas

For data mining and topic modelling:
- Gensim
- NLTK
- SpaCy
- PyLDAvis

The **data_from_pubmed.py** file uses the functions from Biopython for fetching scientific literature from the PubMed database based on search results.
The **topic_modelling.py** file cleans the data (paper abstracts and titles) and performs topic_modelling using gensim and visualize the topic models using Pyldavis.
