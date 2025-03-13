import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Télécharger les ressources NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Chargement des données
st.title("Analyse des Avis sur les Jeux Steam")
st.sidebar.header("Paramètres")

# Importer le dataset
st.sidebar.subheader("Chargement des données")
uploaded_file = st.sidebar.file_uploader("Chargez votre fichier CSV :", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Afficher un aperçu des données
    st.subheader("Aperçu des Données")
    st.write(df.head())

    # Taille du dataset
    st.subheader("Taille du Dataset")
    st.write(f"Nombre total d'entrées : {df.shape[0]} lignes et {df.shape[1]} colonnes.")

    # Nombre de jeux notés
    st.subheader("Nombre de Jeux Notés")
    st.write(len(df['app_id'].unique()))

    # Nombre de personnes dans le dataset
    st.subheader("Nombre de Joueurs")
    st.write(len(df['author_id'].unique()))

    # Répartition des avis
    st.subheader("Répartition des Avis")
    value_count = df['is_positive'].value_counts()
    st.bar_chart(value_count)

    # Nombre d'avis positifs et négatifs pour chaque jeu
    st.subheader("Avis Positifs et Négatifs par Jeu")
    Nb_pos_neg = df.groupby(["app_id", "is_positive"]).size().unstack(fill_value=0)
    st.write(Nb_pos_neg)

    # Ratio des avis pour chaque jeu
    st.subheader("Ratio des Avis par Jeu")
    Nb_pos_neg['Nb_total'] = Nb_pos_neg.sum(axis=1)
    Nb_pos_neg['Ratio positif'] = (Nb_pos_neg['Positive'] / Nb_pos_neg['Nb_total']) * 100
    Nb_pos_neg['Ratio negatif'] = (Nb_pos_neg['Negative'] / Nb_pos_neg['Nb_total']) * 100
    sorted_ratios = Nb_pos_neg.sort_values(by="Ratio positif", ascending=False)
    st.write(sorted_ratios)

    # Visualisation des ratios
    st.subheader("Visualisation des Ratios")
    st.bar_chart(sorted_ratios[['Ratio positif', 'Ratio negatif']])

    # Longueur des avis
    st.subheader("Longueur des Avis")
    df["Longueur avis"] = df["content"].str.len()
    st.write(f"Commentaire le plus long : {df['Longueur avis'].max()} caractères.")
    st.write(f"Commentaire le plus court : {df['Longueur avis'].min()} caractères.")
    st.write(f"Longueur moyenne : {df['Longueur avis'].mean():.2f} caractères.")

    # Recherche d'un mot ou d'une expression
    st.subheader("Recherche d'un Mot ou d'une Expression")
    query = st.text_input("Entrez un mot ou une expression à rechercher :")
    if query:
        def find_pos_exp(expression, documents):
            results = {}
            expression = re.sub(r'[^\w\s]', '', expression.lower())
            for doc_id, text in enumerate(documents):
                words = re.sub(r'[^\w\s]', '', str(text).lower()).split()
                positions = [i for i in range(len(words)) if ' '.join(words[i:i+len(expression.split())]) == expression]
                if positions:
                    results[doc_id] = positions
            return results
        
        def extract_context(expression, documents, window=5):
            """
            Extrait le contexte autour d'une expression dans un corpus.
            
            Parameters:
                expression (str): L'expression à rechercher.
                documents (list): Liste des textes où rechercher l'expression.
                window (int): Nombre de mots avant et après l'expression.
                
            Returns:
                dict: Dictionnaire contenant le document et les extraits.
            """
            expression_cleaned = re.sub(r'[^\w\s]', '', expression.lower())
            results = {}

            for doc_id, text in enumerate(documents):
                words = re.sub(r'[^\w\s]', '', str(text).lower()).split()
                for i in range(len(words) - len(expression_cleaned.split()) + 1):
                    if ' '.join(words[i:i+len(expression_cleaned.split())]) == expression_cleaned:
                        start = max(0, i - window)
                        end = min(len(words), i + len(expression_cleaned.split()) + window)
                        context = ' '.join(words[start:end])
                        if doc_id not in results:
                            results[doc_id] = []
                        results[doc_id].append(context)

            return results
        results = find_pos_exp(query, df['content'].dropna())
        if results:
            st.write(f"Nombre total d'occurrences : {sum(len(pos) for pos in results.values())}") 
            max_results = 5     
            st.write("") 
            context_results = extract_context(query, df['content'].dropna(), window=5)           
            if context_results:
                for i, (doc_id, contexts) in enumerate(context_results.items()):
                    if i >= max_results:
                        break
                    st.write(f"Document {doc_id}:")
                    for context in contexts:
                        st.write(f"... {context} ...")
            st.write("Affichage de 10 occurrences")
            for i, (doc_id, positions) in enumerate(results.items()):
                if i >= max_results:
                    break
                st.write(f"Document {doc_id}, Positions : {positions}")
        else:
            st.write("Aucune occurrence trouvée.")

    # Word Cloud
    st.subheader("Word Cloud autour d'une Expression")
    query_wordcloud = st.text_input("Entrez une expression pour le Word Cloud :", "great")
    if query_wordcloud:
        def generate_context_word_cloud(corpus, query):
            context_words = []
            query_cleaned = re.sub(r'[^\w\s]', '', query.lower())
            for doc in corpus:
                words = re.sub(r'[^\w\s]', '', str(doc).lower()).split()
                for i in range(len(words) - len(query_cleaned.split()) + 1):
                    if ' '.join(words[i:i+len(query_cleaned.split())]) == query_cleaned:
                        if i > 0:
                            context_words.append(words[i-1])
                        if i + len(query_cleaned.split()) < len(words):
                            context_words.append(words[i+len(query_cleaned.split())])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(context_words))
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        generate_context_word_cloud(df['content'].dropna(), query_wordcloud)

else:
    st.write("Veuillez charger un fichier CSV pour commencer.")

