from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import os

app = Flask(__name__, static_folder='static')
load_dotenv()


books = pd.read_csv('data/books_with_emotions.csv')
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


raw_documents = TextLoader("data/tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    
    results = db_books.similarity_search_with_score(query, k=initial_top_k)
    
    
    search_results = []
    for doc, score in results:
        isbn = doc.page_content.split(":")[0]
        search_results.append({
            'isbn13': str(isbn),  
            'score': score
        })
    
    search_df = pd.DataFrame(search_results)
    
   
    search_df['isbn13'] = search_df['isbn13'].astype(str)
    books['isbn13'] = books['isbn13'].astype(str)
    
    
    book_recs = pd.merge(search_df, books, on='isbn13', how='left')
    
    
    if category and category != "All":
        book_recs = book_recs[book_recs['simple_categories'] == category]
    
    
    if tone and tone != "All":
        tone_map = {
            'Happy': 'joy',
            'Surprising': 'surprise',
            'Angry': 'anger',
            'Suspenseful': 'fear',
            'Sad': 'sadness'
        }
        if tone in tone_map:
            emotion_col = tone_map[tone]
            book_recs = book_recs.sort_values(by=emotion_col, ascending=False)
    
    book_recs = book_recs.sort_values('score').head(final_top_k)
    
    return book_recs

@app.route('/')
def home():
    categories = ["All"] + sorted(books["simple_categories"].unique())
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    return render_template('book_rec.html', categories=categories, tones=tones)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get('query', '')
    category = data.get('category', 'All')
    tone = data.get('tone', 'All')
    
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc = " ".join(description.split()[:30]) + "..."
        
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]
            
        results.append({
            'image': row["large_thumbnail"],
            'title': row['title'],
            'authors': authors_str,
            'description': truncated_desc
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)