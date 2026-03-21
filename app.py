from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import gc
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
print("Loading data...")
movies_df  = pd.read_csv("dataset/movies.csv")
ratings_df = pd.read_csv("dataset/ratings.csv")
tags_df    = pd.read_csv("dataset/tags.csv")
links_df   = pd.read_csv("dataset/links.csv")
agg = ratings_df.groupby("movieId")["rating"].agg(["mean","count"]).reset_index()
agg.columns = ["movieId","avgRating","numRatings"]
agg["avgRating"] = agg["avgRating"].round(2)
tags_agg = tags_df.groupby("movieId")["tag"].apply(
    lambda x: list(x.unique()[:5])
).reset_index()
df = movies_df.merge(agg, on="movieId", how="left")
df = df.merge(links_df[["movieId","tmdbId"]], on="movieId", how="left")
df = df.merge(tags_agg, on="movieId", how="left")
df["avgRating"]   = df["avgRating"].fillna(0)
df["numRatings"]  = df["numRatings"].fillna(0).astype(int)
df["tmdbId"]      = df["tmdbId"].fillna("").astype(str).str.replace(".0","",regex=False)
df["tag"]         = df["tag"].apply(lambda x: x if isinstance(x, list) else [])
df["year"]        = df["title"].str.extract(r"\((\d{4})\)")[0].fillna("N/A")
df["genres_list"] = df["genres"].apply(lambda x: x.split("|") if pd.notna(x) else [])
ALL_GENRES = sorted(set(
    g for genres in df["genres_list"]
    for g in genres if g != "(no genres listed)"
))
# ══════════════════════════════════════
# ML 1 — CONTENT-BASED (TF-IDF)
# ══════════════════════════════════════
print("Training Content-Based model...")
df["content"] = (
    df["genres_list"].apply(lambda x: " ".join(x)) + " " +
    df["tag"].apply(lambda x: " ".join(x))
)
tfidf         = TfidfVectorizer(stop_words="english")
tfidf_matrix  = tfidf.fit_transform(df["content"])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype('float32')
gc.collect()
movie_indices = pd.Series(df.index, index=df["movieId"])
print("✅ Content-Based ready!")

# ══════════════════════════════════════
# ML 2 — COLLABORATIVE FILTERING (SVD)
# ══════════════════════════════════════
print("Training SVD model (lightweight)...")
user_movie = ratings_df.pivot_table(
    index="userId", columns="movieId", values="rating"
).fillna(0)
matrix       = csr_matrix(user_movie.values)
U, sigma, Vt = svds(matrix, k=20)  # reduced from 50 to 20
sigma        = np.diag(sigma)
predicted    = np.dot(np.dot(U, sigma), Vt)
preds_df     = pd.DataFrame(
    predicted,
    index=user_movie.index,
    columns=user_movie.columns
)
del matrix, U, sigma, Vt  # free memory immediately
import gc
gc.collect()
print("SVD ready!")
# ══════════════════════════════════════
# ML 3 — HYBRID RECOMMENDER
# ══════════════════════════════════════
def hybrid_recommend(movie_id, top_n=6):
    if movie_id not in movie_indices.index:
        return []

    idx            = movie_indices[movie_id]
    content_scores = list(enumerate(content_sim[idx]))
    content_dict   = {i: score for i, score in content_scores}

    collab_dict = {}
    if movie_id in preds_df.columns:
        movie_col  = preds_df[movie_id]
        collab_raw = preds_df.corrwith(movie_col).dropna()
        c_min, c_max = collab_raw.min(), collab_raw.max()
        for mid, score in collab_raw.items():
            if mid in movie_indices.index:
                norm = (score - c_min) / (c_max - c_min + 1e-9)
                collab_dict[movie_indices[mid]] = norm

    all_idx = set(content_dict.keys()) | set(collab_dict.keys())
    hybrid  = {i: 0.5*content_dict.get(i,0) + 0.5*collab_dict.get(i,0) for i in all_idx}
    hybrid.pop(idx, None)

    top_idx = sorted(hybrid, key=hybrid.get, reverse=True)[:top_n]
    result  = df.iloc[top_idx][[
        "movieId","title","avgRating","numRatings","tmdbId","genres_list","year","tag"
    ]].copy()
    result["tmdbId"] = result["tmdbId"].fillna("").astype(str).str.replace(".0","",regex=False).str.strip()
    result = result.rename(columns={"movieId":"id","tag":"tags","genres_list":"genres"})
    return result.to_dict(orient="records")


def to_json(subset):
    return subset[[
        "movieId","title","avgRating","numRatings","tmdbId","tag","year","genres_list"
    ]].rename(columns={
        "movieId":"id","tag":"tags","genres_list":"genres"
    }).to_dict(orient="records")


@app.route("/")
def index():
    return render_template("index.html", genres=ALL_GENRES)

@app.route("/api/movies")
def get_movies():
    query       = request.args.get("q","").strip().lower()
    genre       = request.args.get("genre","").strip()
    sort_by     = request.args.get("sort","numRatings")
    min_ratings = int(request.args.get("min_ratings",0))
    page        = int(request.args.get("page",1))
    per_page    = int(request.args.get("per_page",20))

    filtered = df.copy()
    if query:        filtered = filtered[filtered["title"].str.lower().str.contains(query, na=False)]
    if genre:        filtered = filtered[filtered["genres_list"].apply(lambda g: genre in g)]
    if min_ratings:  filtered = filtered[filtered["numRatings"] >= min_ratings]

    if sort_by == "avgRating":  filtered = filtered.sort_values(["avgRating","numRatings"], ascending=False)
    elif sort_by == "year":     filtered = filtered.sort_values("year", ascending=False)
    else:                       filtered = filtered.sort_values("numRatings", ascending=False)

    total     = len(filtered)
    start     = (page-1)*per_page
    paginated = filtered.iloc[start:start+per_page]

    return jsonify({
        "movies": to_json(paginated),
        "total": total, "page": page,
        "pages": (total+per_page-1)//per_page
    })

@app.route("/api/movie/<int:movie_id>")
def get_movie(movie_id):
    row = df[df["movieId"] == movie_id]
    if row.empty: return jsonify({"error":"Not found"}), 404
    return jsonify(to_json(row)[0])

@app.route("/api/recommend/<int:movie_id>")
def recommend(movie_id):
    return jsonify(hybrid_recommend(movie_id))

@app.route("/api/stats")
def get_stats():
    return jsonify({
        "totalMovies":  len(df),
        "totalRatings": int(ratings_df.shape[0]),
        "totalUsers":   int(ratings_df["userId"].nunique()),
        "avgRating":    round(float(ratings_df["rating"].mean()), 2),
        "genres":       ALL_GENRES
    })
import requests as req
@app.route("/api/poster/<tmdb_id>")
def fetch_tmdb_poster(tmdb_id):
    try:
        clean_id = str(tmdb_id).replace(".0","").strip()
        if not clean_id or clean_id == "nan":
            return jsonify({"url": None})
        TMDB_KEY = "ca53af58900b285674d07dca69e7bf2e"
        proxies = {
            "http":  "http://98.8.195.160:443",
            "https": "http://98.8.195.160:443"
        }
        r = req.get(
            f"https://api.themoviedb.org/3/movie/{clean_id}?api_key={TMDB_KEY}",
            timeout=10,
            proxies=proxies
        )
        data = r.json()
        if data.get("poster_path"):
            return jsonify({"url": f"https://image.tmdb.org/t/p/w300{data['poster_path']}"})
    except Exception as e:
        print(f"Poster error: {e}")
    return jsonify({"url": None})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
 
