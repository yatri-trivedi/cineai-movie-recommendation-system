from flask import Flask, send_from_directory jsonify, request
import os
import pandas as pd
import numpy as np
import gc
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob   # pip install textblob
import requests as req
import google.genai as genai

app = Flask(__name__, static_folder='.', static_url_path='')


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
df["description"] = ""
df['ratingCount'] = df['numRatings']

ALL_GENRES = sorted(set(
    g for genres in df["genres_list"]
    for g in genres if g != "(no genres listed)"
))

# ══════════════════════════════════════
# NEW — SENTIMENT ANALYSIS ON TAGS
# ══════════════════════════════════════
print("Running Sentiment Analysis on tags...")

def analyse_sentiment(tags):
    """Analyse sentiment of movie tags using TextBlob."""
    if not tags:
        return {"label": "Neutral", "score": 0.0, "emoji": "😐", "color": "#8b949e"}
    combined = " ".join(tags)
    blob     = TextBlob(combined)
    score    = round(blob.sentiment.polarity, 2)   # -1 to +1
    if score >= 0.3:
        return {"label": "Positive", "score": score, "emoji": "😊", "color": "#2dd4bf"}
    elif score <= -0.2:
        return {"label": "Negative", "score": score, "emoji": "😟", "color": "#f87171"}
    else:
        return {"label": "Neutral",  "score": score, "emoji": "😐", "color": "#8b949e"}

df["sentiment"] = df["tag"].apply(analyse_sentiment)
print("Sentiment Analysis ready!")

# ══════════════════════════════════════
# NEW — MOOD-BASED GENRE MAPPING
# ══════════════════════════════════════
MOOD_MAP = {
    "happy":     {"genres": ["Comedy","Animation","Family","Musical"],
                  "emoji": "😄", "label": "Happy",      "color": "#facc15"},
    "sad":       {"genres": ["Drama","Romance"],
                  "emoji": "😢", "label": "Sad",        "color": "#60a5fa"},
    "excited":   {"genres": ["Action","Adventure","Thriller"],
                  "emoji": "🤩", "label": "Excited",    "color": "#f97316"},
    "scared":    {"genres": ["Horror","Mystery","Thriller"],
                  "emoji": "😱", "label": "Scared",     "color": "#a78bfa"},
    "romantic":  {"genres": ["Romance","Drama"],
                  "emoji": "❤️",  "label": "Romantic",  "color": "#fb7185"},
    "curious":   {"genres": ["Documentary","Sci-Fi","Mystery"],
                  "emoji": "🤔", "label": "Curious",    "color": "#34d399"},
    "nostalgic": {"genres": ["Animation","Family","Musical","Comedy"],
                  "emoji": "🌅", "label": "Nostalgic",  "color": "#fbbf24"},
    "tense":     {"genres": ["Crime","Thriller","War","Film-Noir"],
                  "emoji": "😤", "label": "Tense",      "color": "#ef4444"},
}
# ══════════════════════════════════════
# ML 1 — CONTENT-BASED (TF-IDF)
# ══════════════════════════════════════
print("Training Content-Based model...")
df["content"] = (
    df["genres_list"].apply(lambda x: " ".join(x)) + " " +
    df["tag"].apply(lambda x: " ".join(x))
)
tfidf        = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["content"])
content_sim  = cosine_similarity(tfidf_matrix, tfidf_matrix).astype("float32")
gc.collect()
movie_indices = pd.Series(df.index, index=df["movieId"])
print("✅ Content-Based ready!")
# ══════════════════════════════════════
# ML 2 — COLLABORATIVE FILTERING (SVD)
# ══════════════════════════════════════
print("Training SVD model...")
user_movie   = ratings_df.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
matrix       = csr_matrix(user_movie.values)
U, sigma, Vt = svds(matrix, k=20)
sigma        = np.diag(sigma)
predicted    = np.dot(np.dot(U, sigma), Vt)
preds_df     = pd.DataFrame(predicted, index=user_movie.index, columns=user_movie.columns)
del matrix, U, sigma, Vt
gc.collect()
print("✅ SVD ready!")
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
        "movieId","title","avgRating","numRatings","tmdbId","genres_list","year","tag","sentiment"
    ]].copy()
    result["tmdbId"] = result["tmdbId"].fillna("").astype(str).str.replace(".0","",regex=False).str.strip()
    result = result.rename(columns={"movieId":"id","tag":"tags","genres_list":"genres"})
    return result.to_dict(orient="records")
def to_json(subset):
    result = subset[[
        "movieId","title","avgRating","numRatings","tmdbId","tag","year","genres_list","sentiment","description"
    ]].rename(columns={
        "movieId":"id","tag":"tags","genres_list":"genres"
    }).to_dict(orient="records")
    
    # Add posterUrl field (will be populated by async frontend fetch)
    for movie in result:
        movie["posterUrl"] = None
    
    return result
# ══════════════════════════════════════
# ROUTES
# ══════════════════════════════════════
@app.route("/")     
def index():
    moods = [{"key": k, **v} for k, v in MOOD_MAP.items()]
    return send_from_directory('.', 'index.html')
@app.route("/api/movies")
def get_movies():
    query        = request.args.get("q","").strip().lower()
    genre        = request.args.get("genre","").strip()
    sort_by      = request.args.get("sort","numRatings")
    min_ratings  = int(request.args.get("min_ratings",0))
    min_year     = int(request.args.get("min_year",0))
    max_year     = int(request.args.get("max_year",2099))
    min_rating   = float(request.args.get("min_rating",0))
    page         = int(request.args.get("page",1))
    per_page     = int(request.args.get("per_page",20))
    filtered = df.copy()
    if query:        filtered = filtered[filtered["title"].str.lower().str.contains(query, na=False)]
    if genre:        filtered = filtered[filtered["genres_list"].apply(lambda g: genre in g)]
    if min_ratings:  filtered = filtered[filtered["numRatings"] >= min_ratings]
    if min_rating > 0: filtered = filtered[filtered["avgRating"] >= min_rating]
    # Year filtering
    filtered = filtered[filtered["year"].apply(lambda x: x != "N/A" and min_year <= int(x) <= max_year if x != "N/A" else False)]
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

@app.route("/api/personalized-movies")
def get_personalized_movies():
    """Get personalized movie recommendations based on user profile and context"""
    import json
    
    query = request.args.get("q", "").strip().lower()
    genre = request.args.get("genre", "").strip()
    sort_by = request.args.get("sort", "numRatings")
    min_ratings = int(request.args.get("min_ratings", 0))
    min_year = int(request.args.get("min_year", 0))
    max_year = int(request.args.get("max_year", 2099))
    min_rating = float(request.args.get("min_rating", 0))
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    
    preferred_genres_str = request.args.get("preferred_genres", "")
    context_str = request.args.get("context", "")
    
    preferred_genres = []
    context = {}
    
    try:
        if preferred_genres_str:
            preferred_genres = json.loads(preferred_genres_str)
        if context_str:
            context = json.loads(context_str)
    except:
        pass
    
    filtered = df.copy()
    
    # Apply standard filters
    if query:
        filtered = filtered[filtered["title"].str.lower().str.contains(query, na=False)]
    if genre:
        filtered = filtered[filtered["genres_list"].apply(lambda g: genre in g)]
    if min_ratings:
        filtered = filtered[filtered["numRatings"] >= min_ratings]
    if min_rating > 0:
        filtered = filtered[filtered["avgRating"] >= min_rating]
    
    # Year filtering
    filtered = filtered[filtered["year"].apply(lambda x: x != "N/A" and min_year <= int(x) <= max_year if x != "N/A" else False)]
    
    # Apply personalization
    if preferred_genres or context:
        filtered['personalization_score'] = 0
        
        # Boost movies matching preferred genres
        if preferred_genres:
            filtered['personalization_score'] += filtered['genres_list'].apply(
                lambda g: sum(5 for pg in preferred_genres if pg in g)
            )
        
        # Boost movies based on context
        if context.get('timeOfDay') == 'evening':
            family_genres = ['Family', 'Animation', 'Comedy']
            filtered['personalization_score'] += filtered['genres_list'].apply(
                lambda g: sum(3 for fg in family_genres if fg in g)
            )
        elif context.get('isWeekend'):
            weekend_genres = ['Action', 'Adventure', 'Thriller']
            filtered['personalization_score'] += filtered['genres_list'].apply(
                lambda g: sum(3 for wg in weekend_genres if wg in g)
            )
        
        # Sort by personalization score, then by quality
        filtered = filtered.sort_values(
            ['personalization_score', 'avgRating', 'numRatings'],
            ascending=[False, False, False]
        )
    else:
        # Default sorting
        if sort_by == "avgRating":
            filtered = filtered.sort_values(["avgRating", "numRatings"], ascending=False)
        elif sort_by == "year":
            filtered = filtered.sort_values("year", ascending=False)
        else:
            filtered = filtered.sort_values("numRatings", ascending=False)
    
    total = len(filtered)
    start = (page - 1) * per_page
    paginated = filtered.iloc[start:start + per_page]
    
    return jsonify({
        "movies": to_json(paginated),
        "total": total,
        "page": page,
        "pages": (total + per_page - 1) // per_page
    })


@app.route("/api/movie/<int:movie_id>")
def get_movie(movie_id):
    row = df[df["movieId"] == movie_id]
    if row.empty:
        return jsonify({"error": "Not found"}), 404
    movie_data = to_json(row)[0]

    # Fetch description from TMDB if available (for MovieLens movies)
    if movie_data.get("tmdbId") and not movie_data.get("description"):
        try:
            tmdb_id = movie_data["tmdbId"]
            r = req.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key=ca53af58900b285674d07dca69e7bf2e",
                timeout=5
            )
            tmdb_data = r.json()
            if tmdb_data.get("overview"):
                movie_data["description"] = tmdb_data["overview"]
        except:
            pass

    return jsonify(movie_data)


@app.route("/api/recommend/<int:movie_id>")
def recommend(movie_id):
    if movie_id not in movie_indices.index:
        return jsonify([])
    recs = hybrid_recommend(movie_id)
    
    # Add explanations
    if movie_id in df["movieId"].values:
        source_movie = df[df["movieId"] == movie_id].iloc[0]
        source_genres = set(source_movie["genres_list"])
        
        for rec in recs:
            rec_movie = df[df["movieId"] == rec["id"]].iloc[0]
            rec_genres = set(rec_movie["genres_list"])
            matching_genres = list(source_genres & rec_genres)
            
            explanation = "Based on similar content"
            if matching_genres:
                explanation = f"Shares {', '.join(matching_genres[:2])} genre(s)"
            
            rec["explanation"] = explanation
    
    return jsonify(recs)


# ══════════════════════════════════════
# MOOD-BASED RECOMMENDATIONS
# ══════════════════════════════════════
@app.route("/api/mood/<mood_key>")
def mood_recommend(mood_key):
    if mood_key not in MOOD_MAP:
        return jsonify({"error": "Unknown mood"}), 400

    target_genres = MOOD_MAP[mood_key]["genres"]
    mood_info     = MOOD_MAP[mood_key]

    # Filter movies that match ANY of the mood genres
    mask    = df["genres_list"].apply(lambda g: any(genre in g for genre in target_genres))
    matches = df[mask & (df["numRatings"] >= 20)].copy()

    # Sort by weighted rating
    C = matches["avgRating"].mean()
    m = matches["numRatings"].quantile(0.6)
    matches["weighted"] = (
        (matches["numRatings"] / (matches["numRatings"] + m)) * matches["avgRating"] +
        (m / (matches["numRatings"] + m)) * C
    )
    top = matches.nlargest(12, "weighted")

    return jsonify({
        "mood":    mood_info,
        "movies":  to_json(top)
    })


# ══════════════════════════════════════
# TRAILER via TMDB API
# ══════════════════════════════════════
@app.route("/api/trailer/<tmdb_id>")
def fetch_trailer(tmdb_id):
    try:
        clean_id = str(tmdb_id).replace(".0","").strip()
        if not clean_id or clean_id in ("nan",""):
            return jsonify({"trailer_key": None})

        TMDB_KEY = "ca53af58900b285674d07dca69e7bf2e"
        r = req.get(
            f"https://api.themoviedb.org/3/movie/{clean_id}/videos?api_key={TMDB_KEY}",
            timeout=10
        )
        data = r.json()

        # Find official YouTube trailer
        videos = data.get("results", [])
        trailer = next(
            (v for v in videos if v.get("type") == "Trailer" and v.get("site") == "YouTube"),
            None
        )
        if not trailer:
            # fallback — any YouTube video
            trailer = next((v for v in videos if v.get("site") == "YouTube"), None)

        return jsonify({"trailer_key": trailer["key"] if trailer else None})

    except Exception as e:
        print(f"Trailer error for {tmdb_id}: {type(e).__name__}: {e}")
        return jsonify({"trailer_key": None})  # ← Returns None → "No Trailer Found"


@app.route("/api/stats")
def get_stats():
    return jsonify({
        "totalMovies":  len(df),
        "totalRatings": int(ratings_df.shape[0]),
        "totalUsers":   int(ratings_df["userId"].nunique()),
        "avgRating":    round(float(ratings_df["rating"].mean()), 2),
        "genres":       ALL_GENRES
    })


@app.route("/api/poster/<tmdb_id>")
def fetch_tmdb_poster(tmdb_id):
    try:
        clean_id = str(tmdb_id).replace(".0","").strip()
        if not clean_id or clean_id == "nan":
            return jsonify({"url": None})
        TMDB_KEY = "ca53af58900b285674d07dca69e7bf2e"
        r    = req.get(
            f"https://api.themoviedb.org/3/movie/{clean_id}?api_key={TMDB_KEY}",
            timeout=10
        )
        data = r.json()
        if data.get("poster_path"):
            return jsonify({"url": f"https://image.tmdb.org/t/p/w300{data['poster_path']}"})
    except Exception as e:
        print(f"Poster error for {tmdb_id}: {type(e).__name__}: {e}")
    return jsonify({"url": None})


# ══════════════════════════════════════
# NEW — ENHANCED MOVIE DETAILS
# ══════════════════════════════════════
@app.route("/api/movie-full/<int:movie_id>")
def get_movie_full(movie_id):
    """Get full movie details including cast, crew, budget, runtime"""
    row = df[df["movieId"] == movie_id]
    if row.empty:
        return jsonify({"error": "Not found"}), 404
    
    movie_data = to_json(row)[0]
    
    # Fetch extended details from TMDB
    if movie_data.get("tmdbId"):
        try:
            TMDB_KEY = "ca53af58900b285674d07dca69e7bf2e"
            r = req.get(
                f"https://api.themoviedb.org/3/movie/{movie_data['tmdbId']}?api_key={TMDB_KEY}",
                timeout=5
            )
            tmdb = r.json()
            movie_data["description"] = tmdb.get("overview", "")
            movie_data["runtime"] = tmdb.get("runtime", 0)
            movie_data["budget"] = tmdb.get("budget", 0)
            movie_data["revenue"] = tmdb.get("revenue", 0)
            movie_data["production_companies"] = [c["name"] for c in tmdb.get("production_companies", [])]
            movie_data["release_date"] = tmdb.get("release_date", "")
            movie_data["popularity"] = tmdb.get("popularity", 0)
            movie_data["vote_count"] = tmdb.get("vote_count", 0)
        except:
            pass
    
    return jsonify(movie_data)


# ══════════════════════════════════════
# NEW — CAST & CREW
# ══════════════════════════════════════
@app.route("/api/cast/<tmdb_id>")
def get_cast(tmdb_id):
    """Get cast and crew for a movie"""
    try:
        clean_id = str(tmdb_id).replace(".0","").strip()
        if not clean_id or clean_id == "nan":
            return jsonify({"cast": [], "crew": []})
        
        TMDB_KEY = "ca53af58900b285674d07dca69e7bf2e"
        r = req.get(
            f"https://api.themoviedb.org/3/movie/{clean_id}/credits?api_key={TMDB_KEY}",
            timeout=5
        )
        data = r.json()
        
        cast = [{
            "name": c["name"],
            "character": c["character"],
            "profile_path": c.get("profile_path")
        } for c in data.get("cast", [])[:10]]
        
        crew = [{
            "name": c["name"],
            "job": c["job"],
            "profile_path": c.get("profile_path")
        } for c in data.get("crew", []) if c["job"] in ["Director", "Producer", "Writer"]][:5]
        
        return jsonify({"cast": cast, "crew": crew})
    except Exception as e:
        print(f"Cast error for {tmdb_id}: {e}")
        return jsonify({"cast": [], "crew": []})


# ══════════════════════════════════════
# NEW — COLLECTIONS
# ══════════════════════════════════════
@app.route("/api/collections/<collection_type>")
def get_collections(collection_type):
    """Get curated movie collections"""
    
    if collection_type == "top-rated":
        filtered = df[df["numRatings"] >= 50].sort_values(["avgRating","numRatings"], ascending=False).head(20)
        title = "Top Rated Movies"
        subtitle = "Highest rated with significant votes"
    
    elif collection_type == "trending":
        filtered = df.nlargest(20, "numRatings")
        title = "Trending Now"
        subtitle = "Most popular right now"
    
    elif collection_type == "hidden-gems":
        filtered = df[(df["numRatings"] >= 20) & (df["numRatings"] <= 500)].sort_values("avgRating", ascending=False).head(20)
        title = "Hidden Gems"
        subtitle = "High-rated but underrated"
    
    elif collection_type == "recent":
        filtered = df[df["year"] != "N/A"].copy()
        filtered["year"] = pd.to_numeric(filtered["year"], errors="coerce")
        filtered = filtered.dropna(subset=["year"]).sort_values("year", ascending=False).head(20)
        title = "Recent Releases"
        subtitle = "Latest movies"
    
    else:
        return jsonify({"error": "Unknown collection"}), 404
    
    return jsonify({
        "collection": collection_type,
        "title": title,
        "subtitle": subtitle,
        "movies": to_json(filtered)
    })


# ══════════════════════════════════════
# NEW — MOVIES BY DIRECTOR/ACTOR
# ══════════════════════════════════════
@app.route("/api/movies-by-director/<director_name>")
def movies_by_director(director_name):
    """Get movies by director (requires TMDB integration for full support)"""
    # This is a placeholder - real implementation would search TMDB
    return jsonify({"movies": [], "message": "Director search requires extended TMDB integration"})


# ══════════════════════════════════════
# NEW — WATCHLIST & RATINGS (stored client-side)
# ══════════════════════════════════════
@app.route("/api/saved-movies", methods=["GET"])
def get_saved_movies():
    """Placeholder - client stores in localStorage"""
    return jsonify({
        "message": "Watchlist is stored in browser localStorage",
        "note": "Use browser DevTools > Application > Local Storage to view"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
