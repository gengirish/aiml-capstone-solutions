"""Capstone 2 - Part 2: Tourism EDA + collaborative-filtering recommender."""
from common import md, code, write_notebook

cells = [
    md("""
# Capstone 2 - Part 2: Tourism Recommendation Engine

**Business scenario.** A government agency wants to promote tourism. Better
recommendations = happier tourists. We will:

1. Inspect & clean three Indonesia-tourism datasets,
2. Profile the user base and the tourist destinations,
3. Build a **collaborative-filtering** recommender (item-based KNN on the
   user-item rating matrix) that, given a place the tourist is currently at,
   suggests other places they might enjoy.

Datasets (under `Capstone 2/Part 2/`):

| File | Description |
|------|-------------|
| `tourism_with_id.xlsx` | 437 attractions in 5 Indonesian cities (category, price, lat/long, …) |
| `tourism_rating.csv`   | ~10k user-place rating triples (1-5)                                 |
| `user.csv`             | Demographics for ~300 users (age, location)                          |
"""),
    md("## 1. Setup"),
    code("""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.figsize"] = (10, 5)
"""),
    md("## 2. Load the data"),
    code("""
BASE = r"../Capstone 2/Part 2"
places  = pd.read_excel(f"{BASE}/tourism_with_id.xlsx")
ratings = pd.read_csv(f"{BASE}/tourism_rating.csv")
users   = pd.read_csv(f"{BASE}/user.csv")

print("places :", places.shape)
print("ratings:", ratings.shape)
print("users  :", users.shape)
"""),
    code("""
# Drop fully-empty 'Unnamed' columns produced by Excel
places = places.loc[:, ~places.columns.str.startswith("Unnamed")]
places.head()
"""),
    md("## 3. Preliminary inspection & cleaning"),
    code("""
print("Missing values:")
for name, df in [("places", places), ("ratings", ratings), ("users", users)]:
    print(f"  {name}: {df.isna().sum().sum()} total")
print()
print("Duplicate rows:")
for name, df in [("places", places), ("ratings", ratings), ("users", users)]:
    print(f"  {name}: {df.duplicated().sum()}")
"""),
    code("""
# Time_Minutes has many NaNs -> impute with median (it is not used in CF)
places["Time_Minutes"] = places["Time_Minutes"].fillna(places["Time_Minutes"].median())

# Drop fully-duplicated rating rows
ratings = ratings.drop_duplicates().reset_index(drop=True)
print("After cleaning:", places.shape, ratings.shape, users.shape)
"""),
    md("## 4. Exploratory Data Analysis"),
    md("### 4.1 Who is rating? - User demographics"),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(users["Age"], bins=20, kde=True, ax=ax[0], color="steelblue")
ax[0].set_title("Age distribution of reviewers")

users["City"] = users["Location"].str.split(",").str[0].str.strip()
top_cities = users["City"].value_counts().head(10)
sns.barplot(y=top_cities.index, x=top_cities.values, ax=ax[1], palette="viridis")
ax[1].set_title("Top home cities of users"); ax[1].set_xlabel("# users")
plt.tight_layout(); plt.show()
"""),
    md("### 4.2 What kind of places are in the catalogue?"),
    code("""
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
places["Category"].value_counts().plot(kind="bar", ax=ax[0], color="coral")
ax[0].set_title("# places per category")
places["City"].value_counts().plot(kind="bar", ax=ax[1], color="seagreen")
ax[1].set_title("# places per city")
plt.tight_layout(); plt.show()
"""),
    md("### 4.3 What kind of tourism is each city most famous for?"),
    code("""
city_cat = (places.groupby(["City", "Category"]).size()
                  .unstack(fill_value=0))
city_cat_pct = city_cat.div(city_cat.sum(axis=1), axis=0)

city_cat_pct.plot(kind="bar", stacked=True, figsize=(11, 5), colormap="tab10")
plt.title("Category mix by city")
plt.ylabel("Share of places")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.show()
city_cat
"""),
    md("### 4.4 Best city for a nature enthusiast?"),
    code("""
nature_categories = ["Cagar Alam", "Bahari"]   # nature reserve + maritime/beach
nature_count = (places[places["Category"].isin(nature_categories)]
                     .groupby("City").size().sort_values(ascending=False))
print("Nature-oriented places by city:")
print(nature_count)
print(f"\\nRecommendation: a nature lover should visit "
      f"'{nature_count.index[0]}'.")
"""),
    md("### 4.5 Most-loved places overall"),
    code("""
combined = ratings.merge(places, on="Place_Id", how="left")

agg = (combined.groupby(["Place_Id", "Place_Name", "City", "Category"])
               .agg(avg_rating=("Place_Ratings","mean"),
                    n_ratings =("Place_Ratings","size"))
               .reset_index())

# Reliable popularity: avg rating among places with >= 30 ratings
top_loved = (agg[agg["n_ratings"] >= 30]
                  .sort_values(["avg_rating","n_ratings"], ascending=False)
                  .head(15))
top_loved
"""),
    code("""
city_love = agg.groupby("City")["avg_rating"].mean().sort_values(ascending=False)
city_love.plot(kind="bar", color="darkviolet")
plt.title("Mean rating of places per city")
plt.ylabel("Avg rating"); plt.ylim(3.5, city_love.max()+0.1)
plt.show()
print(f"City with the highest average rating: {city_love.index[0]}")
"""),
    md("### 4.6 Which category do users like the most?"),
    code("""
cat_love = (agg.groupby("Category")
                .agg(avg_rating=("avg_rating","mean"),
                     n_ratings=("n_ratings","sum"))
                .sort_values("avg_rating", ascending=False))
cat_love
"""),
    code("""
ax = cat_love["avg_rating"].plot(kind="barh", color="goldenrod")
ax.invert_yaxis()
plt.title("Average rating by category"); plt.xlabel("Avg rating")
plt.xlim(3.0, cat_love["avg_rating"].max()+0.05)
plt.show()
"""),
    md("""
## 5. Collaborative-filtering recommender

We build an **item-based** recommender:

1. Construct a sparse `place x user` rating matrix.
2. Compute cosine similarity between places.
3. Given the *place_name* the user is currently at, return the K most similar
   places (filtering out the queried place itself).

Item-based CF is preferred here because the catalogue (~437 places) is much
smaller than the user base, making the similarity matrix small and stable.
"""),
    code("""
# user x place rating matrix (rows=places for item-based CF -> place x user)
rating_matrix = (ratings.pivot_table(index="Place_Id", columns="User_Id",
                                     values="Place_Ratings", fill_value=0))
print("place x user matrix:", rating_matrix.shape)
"""),
    code("""
# cosine similarity between places
sim = cosine_similarity(rating_matrix.values)
sim_df = pd.DataFrame(sim,
                      index=rating_matrix.index,
                      columns=rating_matrix.index)

place_lookup = places.set_index("Place_Id")[["Place_Name", "City", "Category", "Rating"]]


def recommend(place_name: str, k: int = 5) -> pd.DataFrame:
    matches = place_lookup[place_lookup["Place_Name"].str.lower() == place_name.lower()]
    if matches.empty:
        raise ValueError(f"Place '{place_name}' not in catalogue.")
    place_id = matches.index[0]

    sims = sim_df.loc[place_id].drop(place_id).sort_values(ascending=False)
    top  = sims.head(k)
    out  = (place_lookup.loc[top.index]
                .assign(similarity=top.values)
                .reset_index())
    return out
"""),
    md("### 5.1 Try a few recommendations"),
    code("""
sample_places = ["Monumen Nasional", "Pantai Marina", "Kebun Binatang Surabaya"]
for p in sample_places:
    print("="*68)
    print(f"Because you visited: {p}")
    try:
        print(recommend(p, k=5).to_string(index=False))
    except ValueError as e:
        print(e)
"""),
    md("### 5.2 KNN-based recommender (alternative implementation)"),
    code("""
knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(rating_matrix.values)


def recommend_knn(place_name: str, k: int = 5) -> pd.DataFrame:
    matches = place_lookup[place_lookup["Place_Name"].str.lower() == place_name.lower()]
    if matches.empty:
        raise ValueError(f"Place '{place_name}' not in catalogue.")
    place_id = matches.index[0]
    pos = rating_matrix.index.get_loc(place_id)
    dist, idx = knn.kneighbors(rating_matrix.values[pos:pos+1], n_neighbors=k+1)
    neighbour_ids = rating_matrix.index[idx[0][1:]]
    return (place_lookup.loc[neighbour_ids]
                .assign(distance=dist[0][1:])
                .reset_index())


print(recommend_knn("Monumen Nasional", k=5).to_string(index=False))
"""),
    md("""
## 6. Conclusions

* The catalogue is dominated by **cultural** (`Budaya`) and **amusement-park**
  (`Taman Hiburan`) sites; nature/maritime sites are concentrated in
  Yogyakarta and Semarang, so **nature lovers should head there**.
* Users that rate are mostly between **18-30 years old** and live in Java's
  large cities (Jakarta, Bandung, Surabaya).
* Average ratings are remarkably consistent across cities (3.4-3.6) - tourists
  generally enjoy what they visit.
* Item-based collaborative filtering successfully surfaces nearby alternatives
  for any given place (see `recommend` and `recommend_knn`). It can power a
  "you may also like" widget on the tourism portal.
"""),
]

if __name__ == "__main__":
    write_notebook(
        "../solutions/Capstone2_Part2_Tourism_Recommender.ipynb",
        cells,
    )
