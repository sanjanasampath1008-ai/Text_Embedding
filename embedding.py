import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# --------------------------------------------------
# 1. CREATE DATA FOLDER
# --------------------------------------------------

os.makedirs("data", exist_ok=True)

# --------------------------------------------------
# 2. COLLECT 20 SENTENCES
# --------------------------------------------------

texts = [
    "Artificial intelligence enables computers to perform tasks that normally require human intelligence.",
    "Machine learning allows computers to learn patterns from data without being explicitly programmed.",
    "Deep learning uses neural networks with multiple layers to process complex information.",
    "Natural language processing helps computers understand and generate human language.",
    "Climate change is causing long-term changes in global temperatures and weather patterns.",
    "Renewable energy sources such as solar and wind can reduce dependence on fossil fuels.",
    "Deforestation contributes to biodiversity loss and increases carbon dioxide levels in the atmosphere.",
    "Electric vehicles use electric motors instead of traditional internal combustion engines.",
    "The human brain contains billions of neurons that communicate through electrical and chemical signals.",
    "Regular physical activity can improve cardiovascular health and overall fitness.",
    "The internet allows computers and people around the world to exchange information rapidly.",
    "Cybersecurity protects computer systems and networks from unauthorized access and attacks.",
    "The Earth revolves around the Sun and completes one orbit approximately every 365 days.",
    "The Moon is Earth's natural satellite and takes about 27 days to orbit Earth.",
    "Photosynthesis allows green plants to convert light energy into chemical energy.",
    "Water is essential for life and is found in oceans, rivers, lakes, glaciers, and underground sources.",
    "Economic growth refers to an increase in the production of goods and services in an economy.",
    "Inflation is a general increase in the prices of goods and services over time.",
    "Books provide information, knowledge, and entertainment to readers.",
    "Online education allows students to learn through digital platforms from different locations."
]

print("Total sentences:", len(texts))

# --------------------------------------------------
# 3. LOAD EMBEDDING MODEL
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# 4. GENERATE EMBEDDINGS
# --------------------------------------------------

embeddings = model.encode(texts)

print("Embeddings generated successfully!")
print("Embedding shape:", embeddings.shape)

# --------------------------------------------------
# 5. SAVE EMBEDDINGS
# --------------------------------------------------

np.save("data/embeddings.npy", embeddings)

print("Embeddings saved to:")
print("data/embeddings.npy")

# --------------------------------------------------
# 6. CALCULATE COSINE SIMILARITY
# --------------------------------------------------

similarity_matrix = cosine_similarity(embeddings)

# --------------------------------------------------
# 7. COMPARE EVERY SENTENCE WITH EVERY OTHER
# --------------------------------------------------

results = []

for i in range(len(texts)):

    for j in range(i + 1, len(texts)):

        results.append({
            "Text 1": texts[i],
            "Text 2": texts[j],
            "Cosine Similarity": round(
                similarity_matrix[i][j], 4
            )
        })

# --------------------------------------------------
# 8. CONVERT TO DATAFRAME
# --------------------------------------------------

results_df = pd.DataFrame(results)

# --------------------------------------------------
# 9. SORT BY SIMILARITY
# --------------------------------------------------

results_df = results_df.sort_values(
    by="Cosine Similarity",
    ascending=False
)

# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

results_df.to_csv(
    "data/similarity_results.csv",
    index=False
)

print("\nComparison completed!")

# --------------------------------------------------
# 11. DISPLAY TOP 5
# --------------------------------------------------

print("\nTop 5 most similar sentence pairs:")
print("=" * 70)

print(results_df.head(5).to_string(index=False))

print("\nResults saved to:")
print("data/similarity_results.csv")