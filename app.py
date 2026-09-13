import pandas as pd
import os

print("=" * 70)
print("TEXT EMBEDDINGS - RESULTS VIEWER")
print("=" * 70)

file_path = "data/similarity_results.csv"

if os.path.exists(file_path):

    results = pd.read_csv(file_path)

    print("\nSimilarity results loaded successfully!")

    print("\nTotal sentence pairs:",
          len(results))

    print("\n" + "=" * 70)
    print("TOP 5 MOST SIMILAR SENTENCE PAIRS")
    print("=" * 70)

    top_5 = results.head(5)

    for i, row in top_5.iterrows():

        print(f"\nPair {i + 1}")

        print("\nText 1:")
        print(row["Text 1"])

        print("\nText 2:")
        print(row["Text 2"])

        print(
            "\nCosine Similarity:",
            row["Cosine Similarity"]
        )

        print("-" * 70)

else:

    print("\nSimilarity results file not found.")

    print(
        "Please run embedding.py first."
    )