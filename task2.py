import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from scipy.sparse.linalg import svds



# Main execution
def main():
    model = LookalikeModel()
    customer_profiles = model.create_customer_profiles()
    target_customers = [f'C{str(i).zfill(4)}' for i in range(1, 21)]
    
    print("\nGenerating Recommendations...")
    trad_recommendations = model.get_lookalikes(customer_profiles, target_customers)
    trad_metrics = model.evaluate_recommendations(trad_recommendations, model.test_transactions)
    model.print_evaluation_results(trad_metrics, "Traditional")
    
    hybrid_recommendations = model.create_hybrid_recommendations(customer_profiles, target_customers)
    hybrid_metrics = model.evaluate_recommendations(hybrid_recommendations, model.test_transactions)
    model.print_evaluation_results(hybrid_metrics, "Hybrid")
    
    # Print example recommendations
    print("\nExample Recommendations:")
    print("=" * 50)
    for customer in target_customers[:3]:
        print(f"\nCustomer {customer}:")
        print("Cosine Similarity with Feature Weights:")
        if customer in trad_recommendations:
            for rec_customer, score in trad_recommendations[customer]:
                print(f"  - {rec_customer} (similarity: {score:.3f})")
        print("\nEnsemble of ML Models:")
        if customer in hybrid_recommendations:
            for rec_customer, score in hybrid_recommendations[customer]:
                print(f"  - {rec_customer} (similarity: {score:.3f})")

if __name__ == "__main__":
    main()