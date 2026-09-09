import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from encoding_data_03 import X
# Import the trained model objects from your 06 file
from prediction_data_06 import id3_model, c45_model, cart_model

models = {
    "ID3 Tree Structure": id3_model,
    "C4.5 Tree Structure": c45_model,
    "CART Tree Structure": cart_model
}

# Generate a visual plot for each tree
for title, model_obj in models.items():
    # Made the figure slightly taller (16x9) to give the branches breathing room
    plt.figure(figsize=(16, 9))
    
    plot_tree(
        model_obj,
        max_depth=3,  
        feature_names=list(X.columns),
        class_names=[str(c) for c in model_obj.classes_],
        filled=True,
        fontsize=7,        # Reduced font slightly to ensure text stays inside the boxes
        precision=1,       # Trims decimals (e.g., 7453.272 becomes 7453.3) so text doesn't stretch out
        impurity=False     # Hides the "entropy = " / "gini = " text line to make boxes smaller
    )
    
    plt.title(title, fontsize=14, pad=20)
    
    # Automatically spaces out the tree columns horizontally to prevent overlap
    plt.tight_layout()
    plt.show()
