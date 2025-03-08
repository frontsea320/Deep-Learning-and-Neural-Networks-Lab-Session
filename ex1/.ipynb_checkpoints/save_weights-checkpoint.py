import numpy as np
import os

def save_weights(model, save_dir="weights"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, "W1.npy"), model.W1)
    np.save(os.path.join(save_dir, "W2.npy"), model.W2)
    np.save(os.path.join(save_dir, "W3.npy"), model.W3)
    np.save(os.path.join(save_dir, "W_skip.npy"), model.W_skip)
    print(f"Weights have been saved successfully to '{save_dir}'")
