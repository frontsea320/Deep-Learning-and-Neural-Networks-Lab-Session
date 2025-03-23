import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import mnist

def load_weights(model, weights_dir="weights"):
    model.W1 = np.load(os.path.join(weights_dir, "W1.npy"))
    model.W2 = np.load(os.path.join(weights_dir, "W2.npy"))
    model.W3 = np.load(os.path.join(weights_dir, "W3.npy"))
    model.W_skip = np.load(os.path.join(weights_dir, "W_skip.npy"))

def visualize_predictions(model, num_samples=5, save_path="visualization"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    (x_test, y_test), _ = mnist.load_data()
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    sample_images = x_test[indices] / 255.0
    sample_labels = y_test[indices]

    plt.figure(figsize=(num_samples * 2, 4))
    for i, (img, label) in enumerate(zip(sample_images, sample_labels)):
        input_img = img.reshape(1, -1)
        bias = np.ones(shape=[input_img.shape[0], 1])
        input_img = np.concatenate([input_img, bias], axis=1)

        # Forward pass
        h1 = model.relu1.forward(model.mul_h1.forward(input_img, model.W1))
        h2 = model.relu2.forward(model.mul_h2.forward(h1, model.W2))
        h2_res = h2 + np.matmul(h1, model.W_skip)
        output = model.softmax.forward(model.mul_h3.forward(h2_res, model.W3))

        prediction = np.argmax(output)

        plt.subplot(1, num_samples, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {label}\nPred: {prediction}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "predictions.png"))
    plt.show()
