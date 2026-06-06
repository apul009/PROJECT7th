import numpy as np
import pickle

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def convolve2d(image, kernel):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.sum(image[i:i+k_h, j:j+k_w] * kernel)
    return output

def apply_filters(image, filters):
    feature_maps = []
    for f in filters:
        fm = convolve2d(image, f)
        fm = relu(fm)
        feature_maps.append(fm)
    return np.array(feature_maps)

def max_pool(feature_map, size=2):
    h, w = feature_map.shape
    out_h = h // size
    out_w = w // size
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            output[i, j] = np.max(feature_map[i*size:(i+1)*size, j*size:(j+1)*size])
    return output

class SimpleCNN:
    def __init__(self, num_classes=28):
        np.random.seed(42)
        self.num_classes = num_classes
        self.filters = np.random.randn(8, 3, 3) * 0.1
        self.fc_input_size = 8 * 13 * 13
        self.W1 = np.random.randn(256, self.fc_input_size) * 0.01
        self.b1 = np.zeros(256)
        self.W2 = np.random.randn(num_classes, 256) * 0.01
        self.b2 = np.zeros(num_classes)

    def forward(self, image):
        feature_maps = apply_filters(image, self.filters)
        pooled = np.array([max_pool(fm) for fm in feature_maps])
        flat = pooled.flatten()
        z1 = self.W1 @ flat + self.b1
        a1 = relu(z1)
        z2 = self.W2 @ a1 + self.b2
        probs = softmax(z2)
        self.cache = (flat, z1, a1, z2)
        return probs

    def predict(self, image):
        return np.argmax(self.forward(image))

    def predict_with_confidence(self, image):
        probs = self.forward(image)
        idx = np.argmax(probs)
        confidence = float(probs[idx]) * 100
        return idx, confidence

    def compute_loss(self, probs, true_label):
        return -np.log(probs[true_label] + 1e-9)

    def backward(self, probs, true_label, lr=0.01):
        flat, z1, a1, z2 = self.cache
        dz2 = probs.copy()
        dz2[true_label] -= 1
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        da1 = self.W2.T @ dz2
        dz1 = da1 * relu_derivative(z1)
        dW1 = np.outer(dz1, flat)
        db1 = dz1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def save(self, path='model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump({
                'filters': self.filters,
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2,
                'num_classes': self.num_classes
            }, f)
        print("Model saved to " + path)

    def load(self, path='model.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.filters = data['filters']
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.num_classes = data['num_classes']
        print("Model loaded from " + path)