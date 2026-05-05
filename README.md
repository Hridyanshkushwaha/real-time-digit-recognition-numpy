# Real-Time Handwritten Digit Recognition (NumPy + OpenCV)
Real-time handwritten digit recognition using a neural network built from scratch (NumPy) with OpenCV webcam integration and MNIST-based preprocessing.
A from-scratch neural network (no TensorFlow/PyTorch) that recognizes handwritten digits in real-time using a webcam. The model is trained on the MNIST dataset and performs live inference on user-drawn digits inside a camera frame.
<h2>🚀Features</h2>
<ol><li>Fully implemented Neural Network from scratch (NumPy only)</li>
<li>Real-time digit recognition using OpenCV webcam feed</li>
<li>Custom MNIST-style preprocessing pipeline</li>
<li>Smooth prediction confidence visualization</li>
<li>Lightweight - no heavy ML frameworks required</li>
<li>Auto-save and load trained weights</li></ol>

<h2>🏗️ Architecture</h2>
<p>The neural network follows a simple feedforward architecture:</p>

```
Input Layer:   784 neurons (28×28 image)
Hidden Layer 1: 256 neurons (ReLU)
Hidden Layer 2: 128 neurons (ReLU)
Output Layer:  10 neurons (Softmax)
```
<h4>Key Techniques Used:</h4>
<ul><li>He Initialization (for ReLU stability)</li>
<li>Mini-batch Gradient Descent</li>
<li>Momentum-based Optimization</li>
<li>Cross-Entropy Loss</li></ul>
<h2>📂 Project Structure</h2>

```.
├── main.py          # Complete training + inference pipeline
├── weights.npz     # Saved trained weights (auto-generated)
├── mnist.npz       # MNIST dataset (required)
└── README.md       # Documentation
```
<h2>⚙️ Configuration</h2>
<p>You can modify training and runtime behavior using:</p>

```python
epochs = 20
batch_size = 256
lr = 0.01
momentum = 0.9
roi_size = 280
min_pixel_sum = 500
```

<h2>🧪 Training</h2>
If weights.npz is not found, the model automatically trains on MNIST:
Dataset is loaded from mnist.npz -->
Training runs for defined epochs -->
Test accuracy is printed after each epoch -->
Weights are saved for future runs.

<h2>🔍 Preprocessing Pipeline</h2>
To match MNIST format, each frame undergoes:
<ol>
<li>Grayscale conversion</li>
<li></li>Gaussian blur (noise reduction)</ol>
<li>Otsu thresholding (auto binarization)</li>
<li>Inversion (white digit on black background)</li>
<li>Morphological dilation (thickens strokes)</li>
<li>Bounding box extraction</li>
<li>Resize longest side → 20 pixels</li>
<li>Centering using center-of-mass</li>
<li>Final 28×28 normalization</li>
</ol>
This step is critical for achieving high accuracy.
<h2>📦 Dependencies</h2>

```bash

pip install numpy opencv-python
```
<h2>▶️ How to Run</h2>
Download MNIST dataset (mnist.npz).
Run the script:

```bash
python main.py
```
<h2>💡 Future Improvements</h2>
Replace with CNN for higher accuracy.
Add multi-digit detection.
<h2>📈 Performance</h2>

```
Typical test accuracy: ~97–98%
Real-time inference: ~30 FPS (CPU)
```
<h2>🧩 Key Learning Outcomes</h2>
<ol>This project demonstrates:
<li>
Building neural networks from scratch</li>
<li>Understanding backpropagation deeply</li>
<li>Real-time computer vision integration</li>
<li>Bridging ML models with live systems</li></ol>

<h2>📜 License</h2>
Open-source
