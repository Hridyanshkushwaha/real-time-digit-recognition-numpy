import numpy as np
import cv2
import os

# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════
WEIGHTS_FILE  = "weights.npz"
EPOCHS        = 20
BATCH_SIZE    = 256
LR            = 0.01
MOMENTUM      = 0.9
ROI_SIZE      = 280          # green box size in camera pixels
MIN_PIXEL_SUM = 500          # ignore near-empty ROI frames

# ══════════════════════════════════════════════
#  LOAD MNIST
# ══════════════════════════════════════════════
print("Loading MNIST...")
data    = np.load("mnist.npz")
X_train = data["x_train"].reshape(-1, 784) / 255.0
y_train = data["y_train"]
X_test  = data["x_test"].reshape(-1, 784) / 255.0
y_test  = data["y_test"]
print(f"  {X_train.shape[0]} train / {X_test.shape[0]} test samples\n")

# ══════════════════════════════════════════════
#  NETWORK PARAMETERS  (784 → 256 → 128 → 10)
#  He initialisation for ReLU
# ══════════════════════════════════════════════
np.random.seed(42)
W1 = np.random.randn(784, 256) * np.sqrt(2.0 / 784)
b1 = np.zeros((1, 256))
W2 = np.random.randn(256, 128) * np.sqrt(2.0 / 256)
b2 = np.zeros((1, 128))
W3 = np.random.randn(128, 10)  * np.sqrt(2.0 / 128)
b3 = np.zeros((1, 10))

# Momentum velocity buffers
vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
vW3 = np.zeros_like(W3); vb3 = np.zeros_like(b3)

# ══════════════════════════════════════════════
#  ACTIVATIONS
# ══════════════════════════════════════════════
def relu(x):       return np.maximum(0, x)
def relu_d(x):     return (x > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# ══════════════════════════════════════════════
#  FORWARD / LOSS / BACKWARD
# ══════════════════════════════════════════════
def forward(X):
    z1 = X  @ W1 + b1;  a1 = relu(z1)
    z2 = a1 @ W2 + b2;  a2 = relu(z2)
    z3 = a2 @ W3 + b3;  a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def cross_entropy(p, y):
    return -np.log(p[range(len(y)), y] + 1e-8).mean()

def backward(X, y, z1, a1, z2, a2, a3):
    n = len(y)
    dz3 = a3.copy(); dz3[range(n), y] -= 1; dz3 /= n
    dW3 = a2.T @ dz3;  db3 = dz3.sum(0, keepdims=True)
    dz2 = (dz3 @ W3.T) * relu_d(z2)
    dW2 = a1.T @ dz2;  db2 = dz2.sum(0, keepdims=True)
    dz1 = (dz2 @ W2.T) * relu_d(z1)
    dW1 = X.T  @ dz1;  db1 = dz1.sum(0, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def sgd_momentum(grads):
    global W1,b1,W2,b2,W3,b3
    global vW1,vb1,vW2,vb2,vW3,vb3
    pairs = zip([W1,b1,W2,b2,W3,b3],
                grads,
                [vW1,vb1,vW2,vb2,vW3,vb3])
    for W, dW, v in pairs:
        v[:] = MOMENTUM * v - LR * dW
        W += v

def accuracy(X, y):
    *_, prob = forward(X)
    return (prob.argmax(1) == y).mean() * 100

# ══════════════════════════════════════════════
#  TRAIN OR LOAD WEIGHTS
# ══════════════════════════════════════════════
if os.path.exists(WEIGHTS_FILE):
    saved = np.load(WEIGHTS_FILE)
    W1,b1 = saved["W1"],saved["b1"]
    W2,b2 = saved["W2"],saved["b2"]
    W3,b3 = saved["W3"],saved["b3"]
    print(f"Loaded weights  →  test accuracy: {accuracy(X_test, y_test):.1f}%\n")
else:
    print("Training…")
    n = X_train.shape[0]
    idx = np.arange(n)
    for epoch in range(1, EPOCHS + 1):
        np.random.shuffle(idx)
        Xs, ys = X_train[idx], y_train[idx]
        total_loss, steps = 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            Xb, yb = Xs[i:i+BATCH_SIZE], ys[i:i+BATCH_SIZE]
            z1,a1,z2,a2,z3,a3 = forward(Xb)
            total_loss += cross_entropy(a3, yb); steps += 1
            sgd_momentum(backward(Xb, yb, z1, a1, z2, a2, a3))
        acc_te = accuracy(X_test, y_test)
        print(f"  Epoch {epoch:>2}/{EPOCHS}  "
              f"loss {total_loss/steps:.4f}  test {acc_te:.1f}%")
    np.savez(WEIGHTS_FILE, W1=W1,b1=b1, W2=W2,b2=b2, W3=W3,b3=b3)
    print("Weights saved.\n")

# ══════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════
def predict(vec):
    *_, prob = forward(vec)
    return int(prob.argmax()), float(prob.max())

# ══════════════════════════════════════════════
#  PREPROCESS ROI  →  MNIST-style 28×28
#
#  MNIST format:
#    • white strokes on BLACK background
#    • longest side scaled to 20 px
#    • centred by centre-of-mass (not bounding box centre)
# ══════════════════════════════════════════════
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def preprocess_roi(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu automatically finds the best threshold — no hardcoding needed
    _, th = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to fatten thin pen strokes to match MNIST stroke width
    th = cv2.dilate(th, KERNEL, iterations=1)

    if th.sum() < MIN_PIXEL_SUM * 255:
        return None

    # Bounding box around actual digit pixels
    coords = cv2.findNonZero(th)
    x, y, w, h = cv2.boundingRect(coords)
    digit = th[y:y+h, x:x+w].astype(np.float32)

    # Scale longest side → 20 px
    scale = 20.0 / max(w, h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    digit = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)

    # Place in 28×28 centred by bounding-box centre first
    canvas = np.zeros((28, 28), dtype=np.float32)
    cy = (28 - nh) // 2
    cx = (28 - nw) // 2
    canvas[cy:cy+nh, cx:cx+nw] = digit

    # Then nudge by centre-of-mass (matches MNIST preprocessing exactly)
    ys_idx, xs_idx = np.where(canvas > 0)
    if len(xs_idx) == 0:
        return None
    sx = int(14 - xs_idx.mean())
    sy = int(14 - ys_idx.mean())
    M  = np.float32([[1, 0, sx], [0, 1, sy]])
    canvas = cv2.warpAffine(canvas, M, (28, 28))

    return (canvas / 255.0).reshape(1, 784)

# ══════════════════════════════════════════════
#  DRAW HUD
# ══════════════════════════════════════════════
FONT  = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (50, 220, 80)
WHITE = (220, 220, 220)
GRAY  = (120, 120, 120)

def draw_hud(frame, rx, ry, label, conf, preview):
    fh, fw = frame.shape[:2]

    # Green ROI box
    cv2.rectangle(frame, (rx, ry), (rx+ROI_SIZE, ry+ROI_SIZE), GREEN, 2)
    cv2.putText(frame, "Write digit here",
                (rx, ry - 10), FONT, 0.55, GREEN, 1, cv2.LINE_AA)

    # Prediction text
    if label is not None:
        cv2.putText(frame, f"Digit: {label}   ({conf*100:.1f}%)",
                    (rx, ry + ROI_SIZE + 38),
                    FONT, 1.1, GREEN, 2, cv2.LINE_AA)
        # Confidence bar
        bw = int(conf * ROI_SIZE)
        cv2.rectangle(frame,
                      (rx, ry+ROI_SIZE+52), (rx+ROI_SIZE, ry+ROI_SIZE+64),
                      GRAY, -1)
        cv2.rectangle(frame,
                      (rx, ry+ROI_SIZE+52), (rx+bw, ry+ROI_SIZE+64),
                      GREEN, -1)
    else:
        cv2.putText(frame, "No digit detected",
                    (rx, ry + ROI_SIZE + 38),
                    FONT, 0.8, GRAY, 1, cv2.LINE_AA)

    # 28×28 debug preview — top-right corner
    if preview is not None:
        pv = (preview.reshape(28, 28) * 255).astype(np.uint8)
        pv = cv2.resize(pv, (84, 84), interpolation=cv2.INTER_NEAREST)
        pv_bgr = cv2.cvtColor(pv, cv2.COLOR_GRAY2BGR)
        px, py = fw - 94, 10
        frame[py:py+84, px:px+84] = pv_bgr
        cv2.rectangle(frame, (px-1, py-1), (px+85, py+85), WHITE, 1)
        cv2.putText(frame, "network sees",
                    (px, py+95), FONT, 0.38, WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, "Q = quit",
                (10, fh - 10), FONT, 0.45, GRAY, 1, cv2.LINE_AA)

# ══════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Check device index / permissions.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera open. Write a digit inside the green box.")
print("Press Q to quit.\n")

label_disp = None
conf_disp  = 0.0
preview_buf = None
SMOOTH = 0.6   # exponential smoothing for jitter-free confidence bar

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed — exiting.")
        break

    fh, fw = frame.shape[:2]
    rx = (fw - ROI_SIZE) // 2
    ry = (fh - ROI_SIZE) // 2

    roi = frame[ry:ry+ROI_SIZE, rx:rx+ROI_SIZE]
    vec = preprocess_roi(roi)

    if vec is not None:
        pred, conf = predict(vec)
        if conf > 0.50:
            label_disp  = pred
            conf_disp   = SMOOTH * conf_disp + (1 - SMOOTH) * conf
            preview_buf = vec
        else:
            label_disp = None
    else:
        label_disp  = None
        preview_buf = None

    draw_hud(frame, rx, ry, label_disp, conf_disp, preview_buf)
    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
