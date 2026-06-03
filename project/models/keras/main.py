import scipy.io
import numpy as np
from tensorflow.keras.models import load_model



data = scipy.io.loadmat('data/sat-6-full.mat')
X = data['test_x']
Y = data['test_y']

X = np.transpose(X, (3,0,1,2))

X = X.astype(np.float32) / 255.0

X = X[:1000]

model = load_model('models/keras/t1_float.keras')

print(model.summary())
y_pred = model.predict(X)

X_flat = X.reshape(X.shape[0], -1)

np.savetxt(
    'models/keras/input_features.dat',
    X_flat,
    fmt="%.6f"
)

np.savetxt(
    'models/keras/output_predictions.dat',
    y_pred,
    fmt="%.6f"
)


