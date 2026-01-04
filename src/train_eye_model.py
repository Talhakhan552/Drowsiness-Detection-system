import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
# data_dir = r"C:\Users\Talha\OneDrive\Desktop\ML_Project\train"
# model_path = r"C:\Users\Talha\OneDrive\Desktop\ML_Project\models\eye_cnn.h5"


basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(basedir)
model_path = os.path.join(project_root, 'models/eye_cnn.h5')
data_dir = os.path.join(project_root, 'train')


# Prepare data (auto-resize)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#training
train = datagen.flow_from_directory(
    data_dir,
    target_size=(96, 96),   # <-- handles mixed sizes automatically
    color_mode='grayscale',
    class_mode='binary',
    subset='training'
)

#valid
val = datagen.flow_from_directory(
    data_dir,
    target_size=(96, 96),
    color_mode='grayscale',
    class_mode='binary',
    subset='validation'
)

# Build model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(96,96,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train, validation_data=val, epochs=15)

# Save model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print("✅ Model saved at", model_path)

