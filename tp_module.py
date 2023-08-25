import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Étape 1 : Choix du Jeu de Données
(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()

# Étape 2 : Partitionnement des Données
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.15, random_state=42)

# Étape 2 (suite) : Conversion des étiquettes en encodage catégoriel
num_classes = 10
y_train_categorical = to_categorical(y_train, num_classes)
y_val_categorical = to_categorical(y_val, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)

# Paramètres
input_shape = (32, 32, 3)

# Étape 3.1 : Modèle Préentraîné (VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Ajout de la nouvelle couche de classification
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Congeler les poids des couches du modèle préentraîné
for layer in base_model.layers:
    layer.trainable = False

# Compilation du modèle
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Afficher la structure du modèle
model.summary()

# Étape 3.2 : Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

datagen.fit(x_train)

# Étape 4.1 : Transfert Learning - Entraînement et Évaluation
history = model.fit(x_train, y_train_categorical, batch_size=32, epochs=10, validation_data=(x_val, y_val_categorical))

# Évaluation sur l'ensemble de validation
val_loss, val_acc = model.evaluate(x_val, y_val_categorical)
print(f"Validation Accuracy: {val_acc:.4f}")

# Étape 4.2 : Data Augmentation - Entraînement et Évaluation
augmented_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

augmented_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

augmented_history = augmented_model.fit(datagen.flow(x_train, y_train_categorical, batch_size=32), epochs=10, validation_data=(x_val, y_val_categorical))

# Évaluation sur l'ensemble de validation
aug_val_loss, aug_val_acc = augmented_model.evaluate(x_val, y_val_categorical)
print(f"Augmented Validation Accuracy: {aug_val_acc:.4f}")

# Étape 5 : Comparaison des Performances
print("Performance du modèle Transfert Learning (VGG16) :")
print("Validation Accuracy:", val_acc)
print("\nPerformance du modèle avec Data Augmentation (CNN) :")
print("Augmented Validation Accuracy:", aug_val_acc)