import itertools
from keras import layers
from keras import models
from keras import optimizers
import os
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

base_dir = 'D:/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project/Data'
flower_dir = 'D:/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project/Data/train_image_flower'
not_flower_dir = 'D:/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project/Data/train_image_notflower'
data_path = 'D:/Cours/master/semestre 3/Machine_learning/Advanced-machine-learning-project/Data/train_images'

# Répertoires contenant les ensembles d'entraînement, de validation et de test

train_dir = os.path.join(base_dir, 'train_images_bin')

validation_dir = os.path.join(base_dir, 'validation_images_bin')

test_dir = os.path.join(base_dir, 'test_images_bin')

# Répertoire avec les images de fleurs pour l'entraînement

train_flower_dir = os.path.join(train_dir, 'flower')

# Répertoire avec les images qui ne sont pas des fleurs pour L'entraînement

train_notflower_dir = os.path.join(train_dir, 'notflower')

# Répertoire avec les images de fleurs pour la validation

validation_flower_dir = os.path.join(validation_dir, 'flower')

# Répertoire avec Les images qui ne sont pas des fleurs pour la validation

validation_notflower_dir = os.path.join(validation_dir, 'notflower')

# Répertoire avec les images de fleurs pour le test

test_flower_dir = os.path.join(test_dir, 'flower')

# Répertoire avec les images qui ne sont pas des fleurs pour le test

test_notflower_dir = os.path.join(test_dir, 'notflower')

# ----------------------------------------------------------------------------------------------------------------------

# création du réseau

# ----------------------------------------------------------------------------------------------------------------------

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1000, 700, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

# ----------------------------------------------------------------------------------------------------------------------

# utilisation d imagegenerator pour lire les images dans des répertoires

# ----------------------------------------------------------------------------------------------------------------------

# Modifie L'échelle des valeurs des images en appliquant

# le coefficient 1/255

target_size = (1000, 700)

train_datagen = ImageDataGenerator(rescale=1. / 255)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=target_size,
                                                    batch_size=5,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                        target_size=target_size,
                                                        batch_size=5,
                                                        class_mode='binary')

# ----------------------------------------------------------------------------------------------------------------------

                                               # training part

# ----------------------------------------------------------------------------------------------------------------------
#
# history = model.fit_generator(train_generator,
#                               steps_per_epoch=20,
#                               epochs=50,
#                               validation_data=validation_generator,
#                               validation_steps=50,
#                               verbose=2)
#
# model.save('flower_or_not_flower_1000_700_50.h5')
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
#
# plt.figure()
# plt.plot(epochs, acc, 'bo', label='Entrainement')
# plt.plot(epochs, val_acc, 'b', label='Validation')
# plt.title('Exactitude pendant l\'entrainement et la validation')
# plt.legend()
#
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Entrainement')
# plt.plot(epochs, val_loss, 'b', label='Validation')
# plt.title('Loss pendant l\'entrainement et la validation')
# plt.legend()


# ----------------------------------------------------------------------------------------------------------------------

                                               # Prediction part

# ----------------------------------------------------------------------------------------------------------------------


model = load_model(r'D:\Cours\master\semestre 3\Machine_learning\Advanced-machine-learning-project'
                   r'\flower_or_not_flower_1000_700_50.h5')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                   target_size=target_size,
                                                   batch_size=10,
                                                   class_mode='binary',
                                                   shuffle=True,
                                                   subset='training',
                                                   seed=42)

predictions = model.predict_generator(test_generator, steps=len(test_generator), verbose=2)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=2)
binary_predictions = (predictions > 0.5).astype(int)

# confusion matrix
cm = confusion_matrix(test_generator.classes, binary_predictions)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(test_generator.classes, predictions)

plt.figure()
plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.colorbar()

print('Test Accuracy: %.3f' % (test_acc * 100))
# print(binary_predictions)

plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')

# ROC Curve
roc_auc = roc_auc_score(test_generator.classes, predictions)
fpr, tpr, _ = roc_curve(test_generator.classes, predictions)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Accuracy vs. Threshold Plot
accuracies = []
thresholds = []

for i in range(101):
    threshold = i / 100
    binary_predictions = (predictions > threshold).astype(int)
    accuracy = accuracy_score(test_generator.classes, binary_predictions)
    accuracies.append(accuracy)
    thresholds.append(threshold)

plt.figure()
plt.plot(thresholds, accuracies)
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Threshold')

plt.show()
