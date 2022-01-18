import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy

#data preprocessing
data_generator = ImageDataGenerator(rescale=1.0/255., zoom_range=0.2,rotation_range=15,
                                    width_shift_range=0.05,height_shift_range=0.05)
batch_size = 60
#load training data with iterator 
training_iter = data_generator.flow_from_directory('Covid19-dataset/train',class_mode='categorical',
                                                   color_mode='grayscale',target_size=(256,256),batch_size=batch_size)
x_train,y_train  = training_iter.next()
print(x_train.shape,y_train.shape)

#load testing data
testing_iter = data_generator.flow_from_directory('Covid19-dataset/test',class_mode='categorical',
                                                  color_mode='grayscale',target_size=(256,256),batch_size=batch_size)
x_test,y_test = testing_iter.next()


#create model
model = Sequential(name='sequential')
model.add(Input(shape=(256,256,1)))
model.add(layers.Conv2D(5,5,strides=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(layers.Dropout(0.1))
model.add(layers.Conv2D(3,3,strides=1, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(3,activation='softmax'))

#compile model with adam optimizer and categorical crossentropy loss
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(), 
              metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

print(model.summary())

es = EarlyStopping(monitor='val_auc', mode='min', verbose=1, patience=20)

# fit the model with 5 ephochs and early stopping
print("\nTraining model...")
history=model.fit(training_iter,steps_per_epoch=len(x_train)/batch_size,epochs=50,validation_data=testing_iter,
          validation_steps=len(x_test)/32,callbacks=[es])

print(history.history)
# plotting categorical and validation accuracy over epochs
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
ax1.set_title('model accuracy')
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.legend(['train', 'validation'], loc='upper left')

# plotting auc and validation auc over epochs
# ax2 = fig.add_subplot(2, 1, 2)
# ax2.plot(history.history['auc_9'])
# ax2.plot(history.history['val_auc_9'])
# ax2.set_title('model auc')
# ax2.set_xlabel('epoch')
# ax2.set_ylabel('auc')
# ax2.legend(['train', 'validation'], loc='upper left')

plt.show()


test_steps_per_epoch = numpy.math.ceil(testing_iter.samples / testing_iter.batch_size)
predictions = model.predict(testing_iter, steps=test_steps_per_epoch)
test_steps_per_epoch = numpy.math.ceil(testing_iter.samples / testing_iter.batch_size)
predicted_classes = numpy.argmax(predictions, axis=1)
true_classes = testing_iter.classes
class_labels = list(testing_iter.class_indices.keys())
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)   

cm=confusion_matrix(true_classes,predicted_classes)
print(cm)