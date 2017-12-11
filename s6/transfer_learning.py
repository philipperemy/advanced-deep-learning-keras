from glob import glob

from keras import applications
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 256, 256
train_data_dir = 'transfer_learning/101_ObjectCategories'
validation_data_dir = 'transfer_learning/101_ObjectCategories_VAL'
batch_size = 16
nb_train_samples = len(glob(train_data_dir + '/**/*.jpg', recursive=True)) // batch_size
nb_validation_samples = len(glob(validation_data_dir + '/**/*.jpg', recursive=True)) // batch_size
epochs = 50

NB_CLASSES = 102

model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

model.summary()

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False

# Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NB_CLASSES, activation='softmax')(x)

# creating the final model
model_final = Model(inputs=model.input, outputs=predictions)

# compile the model
model_final.compile(loss='categorical_crossentropy',
                    optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                    metrics=['accuracy'])

# Initiate the train and test generators with data Augumentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode='nearest',
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode='nearest',
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=30)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    class_mode='categorical')

# Save the model according to the conditions
checkpoint = ModelCheckpoint('vgg16_1.h5',
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

early = EarlyStopping(monitor='val_acc',
                      min_delta=0,
                      patience=10,
                      verbose=1,
                      mode='auto')

# Train the model
model_final.fit_generator(
    generator=train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples,
    callbacks=[checkpoint, early])
