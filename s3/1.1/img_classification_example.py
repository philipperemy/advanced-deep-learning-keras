# coding: utf-8

# In[1]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# In[2]:


# In[3]:


# https://gggdomi.github.io/keras-workshop/notebook.html


# In[4]:


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# In[5]:


plt.imshow(mpimg.imread('data/train/cats/cat.10013.jpg'))

# In[6]:


plt.imshow(mpimg.imread('data/train/dogs/dog.10013.jpg'))

# In[7]:


# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1. / 255)

# automatically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')

# In[8]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense

# In[9]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# In[10]:


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# In[ ]:


nb_epoch = 1
nb_train_samples = 2048
nb_validation_samples = 832

# In[ ]:

while True:
    x_train, y_train = train_generator.next()
    print('TRAIN', model.train_on_batch(x_train, y_train))

    x_test, y_test = validation_generator.next()
    print('TEST', model.test_on_batch(x_test, y_test))

# model.fit_generator(
#     train_generator,
#     samples_per_epoch=nb_train_samples,
#     nb_epoch=nb_epoch,
#     validation_data=validation_generator,
#     nb_val_samples=nb_validation_samples)

print('hello done.')

# In[ ]:


model.save_weights('models/1000-samples--1-epochs.h5')

# In[ ]:


model.load_weights('models/without-data-augmentation/1000-samples--32-epochs.h5')

# In[ ]:


model.evaluate_generator(validation_generator, nb_validation_samples)
