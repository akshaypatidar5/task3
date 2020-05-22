#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[2]:


dataset=mnist.load_data('mymnist.db')


# In[3]:


train,test=dataset


# In[4]:


X_train,Y_train =train


# In[5]:


X_test,Y_test=test


# In[6]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


# In[7]:


X_test.shape


# In[8]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[9]:


X_train/=255
X_test/=255


# In[10]:


from keras.utils.np_utils import to_categorical


# In[11]:


number_of_classes = 10

Y_train = to_categorical(Y_train, number_of_classes)
Y_test = to_categorical(Y_test, number_of_classes)


# In[12]:


from keras.layers import Convolution2D


# In[13]:


from keras.layers import MaxPooling2D


# In[14]:


from keras.layers import Flatten


# In[15]:


from keras.layers import Dense


# In[16]:


from keras.models import Sequential


# In[17]:


model = Sequential()


# In[18]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                   input_shape=(28,28,1)
                       ))


# In[19]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[20]:


model.add(Convolution2D(filters=32, 
                        kernel_size=(3,3), 
                        activation='relu',
                       ))


# In[21]:


model.add(MaxPooling2D(pool_size=(2, 2)))


# In[22]:


model.add(Flatten())


# In[23]:


model.add(Dense(units=128, activation='relu'))


# In[24]:


model.add(Dense(units=10, activation='softmax'))


# In[25]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[26]:


from keras_preprocessing.image import ImageDataGenerator


# In[27]:


train_datagen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)

test_datagen = ImageDataGenerator()

train_set= train_datagen.flow(X_train, Y_train, batch_size=64)
test_set = test_datagen.flow(X_test, Y_test, batch_size=64)


model.fit(train_set, steps_per_epoch=60000//64, epochs=5, 
                    validation_data=test_set, validation_steps=10000//64)


# In[28]:


scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

