#!/usr/bin/env python
# coding: utf-8

# # Keras모델을 이용한 
# # Image Binary Classfication
# 
# #### dog(true,1): 210 / cat(false,0): 210

# In[1]:


# 훈련에 사용되는 고양이/개 이미지 경로
# user에 해당 이름의 폴더들 만들어주어야함!

train_dir = './tmp/cats_and_dogs_filtered/train'
validation_dir = './tmp/cats_and_dogs_filtered/validation'

train_cats_dir = './tmp/cats_and_dogs_filtered/train/cats'
train_dogs_dir = './tmp/cats_and_dogs_filtered/train/dogs'
print(train_cats_dir)
print(train_dogs_dir)


## 테스트는 한 파일로만 할 것이기 때문에 폴더 path 주석처리

# # 테스트에 사용되는 고양이/개 이미지 경로
# validation_cats_dir = './tmp/cats_and_dogs_filtered/validation/cats'
# validation_dogs_dir = './tmp/cats_and_dogs_filtered/validation/dogs'
# print(validation_cats_dir)
# print(validation_dogs_dir)


# In[2]:


# 파일 잘 로드 되었는지 확인

import os

train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

print(train_cat_fnames[:5])
print(train_dog_fnames[:5])


# In[3]:


# 파일 개수 확인

print('Total training cat images :', len(os.listdir(train_cats_dir)))
print('Total training dog images :', len(os.listdir(train_dogs_dir)))


# In[4]:


## 이미지 확인이라서 주석처리함!

# %matplotlib inline

# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt

# nrows, ncols = 4, 4
# pic_index = 0

# fig = plt.gcf()
# fig.set_size_inches(ncols*3, nrows*3)

# pic_index+=8

# next_cat_pix = [os.path.join(train_cats_dir, fname)
#                 for fname in train_cat_fnames[ pic_index-8:pic_index]]

# next_dog_pix = [os.path.join(train_dogs_dir, fname)
#                 for fname in train_dog_fnames[ pic_index-8:pic_index]]

# for i, img_path in enumerate(next_cat_pix+next_dog_pix):
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off')

#   img = mpimg.imread(img_path)
#   plt.imshow(img)

# plt.show()


# In[5]:


# 모델 구성

import tensorflow as tf


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()


# In[6]:


# 모델 컴파일 --warning무시

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
            loss='binary_crossentropy',
            metrics = ['accuracy'])


# In[7]:


# 이미지 데이터 전처리

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                  batch_size=20,
                                                  class_mode='binary',
                                                  target_size=(150, 150))
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                       batch_size=20,
                                                       class_mode  = 'binary',
                                                       target_size = (150, 150))


# In[8]:


# 모델 훈련

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=1,
                    validation_steps=50,
                    verbose=2)


# In[9]:


## 그래프 --과대적합이라 epoch가 한 번 밖에 돌지 않아서 의미가 없음...


# import matplotlib.pyplot as plt

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(len(acc))

# plt.plot(epochs, acc, 'bo', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'go', label='Training Loss')
# plt.plot(epochs, val_loss, 'g', label='Validation Loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()


# In[12]:


# 테스트 이미지 분류

import numpy as np
from tensorflow.keras.preprocessing import image
from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.start_preview()

for i in range(5):
   sleep(5)
   tmp = camera.capture('/home/pi/image%s.jpg' % i)
   img=image.load_img('/home/pi/image%s.jpg' % i, target_size=(150, 150)) 

   x=image.img_to_array(img)
   x=np.expand_dims(x, axis=0)
   images = np.vstack([x])

   classes = model.predict(images, batch_size=10)

   print(classes[0])
camera.stop_preview()









