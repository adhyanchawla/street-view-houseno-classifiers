#!/usr/bin/env python
# coding: utf-8

# In[83]:


import os
import numpy as np
import keras
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from skimage import color
from imageio import imread
from matplotlib.pyplot import imshow
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16.0, 4.0)


# In[5]:


np.random.seed(20)


# In[6]:


training_set=loadmat('train_32x32.mat')


# In[7]:


test_set=loadmat('test_32x32.mat')


# In[8]:


train_images=np.array(training_set['X'])
test_images=np.array(test_set['X'])
train_labels=np.array(training_set['y'])
test_labels=np.array(test_set['y'])


# In[9]:


print(train_images.shape)


# In[10]:


print(test_images.shape)


# In[11]:


train_images=np.moveaxis(train_images,-1,0)
test_images=np.moveaxis(test_images,-1,0)
print(train_images.shape)
print(test_images.shape)


# In[12]:


def plot_images(img, labels, nrows, ncols):
    figs,axes= plt.subplots(nrows,ncols)
    for i, ax in enumerate(axes.flat):
        if img[i].shape == (32,32,3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]),ax.set_yticks([])
        ax.set_title(labels[i])


# In[13]:


plot_images(train_images,train_labels,2,8)


# In[14]:


print(np.unique(train_labels))


# In[15]:


print(np.unique(test_labels))


# In[16]:


fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
fig.suptitle('Class Distribution',fontsize=14, fontweight='bold',y=1.05)
ax1.hist(train_labels, bins=10)
ax1.set_title('Training Set')
ax1.set_xlim(1,10)

ax2.hist(test_labels, color='g', bins=10)
ax2.set_title("Test set")
fig.tight_layout()


# In[17]:


train_labels[train_labels==10]=0
test_labels[test_labels==10]=0


# In[18]:


train_images = train_images.astype('float64')
test_images = test_images.astype('float64')


# In[19]:


train_labels = train_labels.astype('int64')
test_labels = test_labels.astype('int64')


# In[20]:


lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.fit_transform(test_labels)


# In[21]:


train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                  test_size=0.13, random_state=7)


# In[22]:


val_labels.shape


# In[23]:


train_images.shape


# In[24]:


datagen = ImageDataGenerator(rotation_range=8,
                             zoom_range=[0.95, 1.05],
                             height_shift_range=0.10,
                             shear_range=0.15)


# In[46]:


keras.backend.clear_session()

aux_model=keras.Sequential([
keras.layers.Conv2D(32,(3,3), padding='same',activation='relu',input_shape=(32,32,3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dropout(0.4),
    keras.layers.Dense(10,activation='softmax')])


learning_schedule = keras.callbacks.LearningRateScheduler(
              lambda epoch: 1e-4 * 10**(epoch / 10))
optimizer = keras.optimizers.Adam(lr=1e-4, amsgrad=True)
aux_model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                 metrics=['accuracy'])


# In[ ]:


history = aux_model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                              epochs=30, validation_data=(X_val, y_val),
                              callbacks=[lr_schedule])


# In[32]:


plt.semilogx(history.history['lr'], history.history['loss'])
plt.axis([1e-4, 1e-1, 0, 4])
plt.xlabel('Learning Rate')
plt.ylabel('Training Loss')
plt.show()


# In[53]:


keras.backend.clear_session()
def create_model():
    model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', 
                           activation='relu',
                           input_shape=(32, 32, 3)),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(32, (3, 3), padding='same', 
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(64, (3, 3), padding='same', 
                           activation='relu'),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(64, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(128, (3, 3), padding='same', 
                           activation='relu'),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(128, (3, 3), padding='same',
                        activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.4),    
    keras.layers.Dense(10,  activation='softmax')
    ])

    early_stopping = keras.callbacks.EarlyStopping(patience=8)
    optimizer = keras.optimizers.Adam(lr=1e-3, amsgrad=True)
    return model


# In[49]:


model.summary()


# In[56]:


model = create_model()
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                   checkpoint_path, 
                   save_best_only=True)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(datagen.flow(train_images, train_labels, batch_size=128),
                              epochs=70, validation_data=(val_images, val_labels),
                              callbacks=[early_stopping, model_checkpoint])


# In[61]:


loss, acc = model.evaluate(val_images,val_labels)


# In[63]:


test_loss, test_acc = model.evaluate(x=test_images, y=test_labels, verbose=0)

print('Test accuracy is: {:0.4f} \nTest loss is: {:0.4f}'.
      format(test_acc, test_loss))


# In[64]:


y_pred = model.predict(train_images)
y_pred = lb.inverse_transform(y_pred, lb.classes_)
y_train = lb.inverse_transform(train_labels, lb.classes_)
print(y_pred.shape)


# In[66]:


np.seterr(all='ignore')


# In[85]:


img_path = 'abc.jpg'
img = image.load_img(img_path, target_size=(32, 32))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x/255.0
print('Input image shape:', x.shape)
my_image = imread(img_path)
imshow(my_image)
print(model.predict(x))


# In[ ]:




