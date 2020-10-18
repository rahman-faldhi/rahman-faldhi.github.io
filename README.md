# SKIN CANCER CLASSIFICATION DEEP LEARNING WITH CNN MODEL
Skin cancer is the most common human malignancy, is primarily diagnosed visually, beginning with an initial clinical screening and followed potentially by dermoscopic analysis, a biopsy and histopathological examination. Automated classification of skin lesions using images is a challenging task owing to the fine-grained variability in the appearance of skin lesions.

This the HAM10000 ("Human Against Machine with 10000 training images") dataset.It consists of 10015 dermatoscopicimages which are released as a training set for academic machine learning purposes and are publiclyavailable through the ISIC archive. This benchmark dataset can be used for machine learning and for comparisons with human experts.

It has 7 different classes of skin cancer which are listed below :
1. Melanocytic nevi
2. Melanoma
3. Benign keratosis-like lesions
4. Basal cell carcinoma
5. Actinic keratoses
6. Vascular lesions
7. Dermatofibroma

In this kernel I will try to detect 7 different classes of skin cancer using Convolution Neural Network with keras tensorflow in backend and then analyse the result to see how the model can be useful in practical scenario.
We will move step by step process to classify 7 classes of cancer.
# Step 1 : Install Kaggle Extension and Download Dataset
The first thing to do is to copy the API token in the Kaggle account which will later be used to download Kaggle.JSON. Then upload the Kaggle.JSON, then copy the json file to the directory. after that download the MNIST HAM10000 skin cancer dataset

```python
! pip install -q kaggle
```
```python
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets list
```
```python
Warning: Looks like you're using an outdated API Version, please consider updating (server 1.5.6 / client 1.5.4)
ref                                                               title                                                 size  lastUpdated          downloadCount  
----------------------------------------------------------------  --------------------------------------------------  ------  -------------------  -------------  
christianlillelund/donald-trumps-rallies                          Donald Trump's Rallies                               720KB  2020-09-26 10:25:08            706  
heeraldedhia/groceries-dataset                                    Groceries dataset                                    257KB  2020-09-17 04:36:08           3331  
andrewmvd/trip-advisor-hotel-reviews                              Trip Advisor Hotel Reviews                             5MB  2020-09-30 08:31:20           1903  
balraj98/stanford-background-dataset                              Stanford Background Dataset                           17MB  2020-09-26 12:57:59            199  
nehaprabhavalkar/indian-food-101                                  Indian Food 101                                        7KB  2020-09-30 06:23:43           2682  
jilkothari/finance-accounting-courses-udemy-13k-course            Finance & Accounting Courses - Udemy (13K+ course)  1000KB  2020-09-17 12:46:12            920  
balraj98/massachusetts-roads-dataset                              Massachusetts Roads Dataset                            6GB  2020-09-26 03:57:49            161  
arslanali4343/top-personality-dataset                             Top Personality Dataset                                9MB  2020-09-27 21:25:45           1036  
oldaandozerskaya/fiction-corpus-for-agebased-text-classification  RusAge: Corpus for Age-Based Text Classification     509MB  2020-09-28 09:30:12             63  
gpreda/chinese-mnist                                              Chinese MNIST                                         10MB  2020-08-05 12:36:00            338  
gpreda/local-elections-romania-2020                               Local Elections Romania 2020                          28MB  2020-09-27 20:46:11            166  
anth7310/mental-health-in-the-tech-industry                       Mental Health in the Tech Industry                     2MB  2020-09-27 11:17:23           1472  
roshansharma/sanfranciso-crime-dataset                            Sanfranciso Crime Dataset                              6MB  2019-05-29 12:45:44           4234  
bppuneethpai/tldr-summary-for-man-pages                           TLDR summary for man pages                             8MB  2020-09-25 09:50:10             34  
sterby/german-recipes-dataset                                     German Recipes Dataset                                 5MB  2019-03-06 16:25:22            906  
thomaskonstantin/top-270-rated-computer-science-programing-books  Top 270 Computer Science / Programing Books           45KB  2020-09-28 16:47:12            554  
arslanali4343/real-estate-dataset                                 Real Estate DataSet                                   12KB  2020-09-28 21:25:33           1165  
leangab/poe-short-stories-corpuscsv                               E.A. Poe's corpus of short stories                   725KB  2020-09-28 11:43:08            123  
anmolkumar/health-insurance-cross-sell-prediction                 Health Insurance Cross Sell Prediction 🏠 🏥             6MB  2020-09-11 18:39:31           4546  
ramjidoolla/ipl-data-set                                          IPL _Data_Set                                          1MB  2020-09-14 10:57:42           4492  
```
```python
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
```
```python
Downloading skin-cancer-mnist-ham10000.zip to /content
100% 5.20G/5.20G [01:48<00:00, 30.8MB/s]
100% 5.20G/5.20G [01:48<00:00, 51.6MB/s]
```
```python
! mkdir skin_cancer_mnist
! unzip skin-cancer-mnist-ham10000.zip -d skin_cancer_mnist
```
```python
inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029326.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029327.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029328.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029329.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029330.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029331.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029332.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029333.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029334.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029335.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029336.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029337.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029338.jpg  
  inflating: skin_cancer_mnist/ham10000_images_part_2/ISIC_0029339.jpg  
```
```python
!rm -r /content/skin_cancer_mnist/ham10000_images_part_1
!rm -r /content/skin_cancer_mnist/ham10000_images_part_2
```
# Step 2: Import Library and Load Images
Import the library used then create a plot history model function which will later be used to display the history graph. 6.	Load the image by combining part 1 and part 2 folders, then create a new dictionary which will act as a bridge between the csv and image files. then create a new column named 'path' which will bridge the column 'image id' with 'image path dict' by displaying the path of the image. then create column 'cell type' which will function by linking column 'dx' with the contents of the dictionary 'lession type dict'. then column 'cell type idx' to display categorycal from column 'cell type' in the form of code
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools

import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
file_dir = os.path.join('..', '/content/skin_cancer_mnist')

# Menggabungkan gambar dari kedua folder HAM10000_images_part1.zip dan HAM10000_images_part2.zip menjadi satu dictionary
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(file_dir, '*', '*.jpg'))}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
skin_df = pd.read_csv(os.path.join(file_dir, 'HAM10000_metadata.csv'))

# Membuat column baru
skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
```
# Step 3: Data Cleansing
On data cleaning, display features that still have a null value, in this case, age still has a null value, then the null value is replaced with the mean value of the whole 'age'. next is resizing the entire image. then display as many as 5 sample images per each skin cancer category
```python
skin_df.head(100)
```
```python

lesion_id	image_id	dx	dx_type	age	sex	localization	path	cell_type	cell_type_idx
0	HAM_0000118	ISIC_0027419	bkl	histo	80.0	male	scalp	/content/skin_cancer_mnist/HAM10000_images_par...	Benign keratosis-like lesions	2
1	HAM_0000118	ISIC_0025030	bkl	histo	80.0	male	scalp	/content/skin_cancer_mnist/HAM10000_images_par...	Benign keratosis-like lesions	2
2	HAM_0002730	ISIC_0026769	bkl	histo	80.0	male	scalp	/content/skin_cancer_mnist/HAM10000_images_par...	Benign keratosis-like lesions	2
3	HAM_0002730	ISIC_0025661	bkl	histo	80.0	male	scalp	/content/skin_cancer_mnist/HAM10000_images_par...	Benign keratosis-like lesions	2
4	HAM_0001466	ISIC_0031633	bkl	histo	75.0	male	ear	/content/skin_cancer_mnist/HAM10000_images_par...	Benign keratosis-like lesions	2
...	...	...	...	...	...	...	...	...	...	...
95	HAM_0000746	ISIC_0027023	bkl	histo	60.0	male	face	/content/skin_cancer_mnist/HAM10000_images_par...	Benign keratosis-like lesions	2
96	HAM_0001473	ISIC_0029022	bkl	histo	70.0	male	face	/content/skin_cancer_mnist/HAM10000_images_par...	Benign keratosis-like lesions	2
97	HAM_0003007	ISIC_0025388	bkl	histo	40.0	female	abdomen	/content/skin_cancer_mnist/HAM10000_images_par...	Benign keratosis-like lesions	2
98	HAM_0003007	ISIC_0028080	bkl	histo	40.0	female	abdomen	/content/skin_cancer_mnist/HAM10000_images_par...	Benign keratosis-like lesions	2
99	HAM_0002957	ISIC_0026153	bkl	histo	70.0	male	back	/content/skin_cancer_mnist/HAM10000_images_par...	Benign keratosis-like lesions	2
```
```python
skin_df.isnull().sum()
```
```python
lesion_id         0
image_id          0
dx                0
dx_type           0
age              57
sex               0
localization      0
path              0
cell_type         0
cell_type_idx     0
```
```python
skin_df['age'].fillna((skin_df['age'].mean()), inplace=True)
```
```python
skin_df.isnull().sum()
```
```python
lesion_id         0
image_id          0
dx                0
dx_type           0
age               0
sex               0
localization      0
path              0
cell_type         0
cell_type_idx     0
dtype: int64
```
```python
#resizing image
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, 
                                         skin_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
fig.savefig('category_samples.png', dpi=300)
```
![5_samples_categories](https://user-images.githubusercontent.com/72899789/96359455-354f5180-113d-11eb-962c-cd189efe267f.png)
```python
skin_df['image'].map(lambda x: x.shape).value_counts()
```
# Step 4: Features Engineering
In feature engineering, splitting the data on the train data and the test data, then normalizing the data, then spilling the data on the train data and validating the data. then reshape each data
```python
features=skin_df.drop(columns=['cell_type_idx'],axis=1)
target=skin_df['cell_type_idx']
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.20,random_state=1234)
#Normalisasi data

x_train = np.asarray(x_train_o['image'].tolist())
x_test = np.asarray(x_test_o['image'].tolist())

x_train_mean = np.mean(x_train) 
x_train_std = np.std(x_train)

x_test_mean = np.mean(x_test)
x_test_std = np.std(x_test)

x_train = (x_train - x_train_mean)/x_train_std
x_test = (x_test - x_test_mean)/x_test_std
y_train = to_categorical(y_train_o, num_classes = 7)
y_test = to_categorical(y_test_o, num_classes = 7)
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.20, random_state = 10)
x_train = x_train.reshape(x_train.shape[0], *(75, 100, 3))
x_test = x_test.reshape(x_test.shape[0], *(75, 100, 3))
x_validate = x_validate.reshape(x_validate.shape[0], *(75, 100, 3))
```
# Step 5: CNN
For CNN, use the hard sequential model, use filters 32 and 64 on the convolutional layer, and then use the pooling (MaxPool2D) layer, and use the dropout and dense. Then, the optimizer used is Adam
```python
input_shape = (75, 100, 3)
num_classes = 7

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape)) #1
model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',)) #2
model.add(MaxPool2D(pool_size = (2, 2))) #3
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same')) #4
model.add(Conv2D(64, (3, 3), activation='relu',padding = 'Same')) #5
model.add(MaxPool2D(pool_size=(2, 2))) #6
model.add(Dropout(0.40))

model.add(Flatten())
model.add(Dense(128, activation='relu')) #7
model.add(Dropout(0.5)) #8
model.add(Dense(num_classes, activation='softmax')) 
model.summary()
```
```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 75, 100, 32)       896       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 75, 100, 32)       9248      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 37, 50, 32)        0         
_________________________________________________________________
dropout (Dropout)            (None, 37, 50, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 37, 50, 64)        18496     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 37, 50, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 18, 25, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 18, 25, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 28800)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               3686528   
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 903       
=================================================================
Total params: 3,752,999
Trainable params: 3,752,999
Non-trainable params: 0
_________________________________________________________________
```
```python
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
```
# Step 6: Features Visualization
Then here is performed data visualization per each layer
```python
from keras.models import Model
layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x_train[10].reshape(1,75,100,3))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap=None)
            activation_index += 1
display_activation(activations, 5, 5, 0)
display_activation(activations, 5, 5, 1)
.....
.....
.....
display_activation(activations, 8, 8, 7)
```
here is the example from 8th layers
![8th_layer_visualization](https://user-images.githubusercontent.com/72899789/96359682-c45d6900-113f-11eb-8c68-8b9b0ec93d32.png)
# Step 7: Image Augmentation With ImageDataGenerator
For the image data generator, each parameter used in the image data generator will be stored in a variable called datagen which will be fitted with x_train.
```python
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

def aug_param(prm):
  img = load_img('/content/skin_cancer_mnist/HAM10000_images_part_1/ISIC_0024307.jpg')
  data = img_to_array(img)
  samples = expand_dims(data, 0)
  datagen2 = (prm)
  it = datagen2.flow(samples, batch_size=1)
  for i in range(9):
    pyplot.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    pyplot.imshow(image)
  pyplot.show()
param=[ImageDataGenerator(width_shift_range=[-200,200]),ImageDataGenerator(height_shift_range=0.5),ImageDataGenerator(horizontal_flip=True),ImageDataGenerator(rotation_range=90),ImageDataGenerator(brightness_range=[0.2,1.0]),ImageDataGenerator(zoom_range=[0.5,1.0])]
name=['Random Horizontal Shift','Random Vertical Shift','Random horizontal Flip','Random Rotation Augmentation','Random Brightness Augmentation','Random Zoom Augmentation']

for i in range(len(name)):
  print('untuk: ',name[i])
  aug_param(param[i])
```
![Random horizontal dan vertical](https://user-images.githubusercontent.com/72899789/96359836-6e89c080-1141-11eb-9b63-7ec4d6c523b2.png)
![Random horizontal flip dan rotation](https://user-images.githubusercontent.com/72899789/96359838-72b5de00-1141-11eb-8883-a2b62ba40bbc.png)
![Random brightness dan zoom](https://user-images.githubusercontent.com/72899789/96359842-78abbf00-1141-11eb-8acf-73db47df51bf.png)
# Step 8: Create Checkpoint and Tensorboard
Next is to create a checkpoint, then define the file path with the model name we want, for example skin.h5, here the largest val_accuracy value will be stored in skin.h5. Here is the fitting of the model to x_train and y_train, here I use epoch 100 and batch size 256. After the process is complete then we do the visualization process using tensorboard
```python
from tensorflow.keras.callbacks import ModelCheckpoint
filepath='skin.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

from tensorflow.keras.callbacks import TensorBoard
import os
import datetime

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Tempat dimana log tensorboard akan di
callbacks_list.append(TensorBoard(logdir, histogram_freq=1))

epochs = 100 
batch_size = 256
history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_validate,y_validate),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=callbacks_list)
```
```python
WARNING:tensorflow:From <ipython-input-46-9263c68be960>:6: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
Please use Model.fit, which supports generators.
Epoch 1/100
 1/25 [>.............................] - ETA: 0s - loss: 2.0134 - accuracy: 0.1211WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
Instructions for updating:
use `tf.profiler.experimental.stop` instead.
25/25 [==============================] - ETA: 0s - loss: 1.4123 - accuracy: 0.6216
Epoch 00001: val_accuracy improved from -inf to 0.68185, saving model to skin.h5
25/25 [==============================] - 12s 469ms/step - loss: 1.4123 - accuracy: 0.6216 - val_loss: 1.0417 - val_accuracy: 0.6818
Epoch 2/100
24/25 [===========================>..] - ETA: 0s - loss: 1.0205 - accuracy: 0.6642
Epoch 00002: val_accuracy improved from 0.68185 to 0.68559, saving model to skin.h5
25/25 [==============================] - 11s 440ms/step - loss: 1.0220 - accuracy: 0.6639 - val_loss: 0.9666 - val_accuracy: 0.6856
Epoch 3/100
25/25 [==============================] - ETA: 0s - loss: 0.9871 - accuracy: 0.6602
Epoch 00003: val_accuracy did not improve from 0.68559
25/25 [==============================] - 11s 442ms/step - loss: 0.9871 - accuracy: 0.6602 - val_loss: 0.9030 - val_accuracy: 0.6843
Epoch 4/100
25/25 [==============================] - ETA: 0s - loss: 0.9335 - accuracy: 0.6634
Epoch 00004: val_accuracy improved from 0.68559 to 0.68746, saving model to skin.h5
25/25 [==============================] - 11s 440ms/step - loss: 0.9335 - accuracy: 0.6634 - val_loss: 0.8694 - val_accuracy: 0.6875
Epoch 5/100
25/25 [==============================] - ETA: 0s - loss: 0.9129 - accuracy: 0.6685
Epoch 00005: val_accuracy improved from 0.68746 to 0.70056, saving model to skin.h5
25/25 [==============================] - 11s 443ms/step - loss: 0.9129 - accuracy: 0.6685 - val_loss: 0.8492 - val_accuracy: 0.7006
Epoch 6/100
25/25 [==============================] - ETA: 0s - loss: 0.9025 - accuracy: 0.6759
Epoch 00006: val_accuracy did not improve from 0.70056
25/25 [==============================] - 11s 438ms/step - loss: 0.9025 - accuracy: 0.6759 - val_loss: 0.8846 - val_accuracy: 0.7006
Epoch 7/100
25/25 [==============================] - ETA: 0s - loss: 0.8869 - accuracy: 0.6733
Epoch 00007: val_accuracy did not improve from 0.70056
25/25 [==============================] - 11s 439ms/step - loss: 0.8869 - accuracy: 0.6733 - val_loss: 0.8470 - val_accuracy: 0.6993
Epoch 8/100
25/25 [==============================] - ETA: 0s - loss: 0.8627 - accuracy: 0.6805
Epoch 00008: val_accuracy improved from 0.70056 to 0.70306, saving model to skin.h5
25/25 [==============================] - 11s 441ms/step - loss: 0.8627 - accuracy: 0.6805 - val_loss: 0.8300 - val_accuracy: 0.7031
Epoch 9/100
25/25 [==============================] - ETA: 0s - loss: 0.8363 - accuracy: 0.6896
Epoch 00009: val_accuracy improved from 0.70306 to 0.71553, saving model to skin.h5
25/25 [==============================] - 11s 444ms/step - loss: 0.8363 - accuracy: 0.6896 - val_loss: 0.8063 - val_accuracy: 0.7155
Epoch 10/100
25/25 [==============================] - ETA: 0s - loss: 0.8200 - accuracy: 0.6925
Epoch 00010: val_accuracy improved from 0.71553 to 0.71616, saving model to skin.h5
25/25 [==============================] - 11s 447ms/step - loss: 0.8200 - accuracy: 0.6925 - val_loss: 0.8013 - val_accuracy: 0.7162
Epoch 11/100
25/25 [==============================] - ETA: 0s - loss: 0.8283 - accuracy: 0.6936
Epoch 00011: val_accuracy did not improve from 0.71616
25/25 [==============================] - 11s 442ms/step - loss: 0.8283 - accuracy: 0.6936 - val_loss: 0.7936 - val_accuracy: 0.7162
Epoch 12/100
25/25 [==============================] - ETA: 0s - loss: 0.8038 - accuracy: 0.7044
Epoch 00012: val_accuracy improved from 0.71616 to 0.72551, saving model to skin.h5
25/25 [==============================] - 11s 444ms/step - loss: 0.8038 - accuracy: 0.7044 - val_loss: 0.7877 - val_accuracy: 0.7255
Epoch 13/100
25/25 [==============================] - ETA: 0s - loss: 0.8153 - accuracy: 0.6980
Epoch 00013: val_accuracy did not improve from 0.72551
25/25 [==============================] - 11s 443ms/step - loss: 0.8153 - accuracy: 0.6980 - val_loss: 0.8106 - val_accuracy: 0.7137
Epoch 14/100
25/25 [==============================] - ETA: 0s - loss: 0.8034 - accuracy: 0.7091
Epoch 00014: val_accuracy improved from 0.72551 to 0.73924, saving model to skin.h5
25/25 [==============================] - 11s 445ms/step - loss: 0.8034 - accuracy: 0.7091 - val_loss: 0.7751 - val_accuracy: 0.7392
Epoch 15/100
25/25 [==============================] - ETA: 0s - loss: 0.7922 - accuracy: 0.7081
Epoch 00015: val_accuracy did not improve from 0.73924
25/25 [==============================] - 11s 442ms/step - loss: 0.7922 - accuracy: 0.7081 - val_loss: 0.7537 - val_accuracy: 0.7336
Epoch 16/100
25/25 [==============================] - ETA: 0s - loss: 0.7782 - accuracy: 0.7115
Epoch 00016: val_accuracy improved from 0.73924 to 0.74236, saving model to skin.h5
25/25 [==============================] - 11s 447ms/step - loss: 0.7782 - accuracy: 0.7115 - val_loss: 0.7476 - val_accuracy: 0.7424
Epoch 17/100
25/25 [==============================] - ETA: 0s - loss: 0.7652 - accuracy: 0.7156
Epoch 00017: val_accuracy did not improve from 0.74236
25/25 [==============================] - 11s 443ms/step - loss: 0.7652 - accuracy: 0.7156 - val_loss: 0.7547 - val_accuracy: 0.7342
Epoch 18/100
25/25 [==============================] - ETA: 0s - loss: 0.7523 - accuracy: 0.7200
Epoch 00018: val_accuracy did not improve from 0.74236
25/25 [==============================] - 11s 441ms/step - loss: 0.7523 - accuracy: 0.7200 - val_loss: 0.7621 - val_accuracy: 0.7236
Epoch 19/100
25/25 [==============================] - ETA: 0s - loss: 0.7383 - accuracy: 0.7260
Epoch 00019: val_accuracy improved from 0.74236 to 0.74735, saving model to skin.h5
25/25 [==============================] - 11s 445ms/step - loss: 0.7383 - accuracy: 0.7260 - val_loss: 0.7355 - val_accuracy: 0.7473
Epoch 20/100
24/25 [===========================>..] - ETA: 0s - loss: 0.7360 - accuracy: 0.7285
Epoch 00020: val_accuracy improved from 0.74735 to 0.74922, saving model to skin.h5
25/25 [==============================] - 11s 447ms/step - loss: 0.7355 - accuracy: 0.7286 - val_loss: 0.7181 - val_accuracy: 0.7492
Epoch 21/100
25/25 [==============================] - ETA: 0s - loss: 0.7486 - accuracy: 0.7258
Epoch 00021: val_accuracy did not improve from 0.74922
25/25 [==============================] - 11s 444ms/step - loss: 0.7486 - accuracy: 0.7258 - val_loss: 0.7141 - val_accuracy: 0.7486
Epoch 22/100
25/25 [==============================] - ETA: 0s - loss: 0.7781 - accuracy: 0.7154
Epoch 00022: val_accuracy did not improve from 0.74922
25/25 [==============================] - 11s 442ms/step - loss: 0.7781 - accuracy: 0.7154 - val_loss: 0.7257 - val_accuracy: 0.7455
Epoch 23/100
25/25 [==============================] - ETA: 0s - loss: 0.7344 - accuracy: 0.7265
Epoch 00023: val_accuracy did not improve from 0.74922
25/25 [==============================] - 11s 443ms/step - loss: 0.7344 - accuracy: 0.7265 - val_loss: 0.7294 - val_accuracy: 0.7255
Epoch 24/100
25/25 [==============================] - ETA: 0s - loss: 0.7163 - accuracy: 0.7335
Epoch 00024: val_accuracy did not improve from 0.74922
25/25 [==============================] - 11s 446ms/step - loss: 0.7163 - accuracy: 0.7335 - val_loss: 0.7140 - val_accuracy: 0.7355
Epoch 25/100
25/25 [==============================] - ETA: 0s - loss: 0.7081 - accuracy: 0.7336
Epoch 00025: val_accuracy did not improve from 0.74922
25/25 [==============================] - 11s 442ms/step - loss: 0.7081 - accuracy: 0.7336 - val_loss: 0.7514 - val_accuracy: 0.7380
Epoch 26/100
25/25 [==============================] - ETA: 0s - loss: 0.7436 - accuracy: 0.7283
Epoch 00026: val_accuracy did not improve from 0.74922
25/25 [==============================] - 11s 444ms/step - loss: 0.7436 - accuracy: 0.7283 - val_loss: 0.7045 - val_accuracy: 0.7473
Epoch 27/100
25/25 [==============================] - ETA: 0s - loss: 0.7101 - accuracy: 0.7367
Epoch 00027: val_accuracy improved from 0.74922 to 0.75795, saving model to skin.h5
25/25 [==============================] - 12s 466ms/step - loss: 0.7101 - accuracy: 0.7367 - val_loss: 0.6878 - val_accuracy: 0.7580
Epoch 28/100
25/25 [==============================] - ETA: 0s - loss: 0.6993 - accuracy: 0.7379
Epoch 00028: val_accuracy did not improve from 0.75795
25/25 [==============================] - 11s 443ms/step - loss: 0.6993 - accuracy: 0.7379 - val_loss: 0.6883 - val_accuracy: 0.7573
Epoch 29/100
24/25 [===========================>..] - ETA: 0s - loss: 0.6984 - accuracy: 0.7401
Epoch 00029: val_accuracy improved from 0.75795 to 0.76107, saving model to skin.h5
25/25 [==============================] - 11s 444ms/step - loss: 0.6980 - accuracy: 0.7403 - val_loss: 0.6859 - val_accuracy: 0.7611
Epoch 30/100
25/25 [==============================] - ETA: 0s - loss: 0.6894 - accuracy: 0.7474
Epoch 00030: val_accuracy did not improve from 0.76107
25/25 [==============================] - 11s 444ms/step - loss: 0.6894 - accuracy: 0.7474 - val_loss: 0.6860 - val_accuracy: 0.7511
Epoch 31/100
25/25 [==============================] - ETA: 0s - loss: 0.6884 - accuracy: 0.7413
Epoch 00031: val_accuracy improved from 0.76107 to 0.76419, saving model to skin.h5
25/25 [==============================] - 11s 449ms/step - loss: 0.6884 - accuracy: 0.7413 - val_loss: 0.6664 - val_accuracy: 0.7642
Epoch 32/100
25/25 [==============================] - ETA: 0s - loss: 0.6662 - accuracy: 0.7561
Epoch 00032: val_accuracy did not improve from 0.76419
25/25 [==============================] - 11s 443ms/step - loss: 0.6662 - accuracy: 0.7561 - val_loss: 0.6695 - val_accuracy: 0.7580
Epoch 33/100
25/25 [==============================] - ETA: 0s - loss: 0.6635 - accuracy: 0.7574
Epoch 00033: val_accuracy did not improve from 0.76419
25/25 [==============================] - 11s 441ms/step - loss: 0.6635 - accuracy: 0.7574 - val_loss: 0.6675 - val_accuracy: 0.7548
Epoch 34/100
25/25 [==============================] - ETA: 0s - loss: 0.6507 - accuracy: 0.7515
Epoch 00034: val_accuracy did not improve from 0.76419
25/25 [==============================] - 11s 441ms/step - loss: 0.6507 - accuracy: 0.7515 - val_loss: 0.6721 - val_accuracy: 0.7555
Epoch 35/100
25/25 [==============================] - ETA: 0s - loss: 0.6923 - accuracy: 0.7385
Epoch 00035: val_accuracy did not improve from 0.76419
25/25 [==============================] - 11s 441ms/step - loss: 0.6923 - accuracy: 0.7385 - val_loss: 0.6833 - val_accuracy: 0.7536
Epoch 36/100
25/25 [==============================] - ETA: 0s - loss: 0.6588 - accuracy: 0.7549
Epoch 00036: val_accuracy improved from 0.76419 to 0.76606, saving model to skin.h5
25/25 [==============================] - 11s 446ms/step - loss: 0.6588 - accuracy: 0.7549 - val_loss: 0.6616 - val_accuracy: 0.7661
Epoch 37/100
25/25 [==============================] - ETA: 0s - loss: 0.6555 - accuracy: 0.7525
Epoch 00037: val_accuracy did not improve from 0.76606
25/25 [==============================] - 11s 443ms/step - loss: 0.6555 - accuracy: 0.7525 - val_loss: 0.7045 - val_accuracy: 0.7461
Epoch 38/100
25/25 [==============================] - ETA: 0s - loss: 0.6549 - accuracy: 0.7554
Epoch 00038: val_accuracy improved from 0.76606 to 0.76669, saving model to skin.h5
25/25 [==============================] - 11s 450ms/step - loss: 0.6549 - accuracy: 0.7554 - val_loss: 0.6561 - val_accuracy: 0.7667
Epoch 39/100
25/25 [==============================] - ETA: 0s - loss: 0.6472 - accuracy: 0.7569
Epoch 00039: val_accuracy did not improve from 0.76669
25/25 [==============================] - 11s 443ms/step - loss: 0.6472 - accuracy: 0.7569 - val_loss: 0.6691 - val_accuracy: 0.7642
Epoch 40/100
25/25 [==============================] - ETA: 0s - loss: 0.6462 - accuracy: 0.7525
Epoch 00040: val_accuracy did not improve from 0.76669
25/25 [==============================] - 11s 443ms/step - loss: 0.6462 - accuracy: 0.7525 - val_loss: 0.6591 - val_accuracy: 0.7598
Epoch 41/100
25/25 [==============================] - ETA: 0s - loss: 0.6380 - accuracy: 0.7582
Epoch 00041: val_accuracy did not improve from 0.76669
25/25 [==============================] - 11s 442ms/step - loss: 0.6380 - accuracy: 0.7582 - val_loss: 0.6639 - val_accuracy: 0.7648
Epoch 42/100
25/25 [==============================] - ETA: 0s - loss: 0.6457 - accuracy: 0.7621
Epoch 00042: val_accuracy did not improve from 0.76669
25/25 [==============================] - 11s 445ms/step - loss: 0.6457 - accuracy: 0.7621 - val_loss: 0.6742 - val_accuracy: 0.7580
Epoch 43/100
25/25 [==============================] - ETA: 0s - loss: 0.6385 - accuracy: 0.7604
Epoch 00043: val_accuracy did not improve from 0.76669
25/25 [==============================] - 11s 442ms/step - loss: 0.6385 - accuracy: 0.7604 - val_loss: 0.6696 - val_accuracy: 0.7623
Epoch 44/100
25/25 [==============================] - ETA: 0s - loss: 0.6773 - accuracy: 0.7470
Epoch 00044: val_accuracy did not improve from 0.76669
25/25 [==============================] - 11s 443ms/step - loss: 0.6773 - accuracy: 0.7470 - val_loss: 0.6758 - val_accuracy: 0.7629
Epoch 45/100
25/25 [==============================] - ETA: 0s - loss: 0.6309 - accuracy: 0.7645
Epoch 00045: val_accuracy did not improve from 0.76669
25/25 [==============================] - 11s 446ms/step - loss: 0.6309 - accuracy: 0.7645 - val_loss: 0.6714 - val_accuracy: 0.7604
Epoch 46/100
25/25 [==============================] - ETA: 0s - loss: 0.6581 - accuracy: 0.7575
Epoch 00046: val_accuracy improved from 0.76669 to 0.77043, saving model to skin.h5
25/25 [==============================] - 11s 447ms/step - loss: 0.6581 - accuracy: 0.7575 - val_loss: 0.6553 - val_accuracy: 0.7704
Epoch 47/100
25/25 [==============================] - ETA: 0s - loss: 0.6239 - accuracy: 0.7614
Epoch 00047: val_accuracy did not improve from 0.77043
25/25 [==============================] - 11s 445ms/step - loss: 0.6239 - accuracy: 0.7614 - val_loss: 0.6591 - val_accuracy: 0.7692
Epoch 48/100
25/25 [==============================] - ETA: 0s - loss: 0.6148 - accuracy: 0.7687
Epoch 00048: val_accuracy improved from 0.77043 to 0.77480, saving model to skin.h5
25/25 [==============================] - 12s 464ms/step - loss: 0.6148 - accuracy: 0.7687 - val_loss: 0.6585 - val_accuracy: 0.7748
Epoch 49/100
25/25 [==============================] - ETA: 0s - loss: 0.6080 - accuracy: 0.7702
Epoch 00049: val_accuracy improved from 0.77480 to 0.77792, saving model to skin.h5
25/25 [==============================] - 11s 448ms/step - loss: 0.6080 - accuracy: 0.7702 - val_loss: 0.6633 - val_accuracy: 0.7779
Epoch 50/100
25/25 [==============================] - ETA: 0s - loss: 0.6310 - accuracy: 0.7570
Epoch 00050: val_accuracy did not improve from 0.77792
25/25 [==============================] - 11s 457ms/step - loss: 0.6310 - accuracy: 0.7570 - val_loss: 0.6756 - val_accuracy: 0.7648
Epoch 51/100
24/25 [===========================>..] - ETA: 0s - loss: 0.6093 - accuracy: 0.7710
Epoch 00051: val_accuracy did not improve from 0.77792
25/25 [==============================] - 11s 442ms/step - loss: 0.6089 - accuracy: 0.7712 - val_loss: 0.6697 - val_accuracy: 0.7642
Epoch 52/100
25/25 [==============================] - ETA: 0s - loss: 0.6105 - accuracy: 0.7721
Epoch 00052: val_accuracy did not improve from 0.77792
25/25 [==============================] - 11s 447ms/step - loss: 0.6105 - accuracy: 0.7721 - val_loss: 0.6724 - val_accuracy: 0.7742
Epoch 53/100
25/25 [==============================] - ETA: 0s - loss: 0.6546 - accuracy: 0.7526
Epoch 00053: val_accuracy did not improve from 0.77792
25/25 [==============================] - 11s 441ms/step - loss: 0.6546 - accuracy: 0.7526 - val_loss: 0.6979 - val_accuracy: 0.7580
Epoch 54/100
25/25 [==============================] - ETA: 0s - loss: 0.6597 - accuracy: 0.7460
Epoch 00054: val_accuracy did not improve from 0.77792
25/25 [==============================] - 11s 451ms/step - loss: 0.6597 - accuracy: 0.7460 - val_loss: 0.6622 - val_accuracy: 0.7711
Epoch 55/100
25/25 [==============================] - ETA: 0s - loss: 0.6512 - accuracy: 0.7554
Epoch 00055: val_accuracy did not improve from 0.77792
25/25 [==============================] - 11s 442ms/step - loss: 0.6512 - accuracy: 0.7554 - val_loss: 0.6713 - val_accuracy: 0.7698
Epoch 56/100
25/25 [==============================] - ETA: 0s - loss: 0.6227 - accuracy: 0.7637
Epoch 00056: val_accuracy improved from 0.77792 to 0.78228, saving model to skin.h5
25/25 [==============================] - 11s 448ms/step - loss: 0.6227 - accuracy: 0.7637 - val_loss: 0.6419 - val_accuracy: 0.7823
Epoch 57/100
25/25 [==============================] - ETA: 0s - loss: 0.6075 - accuracy: 0.7692
Epoch 00057: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 446ms/step - loss: 0.6075 - accuracy: 0.7692 - val_loss: 0.6764 - val_accuracy: 0.7536
Epoch 58/100
25/25 [==============================] - ETA: 0s - loss: 0.6014 - accuracy: 0.7694
Epoch 00058: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 446ms/step - loss: 0.6014 - accuracy: 0.7694 - val_loss: 0.6726 - val_accuracy: 0.7598
Epoch 59/100
25/25 [==============================] - ETA: 0s - loss: 0.5956 - accuracy: 0.7744
Epoch 00059: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 445ms/step - loss: 0.5956 - accuracy: 0.7744 - val_loss: 0.6753 - val_accuracy: 0.7629
Epoch 60/100
25/25 [==============================] - ETA: 0s - loss: 0.5903 - accuracy: 0.7749
Epoch 00060: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 445ms/step - loss: 0.5903 - accuracy: 0.7749 - val_loss: 0.6488 - val_accuracy: 0.7673
Epoch 61/100
25/25 [==============================] - ETA: 0s - loss: 0.6074 - accuracy: 0.7687
Epoch 00061: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 440ms/step - loss: 0.6074 - accuracy: 0.7687 - val_loss: 0.6556 - val_accuracy: 0.7785
Epoch 62/100
25/25 [==============================] - ETA: 0s - loss: 0.5874 - accuracy: 0.7762
Epoch 00062: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 441ms/step - loss: 0.5874 - accuracy: 0.7762 - val_loss: 0.6591 - val_accuracy: 0.7673
Epoch 63/100
25/25 [==============================] - ETA: 0s - loss: 0.5865 - accuracy: 0.7801
Epoch 00063: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 442ms/step - loss: 0.5865 - accuracy: 0.7801 - val_loss: 0.6383 - val_accuracy: 0.7717
Epoch 64/100
25/25 [==============================] - ETA: 0s - loss: 0.5719 - accuracy: 0.7816
Epoch 00064: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 442ms/step - loss: 0.5719 - accuracy: 0.7816 - val_loss: 0.6831 - val_accuracy: 0.7530
Epoch 65/100
25/25 [==============================] - ETA: 0s - loss: 0.6178 - accuracy: 0.7600
Epoch 00065: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 459ms/step - loss: 0.6178 - accuracy: 0.7600 - val_loss: 0.6597 - val_accuracy: 0.7760
Epoch 66/100
25/25 [==============================] - ETA: 0s - loss: 0.5846 - accuracy: 0.7767
Epoch 00066: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 445ms/step - loss: 0.5846 - accuracy: 0.7767 - val_loss: 0.6497 - val_accuracy: 0.7729
Epoch 67/100
25/25 [==============================] - ETA: 0s - loss: 0.5952 - accuracy: 0.7803
Epoch 00067: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 445ms/step - loss: 0.5952 - accuracy: 0.7803 - val_loss: 0.6751 - val_accuracy: 0.7667
Epoch 68/100
25/25 [==============================] - ETA: 0s - loss: 0.5776 - accuracy: 0.7799
Epoch 00068: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 445ms/step - loss: 0.5776 - accuracy: 0.7799 - val_loss: 0.7063 - val_accuracy: 0.7592
Epoch 69/100
25/25 [==============================] - ETA: 0s - loss: 0.5947 - accuracy: 0.7772
Epoch 00069: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 443ms/step - loss: 0.5947 - accuracy: 0.7772 - val_loss: 0.6580 - val_accuracy: 0.7810
Epoch 70/100
25/25 [==============================] - ETA: 0s - loss: 0.5614 - accuracy: 0.7827
Epoch 00070: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 441ms/step - loss: 0.5614 - accuracy: 0.7827 - val_loss: 0.6510 - val_accuracy: 0.7817
Epoch 71/100
25/25 [==============================] - ETA: 0s - loss: 0.5506 - accuracy: 0.7891
Epoch 00071: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 453ms/step - loss: 0.5506 - accuracy: 0.7891 - val_loss: 0.6476 - val_accuracy: 0.7742
Epoch 72/100
25/25 [==============================] - ETA: 0s - loss: 0.5503 - accuracy: 0.7877
Epoch 00072: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 444ms/step - loss: 0.5503 - accuracy: 0.7877 - val_loss: 0.6571 - val_accuracy: 0.7735
Epoch 73/100
25/25 [==============================] - ETA: 0s - loss: 0.5590 - accuracy: 0.7834
Epoch 00073: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 444ms/step - loss: 0.5590 - accuracy: 0.7834 - val_loss: 0.6907 - val_accuracy: 0.7617
Epoch 74/100
25/25 [==============================] - ETA: 0s - loss: 0.5940 - accuracy: 0.7773
Epoch 00074: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 442ms/step - loss: 0.5940 - accuracy: 0.7773 - val_loss: 0.6924 - val_accuracy: 0.7517
Epoch 75/100
25/25 [==============================] - ETA: 0s - loss: 0.5682 - accuracy: 0.7833
Epoch 00075: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 451ms/step - loss: 0.5682 - accuracy: 0.7833 - val_loss: 0.6517 - val_accuracy: 0.7823
Epoch 76/100
25/25 [==============================] - ETA: 0s - loss: 0.5748 - accuracy: 0.7804
Epoch 00076: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 443ms/step - loss: 0.5748 - accuracy: 0.7804 - val_loss: 0.6589 - val_accuracy: 0.7810
Epoch 77/100
25/25 [==============================] - ETA: 0s - loss: 0.5541 - accuracy: 0.7866
Epoch 00077: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 442ms/step - loss: 0.5541 - accuracy: 0.7866 - val_loss: 0.6603 - val_accuracy: 0.7742
Epoch 78/100
25/25 [==============================] - ETA: 0s - loss: 0.5479 - accuracy: 0.7856
Epoch 00078: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 444ms/step - loss: 0.5479 - accuracy: 0.7856 - val_loss: 0.6804 - val_accuracy: 0.7611
Epoch 79/100
25/25 [==============================] - ETA: 0s - loss: 0.5421 - accuracy: 0.7879
Epoch 00079: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 449ms/step - loss: 0.5421 - accuracy: 0.7879 - val_loss: 0.6685 - val_accuracy: 0.7711
Epoch 80/100
25/25 [==============================] - ETA: 0s - loss: 0.5469 - accuracy: 0.7860
Epoch 00080: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 443ms/step - loss: 0.5469 - accuracy: 0.7860 - val_loss: 0.6773 - val_accuracy: 0.7704
Epoch 81/100
25/25 [==============================] - ETA: 0s - loss: 0.5611 - accuracy: 0.7884
Epoch 00081: val_accuracy did not improve from 0.78228
25/25 [==============================] - 12s 466ms/step - loss: 0.5611 - accuracy: 0.7884 - val_loss: 0.6920 - val_accuracy: 0.7704
Epoch 82/100
25/25 [==============================] - ETA: 0s - loss: 0.5360 - accuracy: 0.7882
Epoch 00082: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 444ms/step - loss: 0.5360 - accuracy: 0.7882 - val_loss: 0.6474 - val_accuracy: 0.7704
Epoch 83/100
25/25 [==============================] - ETA: 0s - loss: 0.5708 - accuracy: 0.7780
Epoch 00083: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 441ms/step - loss: 0.5708 - accuracy: 0.7780 - val_loss: 0.7048 - val_accuracy: 0.7692
Epoch 84/100
25/25 [==============================] - ETA: 0s - loss: 0.6271 - accuracy: 0.7682
Epoch 00084: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 442ms/step - loss: 0.6271 - accuracy: 0.7682 - val_loss: 0.6851 - val_accuracy: 0.7654
Epoch 85/100
25/25 [==============================] - ETA: 0s - loss: 0.6746 - accuracy: 0.7598
Epoch 00085: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 441ms/step - loss: 0.6746 - accuracy: 0.7598 - val_loss: 0.6804 - val_accuracy: 0.7735
Epoch 86/100
25/25 [==============================] - ETA: 0s - loss: 0.6103 - accuracy: 0.7726
Epoch 00086: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 455ms/step - loss: 0.6103 - accuracy: 0.7726 - val_loss: 0.6563 - val_accuracy: 0.7792
Epoch 87/100
25/25 [==============================] - ETA: 0s - loss: 0.5485 - accuracy: 0.7915
Epoch 00087: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 444ms/step - loss: 0.5485 - accuracy: 0.7915 - val_loss: 0.6707 - val_accuracy: 0.7623
Epoch 88/100
25/25 [==============================] - ETA: 0s - loss: 0.5524 - accuracy: 0.7882
Epoch 00088: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 439ms/step - loss: 0.5524 - accuracy: 0.7882 - val_loss: 0.6540 - val_accuracy: 0.7773
Epoch 89/100
25/25 [==============================] - ETA: 0s - loss: 0.5751 - accuracy: 0.7725
Epoch 00089: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 442ms/step - loss: 0.5751 - accuracy: 0.7725 - val_loss: 0.6644 - val_accuracy: 0.7573
Epoch 90/100
25/25 [==============================] - ETA: 0s - loss: 0.5443 - accuracy: 0.7894
Epoch 00090: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 442ms/step - loss: 0.5443 - accuracy: 0.7894 - val_loss: 0.6543 - val_accuracy: 0.7723
Epoch 91/100
25/25 [==============================] - ETA: 0s - loss: 0.5436 - accuracy: 0.7912
Epoch 00091: val_accuracy did not improve from 0.78228
25/25 [==============================] - 11s 443ms/step - loss: 0.5436 - accuracy: 0.7912 - val_loss: 0.6988 - val_accuracy: 0.7611
Epoch 92/100
25/25 [==============================] - ETA: 0s - loss: 0.5557 - accuracy: 0.7894
Epoch 00092: val_accuracy improved from 0.78228 to 0.78790, saving model to skin.h5
25/25 [==============================] - 11s 444ms/step - loss: 0.5557 - accuracy: 0.7894 - val_loss: 0.6505 - val_accuracy: 0.7879
Epoch 93/100
25/25 [==============================] - ETA: 0s - loss: 0.5190 - accuracy: 0.7997
Epoch 00093: val_accuracy did not improve from 0.78790
25/25 [==============================] - 11s 454ms/step - loss: 0.5190 - accuracy: 0.7997 - val_loss: 0.6586 - val_accuracy: 0.7767
Epoch 94/100
25/25 [==============================] - ETA: 0s - loss: 0.5415 - accuracy: 0.7897
Epoch 00094: val_accuracy did not improve from 0.78790
25/25 [==============================] - 11s 442ms/step - loss: 0.5415 - accuracy: 0.7897 - val_loss: 0.6555 - val_accuracy: 0.7817
Epoch 95/100
25/25 [==============================] - ETA: 0s - loss: 0.5303 - accuracy: 0.7910
Epoch 00095: val_accuracy did not improve from 0.78790
25/25 [==============================] - 11s 442ms/step - loss: 0.5303 - accuracy: 0.7910 - val_loss: 0.6701 - val_accuracy: 0.7773
Epoch 96/100
25/25 [==============================] - ETA: 0s - loss: 0.5598 - accuracy: 0.7842
Epoch 00096: val_accuracy did not improve from 0.78790
25/25 [==============================] - 11s 454ms/step - loss: 0.5598 - accuracy: 0.7842 - val_loss: 0.6806 - val_accuracy: 0.7810
Epoch 97/100
25/25 [==============================] - ETA: 0s - loss: 0.5346 - accuracy: 0.7882
Epoch 00097: val_accuracy did not improve from 0.78790
25/25 [==============================] - 11s 441ms/step - loss: 0.5346 - accuracy: 0.7882 - val_loss: 0.6660 - val_accuracy: 0.7810
Epoch 98/100
25/25 [==============================] - ETA: 0s - loss: 0.5424 - accuracy: 0.7861
Epoch 00098: val_accuracy did not improve from 0.78790
25/25 [==============================] - 11s 441ms/step - loss: 0.5424 - accuracy: 0.7861 - val_loss: 0.6620 - val_accuracy: 0.7717
Epoch 99/100
25/25 [==============================] - ETA: 0s - loss: 0.5303 - accuracy: 0.7934
Epoch 00099: val_accuracy did not improve from 0.78790
25/25 [==============================] - 11s 442ms/step - loss: 0.5303 - accuracy: 0.7934 - val_loss: 0.6704 - val_accuracy: 0.7867
Epoch 100/100
25/25 [==============================] - ETA: 0s - loss: 0.5165 - accuracy: 0.7960
Epoch 00100: val_accuracy did not improve from 0.78790
25/25 [==============================] - 11s 441ms/step - loss: 0.5165 - accuracy: 0.7960 - val_loss: 0.6848 - val_accuracy: 0.7698
```
```python
%load_ext tensorboard
%tensorboard --logdir logs
```
![Tensorboard](https://user-images.githubusercontent.com/72899789/96359977-bbba6200-1142-11eb-9a87-a2897f54180f.png)
```python
model_filename = "skin.h5"

from tensorflow.keras.models import load_model

model = load_model(model_filename)
```
```python
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(x_validate)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_validate,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

 

# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(7)) 

```
![confussion matrix](https://user-images.githubusercontent.com/72899789/96360092-cd503980-1143-11eb-9aa7-cb9368e23e20.png)
# Step 9: Prediction using validation data
Prediction using validation data, here is the prediction of data on x_validate, here is an example
```python
predictions = model.predict(x_validate)
predictions

def index_pred(ind):
  plt.imshow(x_validate[ind][:,:,0]);

  print()
  
pred=str(np.argmax(predictions[ind]))
if pred == '0':
  print("Actinic keratoses")
elif pred == '1':
  print('Basal cell carcinoma')
elif pred == '2':
  print('Benign keratosis-like lesions')
elif pred == '3':
  print('Dermatofibroma')
elif pred == '4':
  print('Melanocytic nevi')
elif pred == '5':
  print('melanoma')
elif pred == '6':
  print('Vascular lesions')

index_pred(53)
```

