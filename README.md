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
# Step 4: Data Cleansing
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
