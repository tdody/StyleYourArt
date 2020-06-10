<figure>
    <img src="./static/oil-paint-banner.jpg" style="width:1920px;">
</figure>

## Table of Content

1. Import Modules and Libraries
2. Import Dataset
3. Set Directory Structure
4. Data Cleaning
5. Data Exploration

## Import Modules and Libraries


```python
import os, sys
sys.path.append('..')

%matplotlib inline
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import seaborn as sns
sns.set_style("darkgrid")
sns.set(font_scale = 1.5)

from sklearn.model_selection import train_test_split

import StyleYourArt
from StyleYourArt.dataloader import *
from StyleYourArt.tools import *
from StyleYourArt.models import *
```

    Using TensorFlow backend.


## Import Dataset


```python
## Locations
DATA_DIR = '../data/csv/'
IMAGE_DIR = '../data/images/'
SAVED_DIR = '../data/classes'
```

In a previous notebook, we built a module to load and clear the data. The code can be found in `dataloader.py`.


```python
## extract data from csv file
df = load_clean_data(DATA_DIR, IMAGE_DIR, verbose=True)
```

    ... loading painting data from master.csv
    ... painting data imported from master.csv
    ... setting dataframe schema
    ... exporting unique styles to styles.csv
    ... data loaded
    ... finding top 25 most popular styles
    ... filtering data.
       ... record count before filtering: 169677
       ... record count after filtering: 117050
    ... replace missing values



```python
## display dataset info
print_data_info(df)
```

    ******************* SUMMARY *******************

    Content
    -----------------------------------------------
    Rows: 117,050
    Cols: 14
    Number of artists: 2,434
    Number of unique styles: 18
    Date range: from 1401 to 2020


    Info
    -----------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 117050 entries, 3 to 19
    Data columns (total 14 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   artist_name      117050 non-null  object
     1   artist_url       117050 non-null  object
     2   completion_year  90570 non-null   float64
     3   content_id       117050 non-null  int64  
     4   height           117050 non-null  int64  
     5   image            117050 non-null  object
     6   json_file        117050 non-null  object
     7   style            117050 non-null  object
     8   title            117050 non-null  object
     9   url              117050 non-null  object
     10  width            117050 non-null  int64  
     11  format           117050 non-null  object
     12  file_loc         117050 non-null  object
     13  image_exists     117050 non-null  bool   
    dtypes: bool(1), float64(1), int64(3), object(9)
    memory usage: 12.6+ MB
    None



```python
## display a few records
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist_name</th>
      <th>artist_url</th>
      <th>completion_year</th>
      <th>content_id</th>
      <th>height</th>
      <th>image</th>
      <th>json_file</th>
      <th>style</th>
      <th>title</th>
      <th>url</th>
      <th>width</th>
      <th>format</th>
      <th>file_loc</th>
      <th>image_exists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>MISHRA ANIRBAN</td>
      <td>a-mishra</td>
      <td>NaN</td>
      <td>9223372032559846598</td>
      <td>3354</td>
      <td>https://uploads7.wikiart.org/00272/images/a-mi...</td>
      <td>a-mishra.json</td>
      <td>Conceptual Art</td>
      <td>Vision</td>
      <td>vision</td>
      <td>3559</td>
      <td>jpg</td>
      <td>../data/images/a-mishra/unknown-year/922337203...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MISHRA ANIRBAN</td>
      <td>a-mishra</td>
      <td>NaN</td>
      <td>9223372032559846599</td>
      <td>3354</td>
      <td>https://uploads5.wikiart.org/00272/images/a-mi...</td>
      <td>a-mishra.json</td>
      <td>Conceptual Art</td>
      <td>Time</td>
      <td>time</td>
      <td>3559</td>
      <td>jpg</td>
      <td>../data/images/a-mishra/unknown-year/922337203...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MISHRA ANIRBAN</td>
      <td>a-mishra</td>
      <td>NaN</td>
      <td>9223372032559846600</td>
      <td>3656</td>
      <td>https://uploads2.wikiart.org/00272/images/a-mi...</td>
      <td>a-mishra.json</td>
      <td>Conceptual Art</td>
      <td>Persistence of memory</td>
      <td>persistence-of-memory</td>
      <td>3440</td>
      <td>jpg</td>
      <td>../data/images/a-mishra/unknown-year/922337203...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MISHRA ANIRBAN</td>
      <td>a-mishra</td>
      <td>NaN</td>
      <td>9223372032559846597</td>
      <td>3278</td>
      <td>https://uploads0.wikiart.org/00272/images/a-mi...</td>
      <td>a-mishra.json</td>
      <td>Conceptual Art</td>
      <td>Conversation</td>
      <td>conversation</td>
      <td>3511</td>
      <td>jpg</td>
      <td>../data/images/a-mishra/unknown-year/922337203...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MISHRA ANIRBAN</td>
      <td>a-mishra</td>
      <td>NaN</td>
      <td>9223372032559846454</td>
      <td>3800</td>
      <td>https://uploads4.wikiart.org/00272/images/a-mi...</td>
      <td>a-mishra.json</td>
      <td>Abstract Art</td>
      <td>Expression of sadness - II</td>
      <td>expression-of-sadness-ii</td>
      <td>4665</td>
      <td>jpg</td>
      <td>../data/images/a-mishra/unknown-year/922337203...</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_order = df[['completion_year', 'style']]
df_order = pd.DataFrame(df_order.groupby('style').quantile(0.02)['completion_year'].sort_values()).reset_index()
df_order = pd.merge(df[['completion_year', 'style']], df_order, on="style")
df_order = df_order.sort_values(by='completion_year_y')
df_order = df_order.rename(columns={"completion_year_x": "Year"})
```

## Set Directory Structure

Keras can use the directory structure to determine the image classes. To do so, we need to store each image into a folder titled with the painting style.  
A python function `organize_directories` in module `models.py` is used to move each picture in its corresponding style directory. Note that the images are also resized as **224x224** and saved as **png**. Finally, the distribution is done by splitting the dataset into a train and test sets. The split is done by stratifying the styles and assigning randomly **20%** of each styles into the test set.



```python
!tree ../data/classes -L 1
```

    [01;34m../data/classes[00m
    ├── [01;34mtest[00m
    └── [01;34mtrain[00m

    2 directories, 0 files



```python
!tree ../data/classes/train -L 1
```

    [01;34m../data/classes/train[00m
    ├── [01;34mAbstract\ Art[00m
    ├── [01;34mArt\ Nouveau\ (Modern)[00m
    ├── [01;34mBaroque[00m
    ├── [01;34mConceptual\ Art[00m
    ├── [01;34mCubism[00m
    ├── [01;34mExpressionism[00m
    ├── [01;34mImpressionism[00m
    ├── [01;34mMinimalism[00m
    ├── [01;34mNaïve\ Art\ (Primitivism)[00m
    ├── [01;34mNeo-Expressionism[00m
    ├── [01;34mNeoclassicism[00m
    ├── [01;34mPop\ Art[00m
    ├── [01;34mRealism[00m
    ├── [01;34mRenaissance[00m
    ├── [01;34mRomanticism[00m
    ├── [01;34mSurrealism[00m
    ├── [01;34mSymbolism[00m
    └── [01;34mUkiyo-e[00m

    18 directories, 0 files



```python
## repeat split
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['style'])

## disconnect train and test from original dataset
X_train = X_train.copy()
X_test = X_test.copy()
```


```python
## create a new feature corresponding to the final locations
X_train['save_dir'] = SAVED_DIR + '/train/' + X_train['style'] + '/' + X_train['content_id'].astype(str) + '.png'
X_test['save_dir'] = SAVED_DIR + '/test/' + X_test['style'] + '/' + X_test['content_id'].astype(str) + '.png'
```


```python
## display examples of the resized images
%matplotlib inline
display_images(X_train, list(df_order['style'].unique()), dir_feature='save_dir')
plt.tight_layout()
plt.show();
```

<figure>
<p align="center">
    <img src="./static/output_19_0.png">
    </p>
</figure>



```python
## compbine both sets into a unique dataset
X_train['set'] = 'Train'
X_test['set'] = 'Test'
df_full = pd.concat([X_train, X_test])
```


```python
## compute class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(X_train['style']),
                                                  X_train['style'])
weights = pd.Series(dict(zip(np.unique(X_train['style']),class_weights))).sort_values(ascending=False)
```


```python
## compute distribution
%matplotlib inline
df_plot = df_full.groupby(['set', 'style']).size().reset_index().pivot(columns='set', index='style', values=0).sort_values(['Train'], ascending=True)
df_plot = df_plot[['Train', 'Test']]

## plot style distribution
fig, axes = plt.subplots(1,2,figsize=(16,10), sharey=True)
df_plot.plot(kind='barh', stacked=True, ax=axes[0])
weights.plot(kind='barh', ax=axes[1], color='r')
plt.tight_layout()
axes[0].set_title("Distribution of Training and Test Sets")
axes[1].set_title("Class weights")
axes[0].set_xlabel("Count")
axes[1].set_xlabel("Weight value")
plt.show();
```

<figure>
<p align="center">
    <img src="./static/output_22_0.png">
    </p>
</figure>


As shown above, the dataset has been divided into the train and test set while maintaining the class proportion. In order to avoid bias over class that are overly represented (Impressionism, Realism...). The metrics of interest will be weighted so that each class is assigned the same importance.

## Convolutional Neural Network
In this section, we will train a CNN to predict the species feature. The approach is divided between the following steps:

- Encode the target feature
- Download the data
- Split the data between a training and test set
- Perform data augmentation
- Determine the cost function to be optimized
- Data Loader, Validation and Data Augmentation

In order for our model to generalize well on unseen data, a good practice consists of using image transformation to create new unseen examples.
We need to ensure that our model does not over fit the training data. To do so, we are using a training set and a test set both taken from the original dataset.

Keras contains useful tools to help process image files and feed them in batches to the model. We will be using a generator for both the train and test phases.

First, we must create a new feature to our dataset which contains the full path to each image.
Then, we can create two generators, the training generator will contains several data augmentation transformation (horizontal and vertical flips, zoom).
Both the train and test generator will normalize the pixel values.
Finally, the images will be sent to the model using batches of 16 RGB images reshaped at 224x224.

### Transfer Learning - First Generation  
Before we train a model on the entire dataset, we need to investigate the following modeling choices:
1. Architecture
2. Optimization metrics
3. Callbacks
4. Optimizers

During this initial phase, we will test 5 different base models using only 1,000 images per class. The considered base models are:
- ResNet50
- Inception V3
- MobileNet V2
- Xception
- VGG16

For each model, we remove the top layer and add a custom model to it. This top model is defined as follows:
- Conv2D (512, 2, relu)
- Flatten
- Dense (2048, relu)
- Dense (1024, relu)
- Dense (512, relu)
- Dense (256, relu)
- Dense (64, relu)
- Dense (18, softmax)


```python
## load results
results = pd.read_csv('../data/training_phase1.csv')
results['size_total_MB'] = results['size_base_MB'] + results['size_top_MB']
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>tag</th>
      <th>val_loss</th>
      <th>val_accuracy</th>
      <th>loss</th>
      <th>accuracy</th>
      <th>size_base_MB</th>
      <th>size_top_MB</th>
      <th>size_total_MB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.a.1000</td>
      <td>ResNet50</td>
      <td>2.512670</td>
      <td>0.208333</td>
      <td>2.610792</td>
      <td>0.181167</td>
      <td>94.7</td>
      <td>131.6</td>
      <td>226.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.b.1000</td>
      <td>VGG16</td>
      <td>1.930195</td>
      <td>0.373611</td>
      <td>1.895698</td>
      <td>0.385111</td>
      <td>58.9</td>
      <td>106.5</td>
      <td>165.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.c.1000</td>
      <td>InceptionV3</td>
      <td>1.890462</td>
      <td>0.391667</td>
      <td>1.687830</td>
      <td>0.453889</td>
      <td>87.9</td>
      <td>89.7</td>
      <td>177.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.d.1000</td>
      <td>MobileNetV2</td>
      <td>1.745986</td>
      <td>0.436389</td>
      <td>1.514181</td>
      <td>0.501667</td>
      <td>9.4</td>
      <td>119.0</td>
      <td>128.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.e.1000</td>
      <td>Xception</td>
      <td>1.799670</td>
      <td>0.419722</td>
      <td>1.740675</td>
      <td>0.439667</td>
      <td>83.7</td>
      <td>131.6</td>
      <td>215.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
fig, ax = plt.subplots(figsize=(8,6))
results[['tag', 'val_accuracy', 'accuracy']].set_index('tag').plot(ax=ax, kind='barh', color=['b','r'])
ax.set_title("Model Accuracy")
ax.set_xlabel('Accuracy')
plt.show();
```

<figure>
<p align="center">
    <img src="./static/output_27_0.png">
    </p>
</figure>


The MobileNet V2 seems to be the best choice, it scores the highest in term of validation accuracy and is also the lightest model (128M). However, this weight is still a bit much to easily deploy on a Heroku instance. We make a new version of the top model by decreasing the size of the top model first dense layer from 2048 neurons to 1024.

### Transfer Learning - Second Generation  

#### Architecture - MobileNet V2

<figure>
<p align="center">
    <img src="./static/MobileNetV2.png" style="width:500px;">
    </p>
</figure>

- There are 3 layers for both types of blocks.
- The first layer is 1×1 convolution with ReLU6.
- The second layer is the depthwise convolution.
- The third layer is another 1×1 convolution but without any non-linearity. It is claimed that if ReLU is used again, the deep networks only have the power of a linear classifier on the non-zero volume part of the output domain.

<figure>
<p align="center">
    <img src="./static/MobileNetV2_structure.png" style="width:400px;">
    </p>
</figure>

<figure>
<p align="center">
    <img src="./static/MobileNetV2_overall.png" style="width:350px;">
    </p>
</figure>

where:
- t: expansion factor
- c: number of output channels
- n: repeating number
- s: stride.
- 3×3 kernels are used for spatial convolution.  

This top model is defined as follows:
- Conv2D (512, 2, relu)
- Flatten
- Dense (1024, relu)
- Dense (1024, relu)
- Dense (512, relu)
- Dense (256, relu)
- Dense (64, relu)
- Dense (18, softmax)

#### Results

**Training**
<figure>
<img src="./static/1.d.1000/history.png" style="width:1920px;">
</figure>

**Training**
<table>
    <tr>
        <td>
<figure>
    <img src="./static/1.d.1000/Train_Set_AUC_ROC.png" style="width:600px;"></figure></td>
        <td>
            <figure>
    <img src="./static/1.d.1000/Test_Set_AUC_ROC.png" style="width:600px;"></figure>
        </td>
    </tr>
    </table>

**Predictions**
<table>
    <tr>
        <td>
<figure>
    <img src="./static/1.d.1000/Train_Set_confusion_matrix.png"></figure></td>
        <td>
            <figure>
    <img src="./static/1.d.1000/Train_Set_Normalize_confusion_matrix.png"></figure>
        </td>
    </tr>
    </table>

<table>
    <tr>
        <td>
<figure>
    <img src="./static/1.d.1000/Test_Set_confusion_matrix.png"></figure></td>
        <td>
            <figure>
    <img src="./static/1.d.1000/Test_Set_Normalize_confusion_matrix.png"></figure>
        </td>
    </tr>
    </table>


```python
## load prediction reports
train_report = pd.read_csv("../models/1.d.1000/Train_Set_report.csv", index_col=0)[['precision', 'recall', 'f1-score']]
test_report = pd.read_csv("../models/1.d.1000/Test_Set_report.csv", index_col=0)[['precision', 'recall', 'f1-score']]

train_report = train_report.iloc[0:-3,:]
test_report = test_report.iloc[0:-3,:]

report = pd.merge(left=train_report, right=test_report, left_index=True, right_index=True, suffixes=('_train', '_test'))
```


```python
## plot report
sns.set(font_scale = 1.5)
sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize=(16,6))
report.plot(ax=ax, style=['--',':','-.','--',':','-.'], color=['b','r','g', 'orange', 'mediumpurple', 'crimson'])

ax.set_xticks(range(0,18))
ax.set_xticklabels(report.index, fontsize=15)
plt.xticks(rotation=45, ha = 'right')

ax.legend(loc='upper center', bbox_to_anchor=(1.1, 0.8),fancybox=True, shadow=True, ncol=1)
ax.set_title("Prediction Report")
ax.set_ylim(0, 1.)
ax.set_xlim(0, 17)
plt.tight_layout();
```

<figure>
    <img src="./static/output_38_0.png">
</figure>


### Transfer Learning - Third Generation  

**Training**
<figure>
<img src="./static/2_d/history.png" style="width:1920px;">
</figure>

**Training**
<table>
    <tr>
        <td>
<figure>
    <img src="./static/2_d/Train_Set_AUC_ROC.png" style="width:600px;"></figure></td>
        <td>
            <figure>
    <img src="./static/2_d/Test_Set_AUC_ROC.png" style="width:600px;"></figure>
        </td>
    </tr>
    </table>

**Predictions**
<table>
    <tr>
        <td>
<figure>
    <img src="./static/2_d/Train_Set_confusion_matrix.png"></figure></td>
        <td>
            <figure>
    <img src="./static/2_d/Train_Set_Normalize_confusion_matrix.png"></figure>
        </td>
    </tr>
    </table>

<table>
    <tr>
        <td>
<figure>
    <img src="./static/2_d/Test_Set_confusion_matrix.png"></figure></td>
        <td>
            <figure>
    <img src="./static/2_d/Test_Set_Normalize_confusion_matrix.png"></figure>
        </td>
    </tr>
    </table>


```python
## load prediction reports
train_report = pd.read_csv("../models/2_d/Train_Set_report.csv", index_col=0)[['precision', 'recall', 'f1-score']]
test_report = pd.read_csv("../models/2_d/Test_Set_report.csv", index_col=0)[['precision', 'recall', 'f1-score']]

train_report = train_report.iloc[0:-3,:]
test_report = test_report.iloc[0:-3,:]

report = pd.merge(left=train_report, right=test_report, left_index=True, right_index=True, suffixes=('_train', '_test'))
```


```python
## plot report
sns.set(font_scale = 1.5)
sns.set_style('whitegrid')

fig, ax = plt.subplots(figsize=(16,6))
report.plot(ax=ax, style=['--',':','-.','--',':','-.'], color=['b','r','g', 'orange', 'mediumpurple', 'crimson'])

ax.set_xticks(range(0,18))
ax.set_xticklabels(report.index, fontsize=15)
plt.xticks(rotation=45, ha = 'right')

ax.legend(loc='upper center', bbox_to_anchor=(1.1, 0.8),fancybox=True, shadow=True, ncol=1)
ax.set_title("Prediction Report")
ax.set_ylim(0, 1.)
ax.set_xlim(0, 17)
plt.tight_layout();
```

<figure>
<p align="center">
    <img src="./static/output_45_0.png">
    </p>
</figure>
<br>