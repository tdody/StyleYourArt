# TODO

## Objectives

The goals of this project are defined as follows:
1. Build an image classifier to identify the style of a painting.
2. The final product must be web-based and allow the users their own pictures.
3. The final API will present the results of the prediction (plot) and suggest additional information to the user (similar paintings, famous artists)

**The goal is not to archive the best accuracy at all costs. The main objective of this project is instead to go through a full development (data retrieval, EDA, modeling, API).**

## Data

[WikiArt](https://www.wikiart.org/) is our data source. This domain presents both public domain and copyright protected artworks. WikiArt is a very popular source of data for Machine Learning applications. Indeed, data retrieval is relatively easy and existing python libraries have already been developed.

## Load Data

The data is retrieved using a web scrapper from [Lucas David](https://github.com/lucasdavid/wikiart).
The content scrapper retrieves all the available information hosted on [WikiArt](https://www.wikiart.org/).

The following dataset and weights are used for our training:
<figure>
    <img src="https://github.com/tdody/StyleYourArt/blob/master/app/static/output_22_0.png">
</figure>

## App Architecture

<figure>
    <img src="https://github.com/tdody/StyleYourArt/blob/master/app/static/Architecture.png">
</figure>

## Model

In order to make our predictions, we will use transfer learning. Our base model will be the famous [MobileNetV2](https://arxiv.org/abs/1801.04381). This model has several advantages against larger models such as VGG or ResNet. First, it is very lightweight. This helps our app to be more responsive when deployed. Second, after performing several test on a subset of the dataset, the accuracy and loss scores archived by the MobileNetV2 were superior to those of other base models.  
Our top model consists of the following layers:
 - Conv2D (512, 2, relu)
 - Flatten
 - Dense (2048, relu)
 - Dense (1024, relu)
 - Dense (512, relu)
 - Dense (256, relu)
 - Dense (64, relu)
 - Dense (18, softmax)

## Performance

### Training Results

<figure>
    <img src="https://github.com/tdody/StyleYourArt/blob/master/app/static/2_d/history.png" style="height:500px">
</figure>

### Prediction Results

<table>
    <tr>
        <td>
            <figure>
                <img src="https://github.com/tdody/StyleYourArt/blob/master/app/static/2_d/Train_Set_AUC_ROC.png">
            </figure>
        </td>
        <td>
            <figure>
                <img src="https://github.com/tdody/StyleYourArt/blob/master/app/static/2_d/Test_Set_AUC_ROC.png">
            </figure>
        </td>
    </tr>
</table>

<table>
    <tr>
        <td>
            <figure>
                <img src="https://github.com/tdody/StyleYourArt/blob/master/app/static/2_d/Train_Set_Normalize_confusion_matrix.png">
            </figure>
        </td>
        <td>
            <figure>
                <img src="https://github.com/tdody/StyleYourArt/blob/master/app/static/2_d/Test_Set_Normalize_confusion_matrix.png">
            </figure>
        </td>
    </tr>
</table>

<figure>
    <img src="https://github.com/tdody/StyleYourArt/blob/master/app/static/output_45_0.png">
</figure>