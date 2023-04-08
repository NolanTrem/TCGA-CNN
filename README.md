# TCGA-CNN
A convolutional neural network implementation for predicting gene expression from H&amp;E-stained histology slides.

## Data labeling method

Data presented by TCGA project doesn't include labels conducive to training a neural network in their manifest data.
To circumvent this problem, a python script to scrape web data is implemented. This script scrapes the TCGA whole slide
image database for the appropriate label—either as primary tumor or as solid tissue normal. The input for this script
is the manifest file produced by the GDC Data Portal.

![Alt text](images/exampleManifestBeforeLabeling.png "Manifest before labeling")
Manifest before labeling.
![Alt text](images/exampleManifesetAfterLabeling.png "Manifest after labeling")
Manifest after labeling.

The manifest is created in the [GDC Data Portal repository](https://portal.gdc.cancer.gov/repository) by adding the
desired files to the cart and downloading. From there, the manifest is entered into the dataProcessing.py file for
processing to occur.

