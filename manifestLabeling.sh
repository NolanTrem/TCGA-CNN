#!/bin/bash

if grep -q "Primary Tumor\|Solid Tissue Normal" /Volumes/NolansDrive/TCGA-CNN/Breast/breastmanifestfinal.txt; then
	echo "Manifest has been labeled already. No need to scrape again."
else
	echo "Labeling manifest now."
	python dataPreprocessing.py /Volumes/NolansDrive/TCGA-CNN/Breast/breastmanifestfinal.txt
fi
if [ ! -d "/Volumes/NolansDrive/TCGA-CNN/Breast/breastCancerImages" ]; then
	echo "Creating images from .svs files."
	matlab -nodisplay -nosplash -nodesktop -r "my_script('/Volumes/NolansDrive/TCGA-CNN/Breast/breastmanifestfinal.txt', '/Volumes/NolansDrive/TCGA-CNN/Breast', 'breast'); quit;" -args /Volumes/NolansDrive/TCGA-CNN/Breast/breastmanifestfinal.txt /Volumes/NolansDrive/TCGA-CNN/Breast breast
else
	echo "Images have already been created. No need to create them again."
fi

if grep -q "Primary Tumor\|Solid Tissue Normal" /Volumes/NolansDrive/TCGA-CNN/Kidney/kidneymanifestfinal.txt; then
	echo "Manifest has been labeled already. No need to scrape again."
else
	echo "Labeling manifest now."
	python dataPreprocessing.py /Volumes/NolansDrive/TCGA-CNN/Kidney/kidneymanifestfinal.txt
fi
if [ ! -d "/Volumes/NolansDrive/TCGA-CNN/Kidney/kidneyCancerImages" ]; then
	echo "Creating images from .svs files."
	matlab -nodisplay -nosplash -nodesktop -r "my_script('/Volumes/NolansDrive/TCGA-CNN/Kidney/kidneymanifestfinal.txt', '/Volumes/NolansDrive/TCGA-CNN/Kidney', 'kidney'); quit;" -args /Volumes/NolansDrive/TCGA-CNN/Kidney/kidneymanifestfinal.txt /Volumes/NolansDrive/TCGA-CNN/Kidney kidney
else
	echo "Images have already been created. No need to create them again."
fi

if grep -q "Primary Tumor\|Solid Tissue Normal" /Volumes/NolansDrive/TCGA-CNN/Lung/lungmanifestfinal.txt; then
	echo "Manifest has been labeled already. No need to scrape again."
else
	echo "Labeling manifest now."
	python dataPreprocessing.py /Volumes/NolansDrive/TCGA-CNN/Lung/lungmanifestfinal.txt
fi
if [ ! -d "/Volumes/NolansDrive/TCGA-CNN/Lung/lungCancerImages" ]; then
	echo "Creating images from .svs files."
	matlab -nodisplay -nosplash -nodesktop -r "my_script('/Volumes/NolansDrive/TCGA-CNN/Lung/lungmanifestfinal.txt', '/Volumes/NolansDrive/TCGA-CNN/Lung', 'lung'); quit;" -args /Volumes/NolansDrive/TCGA-CNN/Lung/lungmanifestfinal.txt /Volumes/NolansDrive/TCGA-CNN/Lung lung
else
	echo "Images have already been created. No need to create them again."
fi