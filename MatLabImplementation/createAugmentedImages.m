%function [] = augmentedImages(originalImages)

%%

originalImages = '/Volumes/NolansDrive/TCGA-CNN/commonCancerDataset';
if ~exist(strcat(originalImages, '/augmentedImages'), 'dir')
    mkdir(strcat(originalImages, '/augmentedImages'));
end
outputPath = '/Volumes/NolansDrive/TCGA-CNN/commonCancerDataset/augmentedImages';
primaryTumorPath = dir(strcat(originalImages, '/primaryTumor/*.png'));
solidTissueNormalPath = dir(strcat(originalImages, '/solidTissueNormal/*.png'));

primaryTumorCleanImages = dir(strcat(originalImages, '/primaryTumor/*.png'));
solidTissueNormalCleanImages = dir(strcat(originalImages, '/solidTissueNormal/*.png'));

% if ~exist(strcat(outputPath, '/primaryTumor'), 'dir')
%     mkdir(strcat(outputPath, '/primaryTumor'));
% end
% 
% if ~exist(strcat(outputPath, '/solidTissueNormal'), 'dir')
%     mkdir(strcat(outputPath, '/solidTissueNormal'));
% end

primaryTumorAugmentedImages = dir(strcat(outputPath, '/primaryTumor'));
solidTissueNormalAugmentedImages = dir(strcat(outputPath, '/solidTissueNormal'));

for i = 1:numel(primaryTumorCleanImages)
    image = imread(fullfile(originalImages, 'primaryTumor/', primaryTumorCleanImages(i).name));
    
    tform1 = randomAffine2d(Rotation=[0 5]);
    image = imwarp(image,tform1);

    tform2 = randomAffine2d(XShear=[-5 5]);
    image = imwarp(image, tform2);

    tform3 = randomAffine2d(YShear=[-5 5]);
    image = imwarp(image, tform3);

    image = jitterColorHSV(image,'Contrast',0.2,'Hue',0.2,'Saturation',0.2,'Brightness',0.2);

    sigma = 1+rand;
    image= imgaussfilt(image,sigma);
    name = strcat('augId1', primaryTumorCleanImages(i).name);

    imwrite(image, fullfile(outputPath, 'primaryTumor/', name), 'jpg');
end

for i = 1:numel(solidTissueNormalCleanImages)
    name = solidTissueNormalCleanImages(i).name;
    image = imread(fullfile(originalImages, 'solidTissueNormal/', solidTissueNormalCleanImages(i).name));
    
    tform1 = randomAffine2d(Rotation=[0 5]);
    image = imwarp(image,tform1);

    tform2 = randomAffine2d(XShear=[-5 5]);
    image = imwarp(image, tform2);

    tform3 = randomAffine2d(YShear=[-5 5]);
    image = imwarp(image, tform3);

    image = jitterColorHSV(image,'Contrast',0.2,'Hue',0.2,'Saturation',0.2,'Brightness',0.2);

    sigma = rand;
    image= imgaussfilt(image,sigma);
    name = strcat('augId1', solidTissueNormalCleanImages(i).name);

    imwrite(image, fullfile(outputPath, 'solidTissueNormal/', name), 'jpg');
end
