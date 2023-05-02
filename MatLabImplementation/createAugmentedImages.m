%function [] = augmentedImages(originalImages)

%%

originalImages = 'E:\TCGA-CNN\commonCancerDataset';
if ~exist(strcat(originalImages, '/augmentedImages'), 'dir')
    mkdir(strcat(originalImages, '/augmentedImages'));
end
outputPath = 'E:\TCGA-CNN\commonCancerDataset\augmentedImages';
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
    image = imread(fullfile(originalImages, 'primaryTumor\', primaryTumorCleanImages(i).name));
    
    tform1 = randomAffine2d(Rotation=[0 45]);
    image = imwarp(image,tform1);

    tform2 = randomAffine2d(XShear=[-45 45]);
    image = imwarp(image, tform2);

    tform3 = randomAffine2d(YShear=[-45 45]);
    image = imwarp(image, tform3);

    image = jitterColorHSV(image,'Contrast',1,'Hue',1,'Saturation',1,'Brightness',1);

    sigma = 1+5*rand;
    image= imgaussfilt(image,sigma); 

    imwrite(image, fullfile(outputPath, 'primaryTumor\' , name, 'augId1.png'));
end

for i = 1:numel(solidTissueNormalCleanImages)
    name = solidTissueNormalCleanImages(i).name;
    image = imread(fullfile(originalImages, 'solidTissueNormal\', solidTissueNormalCleanImages(i).name));
    
    tform1 = randomAffine2d(Rotation=[0 45]);
    image = imwarp(image,tform1);

    tform2 = randomAffine2d(XShear=[-45 45]);
    image = imwarp(image, tform2);

    tform3 = randomAffine2d(YShear=[-45 45]);
    image = imwarp(image, tform3);

    image = jitterColorHSV(image,'Contrast',1,'Hue',1,'Saturation',1,'Brightness',1);

    sigma = 1+5*rand;
    image= imgaussfilt(image,sigma); 

    imwrite(image, fullfile(outputPath, 'solidTissueNormal\' , name, 'augId1.png'));
end
