function [] = createAugmentedImages(originalImages)

%%
if ~exist(strcat(originalImages, '/augmentedImages'), 'dir')
    mkdir(strcat(originalImages, '/augmentedImages'));
end
outputPath = '/Users/nolantremelling/Downloads/commonCancerDataset/augmentedImages';
primaryTumorPath = dir(strcat(originalImages, '/primaryTumor'));
solidTissueNormalPath = dir(strcat(originalImages, '/solidTissueNormal'));

primaryTumorCleanImages = dir(strcat(originalImages, '/primaryTumor/*.png'));
solidTissueNormalCleanImages = dir(strcat(originalImages, '/solidTissueNormal/*.png'));

if ~exist(strcat(outputPath, '/augmentedImages/primaryTumor'), 'dir')
    mkdir(strcat(outputPath, '/augmentedImages/primaryTumor'));
end

if ~exist(strcat(outputPath, '/augmentedImages/solidTissueNormal'), 'dir')
    mkdir(strcat(outputPath, '/augmentedImages/solidTissueNormal'));
end

primaryTumorAugmentedImages = dir(strcat(outputPath, '/primaryTumor'));
solidTissueNormalAugmentedImages = dir(strcat(outputPath, '/solidTissueNormal'));

for i = 1:numel(primaryTumorCleanImages)
    name = image_files(i).name;
    image = imread(fullfile(primaryTumorPath, image_files(i).name));
    
    tform1 = randomAffine2d(Rotation=[0 45]);
    image = imwarp(image,tform1);

    tform2 = randomAffine2d(XShear=[-45 45]);
    image = imwarp(image, tform2);

    tform3 = randomAffine2d(YShear=[-45 45]);
    image = imwarp(image, tform3);

    image = jitterColorHSV(image,'Contrast',[-1 1],'Hue',[-1 1],'Saturation',[-1 1],'Brightness',[-1 1]);

    sigma = 1+5*rand;
    image= imgaussfilt(image,sigma); 

    imwrite(image, fullfile(primaryTumorAugmentedImages , name, 'augId1'))
end

for i = 1:numel(solidTissueNormalCleanImages)
    name = image_files(i).name;
    image = imread(fullfile(solidTissueNormalPath, image_files(i).name));
    
    tform1 = randomAffine2d(Rotation=[0 45]);
    image = imwarp(image,tform1);

    tform2 = randomAffine2d(XShear=[-45 45]);
    image = imwarp(image, tform2);

    tform3 = randomAffine2d(YShear=[-45 45]);
    image = imwarp(image, tform3);

    image = jitterColorHSV(image,'Contrast',[-1 1],'Hue',[-1 1],'Saturation',[-1 1],'Brightness',[-1 1]);

    sigma = 1+5*rand;
    image= imgaussfilt(image,sigma); 

    imwrite(image, fullfile(solidTissueNormalAugmentedImages , name, 'augId1'))
end