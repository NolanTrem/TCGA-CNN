function [] = trainingAlexNet(datasetPath)

datasetPath = '/Users/nolantremelling/Downloads/commonCancerDataset';
%% Create image datastore

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imageSize = [500 500 3];

%% Create augmented image datastore
% n = numel(imds.Files);
% 
% imageMatrix = zeros(500, 500, 3, n);
% 
% for i = 1:n
%     s = imds.Files(i);
%     I = imread(char(s{1}));
%     imageMatrix(:,:,:,i) = I;
% end
% 
% XTrain = imageMatrix;
% YTrain = imds.Labels;



% Separate the dataset into two classes
smallClass = subset(imds, find(imds.Labels == 'solidTissueNormal'));
largeClass = subset(imds, find(imds.Labels == 'primaryTumor'));

% Determine which class is smaller
if numel(smallClass.Files) < numel(largeClass.Files)
    smallerClass = smallClass;
    largerClass = largeClass;
else
    smallerClass = largeClass;
    largerClass = smallClass;
end

% Define the image augmentation parameters
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-45,45], ...
    'RandXTranslation',[-50 50], ...
    'RandYTranslation',[-50 50], ...
    'RandXReflection', true, ...
    'RandYReflection', true);

% Create an augmented image datastore for the smaller class
augmentedSmallerClass = augmentedImageDatastore(imageSize, smallerClass, 'DataAugmentation', imageAugmenter);

% Set the desired number of augmented images for the smaller class
desiredNumSmallerClassImages = numel(largerClass.Files);

% Calculate the number of times to repeat the smaller class images
numRepeats = ceil(desiredNumSmallerClassImages / numel(smallerClass.Files));

% Convert the smaller class datastore to a cell array and repeat it
smallClassCells = num2cell(smallerClass.Files);
smallClassCellsRepeated = repmat(smallClassCells, [numRepeats 1]);

% Convert the repeated smaller class cell array back to an image datastore
smallClassDS = imageDatastore(cat(1, smallClassCellsRepeated{:}));

% Shuffle the smaller class datastore
smallClassDS = shuffle(smallClassDS);

% Take the first `desiredNumSmallerClassImages` images from the shuffled smaller class datastore
smallClassDSFinal = subset(smallClassDS, 1:desiredNumSmallerClassImages);

% Create labels for the larger class datastore
largeClassLabels = repmat({largeClass.Labels}, [numRepeats 1]);
largeClassLabels = cat(1, largeClassLabels{:});

% Combine the smaller and larger class datastores
balancedDS = imageDatastore(cat(1, smallClassDSFinal.Files, largerClass.Files));
balancedDS.Labels = cat(1, smallClassDSFinal.Labels, largeClassLabels);

numTrainingFiles = floor(numel(balancedDS.Files)*0.8);
[imdsTrain,imdsTest] = splitEachLabel(balancedDS,numTrainingFiles,'randomize');

augmentedImdsTrain = augmentedImageDatastore(imageSize, imdsTrain, 'DataAugmentation', imageAugmenter);
augmentedImdsTest = augmentedImageDatastore(imageSize, imdsTest, 'DataAugmentation', imageAugmenter);
%%

% imageAugmenter = imageDataAugmenter( ...
%     'RandRotation',[-10,10], ...
%     'RandXTranslation',[-10 10], ...
%     'RandYTranslation',[-10 10], ...
%     'RandXReflection', true, ...
%     'RandYReflection', true);
% 
% augimds = augmentedImageDatastore(imageSize, XTrain, YTrain, 'dataAugmentation', imageAugmenter);
% 
% idx = randperm(size(XTrain,4),50);
% XValidation = XTrain(:,:,:,idx);
% XTrain(:,:,:,idx) = [];
% YValidation = YTrain(idx);
% YTrain(idx) = [];

%% Create figures

% figure
% numImages = 521;
% perm = randperm(numImages,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imds.Files{perm(i)});
%     drawnow;
% end
% sgtitle("Images before augmentation");
% 
% figure
% numImages = 521;
% perm = randperm(numImages,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(augimds.Files{perm(i)});
%     drawnow;
% end
% sgtitle("Images after augmentation");

%% define architecture of neural network
% determine weights of classes and number of classes
% class_weights = 1./countcats(testingLabels)';
% class_weights = class_weights/mean(class_weights);
% n_classes = length(categories(testingLabels));

% define architecture
layers = [
    imageInputLayer([500 500 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% set training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'L2Regularization', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 5, ...
    'InitialLearnRate', 1e-4, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% train the network

net_trained = trainNetwork(augmentedBalancedDS, layers, options);

%% Calculate the accuracy of the network
% YPred = classify(net,imdsValidation);
% YValidation = imdsValidation.Labels;
% accuracy = mean(YPred == YValidation)
% save(trainedNetwork,'net_trained', 'accuracy', 'predictionLabels', 'predictions');