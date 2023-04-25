function [] = trainingAlexNet(datasetPath)

%% Create image datastore

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

imageSize = [500 500 3];
numTrainingFiles = 207;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');

%% Create augmented image datastore
n = numel(imds.Files);

imageMatrix = zeros(500, 500, 3, n);

for i = 1:n
    s = imds.Files(i);
    I = imread(char(s{1}));
    imageMatrix(:,:,:,i) = I;
end

XTrain = imageMatrix;
YTrain = imds.Labels;

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-180,180], ...
    'RandXTranslation',[-100 100], ...
    'RandYTranslation',[-100 100], ...
    'RandXReflection', true, ...
    'RandYReflection', true);

augimds = augmentedImageDatastore(imageSize, XTrain, YTrain, 'dataAugmentation', imageAugmenter);

idx = randperm(size(XTrain,4),50);
XValidation = XTrain(:,:,:,idx);
XTrain(:,:,:,idx) = [];
YValidation = YTrain(idx);
YTrain(idx) = [];

%% Create figures

figure
numImages = 521;
perm = randperm(numImages,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
    drawnow;
end
sgtitle("Images before augmentation");

figure
numImages = 521;
perm = randperm(numImages,20);
for i = 1:20
    subplot(4,5,i);
    imshow(augimds.Files{perm(i)});
    drawnow;
end
sgtitle("Images after augmentation");

%% define architecture of neural network
% determine weights of classes and number of classes
% class_weights = 1./countcats(testingLabels)';
% class_weights = class_weights/mean(class_weights);
% n_classes = length(categories(testingLabels));

% define architecture
layers = [
    imageInputLayer([500 500 3],"Name","data")
    convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4])
    reluLayer("Name","relu1")
    crossChannelNormalizationLayer(5,"Name","norm1","K",1)
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    groupedConvolution2dLayer([5 5],128,2,"Name","conv2","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    reluLayer("Name","relu2")
    crossChannelNormalizationLayer(5,"Name","norm2","K",1)
    maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],384,"Name","conv3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu3")
    groupedConvolution2dLayer([3 3],192,2,"Name","conv4","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu4")
    groupedConvolution2dLayer([3 3],128,2,"Name","conv5","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu5")
    maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
    fullyConnectedLayer(4096,"Name","fc6","BiasLearnRateFactor",2)
    reluLayer("Name","relu6")
    dropoutLayer(0.5,"Name","drop6")
    fullyConnectedLayer(4096,"Name","fc7","BiasLearnRateFactor",2)
    reluLayer("Name","relu7")
    dropoutLayer(0.5,"Name","drop7")
    fullyConnectedLayer(2,"Name","fc8","BiasLearnRateFactor",2)
    softmaxLayer("Name","prob")
    classificationLayer("Name","output")];

% set training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 32, ...
    'InitialLearnRate', 1e-4, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');

%% train the network

net_trained = trainNetwork(augimds, layers, options);

%% Calculate the accuracy of the network
% YPred = classify(net,imdsValidation);
% YValidation = imdsValidation.Labels;
% accuracy = mean(YPred == YValidation)
% save(trainedNetwork,'net_trained', 'accuracy', 'predictionLabels', 'predictions');