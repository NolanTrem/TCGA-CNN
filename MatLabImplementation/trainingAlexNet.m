%function [] = trainingAlexNet(folderPath)
%% 
datasetPath = "/Volumes/NolansDrive/TCGA-CNN/Breast/breastCancerImages";
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%%
% Define the target image size
imageSize = [227 227 3];

% Define the image augmenter to resize the images
auimds = augmentedImageDatastore(imageSize,imds);


%% define architecture of neural network
% determine weights of classes and number of classes
% class_weights = 1./countcats(testingLabels)';
% class_weights = class_weights/mean(class_weights);
% n_classes = length(categories(testingLabels));

% define architecture
layers = [
    imageInputLayer([227 227 3],"Name","data")
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
    fullyConnectedLayer(1000,"Name","fc8","BiasLearnRateFactor",2)
    softmaxLayer("Name","prob")
    classificationLayer("Name","output")];

%% set training options
options = trainingOptions('sgdm', ...
    'MaxEpochs',20,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% train the network
%  - the inputs are
%  -- "option" from section "set training options"

net_trained = trainNetwork(auimds, layers, options);

%% Calculate the accuracy of the network
% predictions = classify(net_trained, testingOutput);
% predictionLabels = transpose(testingLabels);
% accuracy = sum(predictions == predictionLabels)/numel(predictionLabels);
% 
% %% save the trained network
% save(trainedNetwork,'net_trained', 'accuracy', 'predictionLabels', 'predictions');