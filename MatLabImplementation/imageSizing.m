function [] = imageSizing(folder)
% Create a new folder to save the reduced images
newFolder = 'resizedImages';
if ~exist(newFolder, 'dir')
    mkdir(newFolder);
end

% Get a list of all the PNG files in the folder
fileList = dir(fullfile(folder, '**/*.png'));

% Loop through each file and resize it
for i = 1:length(fileList)
    % Read in the image
    img = imread(fullfile(fileList(i).folder, fileList(i).name));
    img = imresize(img, [500 500]);

    % Determine the label of the image based on the subdirectory
    [~, label] = fileparts(fileList(i).folder);
    
    % Create a new subdirectory in the new folder with the label
    labelFolder = fullfile(newFolder, label);
    if ~exist(labelFolder, 'dir')
        mkdir(labelFolder);
    end
    
    % Save the resized image to the label subdirectory
    imwrite(img, fullfile(labelFolder, fileList(i).name));
end
