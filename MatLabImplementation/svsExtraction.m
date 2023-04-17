function [] = svsExtraction(manifestPath, folderPath, cancerType)

%% Data initialization
filePattern = fullfile(folderPath, "*.svs");
theFiles= dir(filePattern);
manifestFile = readtable(manifestPath, Delimiter='\t', ReadVariableNames=true);

columnVector = cell(length(theFiles), 2);

%% Image extraction
for i = 1 : length(theFiles)
    fullFileName = fullfile(theFiles(i).folder, theFiles(i).name);
    fileName = theFiles(i).name;
    savedAs = strcat("/Volumes/NolansDrive/TCGA-CNN/Breast/breastCancerImages/", extractBefore(fileName, length(fileName)-3), ".png");
    
    io = imread(fullFileName, 'Index', 3);
    imwrite(io, savedAs);

    columnVector {i,1} = fileName;
    %BreastColumnVector {i,1} = fileName;
    %BreastColumnVector {i,2} = io;
    
    %% Labeling
    columnVector {i,2} = string(table2cell(manifestFile(strcmp(manifestFile{:,2}, fileName), 6)));
    disp(i)
    % columnVector{i,3} = 'done';
end

datasetName = cancerType + 'Labels';
assignin('base', datasetName, columnVector);
%outputName = strcat("/Volumes/NolansDrive/TCGA-CNN/", datasetName, ".mat");
%save(outputName, "columnVector");

end
