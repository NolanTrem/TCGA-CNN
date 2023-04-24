function [] = svsExtraction(manifestPath, folderPath, cancerType)

%% Data initialization
filePattern = fullfile(folderPath, "*.svs");
theFiles= dir(filePattern);
manifestFile = readtable(manifestPath, Delimiter='\t', ReadVariableNames=true);

imageDirectory = strcat(folderPath, "/", cancerType, "Cancerimages");
primaryTumorSubdirectory = strcat(imageDirectory, "/primaryTumor");
normalTissueSubdirectory = strcat(imageDirectory, "/solidTissueNormal");

mkdir(imageDirectory)
mkdir(primaryTumorSubdirectory);
mkdir(normalTissueSubdirectory);

%% Image extraction
for i = 1 : length(theFiles)
    fullFileName = fullfile(theFiles(i).folder, theFiles(i).name);
    fileName = theFiles(i).name;

    disp(string(table2cell(manifestFile(strcmp(manifestFile{:,2}, fileName), 6))))
    
    if strcmp(string(table2cell(manifestFile(strcmp(manifestFile{:,2}, fileName), 6))), "Primary Tumor")
        savedAs = strcat(primaryTumorSubdirectory, "/", extractBefore(fileName, length(fileName)-3), ".png");
        io = imread(fullFileName, 'Index', 3);
        imwrite(io, savedAs);
    elseif strcmp(string(table2cell(manifestFile(strcmp(manifestFile{:,2}, fileName), 6))), "Solid Tissue Normal")
        savedAs = strcat(normalTissueSubdirectory, "/", extractBefore(fileName, length(fileName)-3), ".png");
        io = imread(fullFileName, 'Index', 3);
        imwrite(io, savedAs);
    else
        disp('Error in writing file, check manifest.')
    end

end

end
