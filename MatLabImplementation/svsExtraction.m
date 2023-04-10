function [] = svsExtraction(manifestPath, folderPath, cancerType)

%% Data initialization
filePattern = fullfile(folderPath, "*.svs");
theFiles= dir(filePattern);
manifestFile = readtable(manifestPath, Delimiter='\t', ReadVariableNames=true);

columnVector = cell(length(theFiles), 3);

%% Image extraction
for i = 1 : length(theFiles)
    fullFileName = fullfile(theFiles(i).folder, theFiles(i).name);
    
    io = imread(fullFileName, 'Index', 3);
    
    fileName = theFiles(i).name;

    columnVector{i,1} = fileName;
    columnVector{i,2} = io;
    
    %% Labeling
    columnVector{i,3} = string(table2cell(manifestFile(strcmp(manifestFile{:,2}, fileName), 6)));

    % columnVector{i,3} = 'done';
end

assignin('base', cancerType + 'ColumnVector', olumnVector);

end
