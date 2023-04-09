function [] = svsExtraction(folderPath, cancerType)

filePattern = fullfile(folderPath, "*.svs");
theFiles= dir(filePattern);

columnVector = cell(length(theFiles), 3);

for i = 1 : length(theFiles)
    fullFileName = fullfile(theFiles(i).folder, theFiles(i).name);
    
    io = imread(fullFileName, 'Index', 3);
    
    columnVector{i,1} = theFiles(i).name;
    columnVector{i,2} = io;
    
    % TODO: This needs to parse each line of the manifest file and return
    % the last entry, which is the label for the data.
    % https://www.mathworks.com/matlabcentral/answers/497882-parsing-text-files-for-beginners
    % columnVector{i,3} = 'done';
end

assignin('base', cancerType + 'columnVector', columnVector);

end
