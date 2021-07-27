maxSamples = 50;
originalToEvaluateIndex = 10;
originalNameArray = ["abstractObject", "city", "crystal", "crystalCluster", "cubes", "rocks", "simpleForest", "sinWaves", "spyrals", "trees"];
originalName = originalNameArray(originalToEvaluateIndex);
originalFolderName = upper(extractBefore(originalName,2)) + extractAfter(originalName,1);
originalPath = "C:\Users\Dva\Documents\Uni\Tesi\MasterThesis\SinGAN-master-custom\Evaluation\" + originalFolderName + "\" + originalName + ".json";
ssimResultMatrix = zeros(length(originalNameArray), maxSamples);
mssimResultMatrix = zeros(length(originalNameArray), maxSamples);
for i=1:length(originalNameArray)
    sampleCounter = 0;
    for j=1:maxSamples
        toCompareName = num2str(sampleCounter);
        toCompareFolderName = upper(extractBefore(originalNameArray(i),2)) + extractAfter(originalNameArray(i),1);
        toComparePath = "C:\Users\Dva\Documents\Uni\Tesi\MasterThesis\SinGAN-master-custom\Evaluation\" + toCompareFolderName + "\Samples\" + toCompareName + ".json";
        data1 = jsondecode(fileread(originalPath));
        data2 = jsondecode(fileread(toComparePath));
        voxelArray1 = data1.voxels;
        voxelArray2 = data2.voxels;
        dataOriginal = zeros(data1.dimension(1).width, data1.dimension(1).height, data1.dimension(1).depth);
        dataSample = zeros(data2.dimension(1).width, data2.dimension(1).height, data2.dimension(1).depth);
        for k=1:length(voxelArray1)
            dataOriginal((voxelArray1(k).x+1),(voxelArray1(k).y+1),(voxelArray1(k).z+1)) = voxelArray1(k).value;
        end
        for k=1:length(voxelArray2)
            dataSample((voxelArray2(k).x+1),(voxelArray2(k).y+1),(voxelArray2(k).z+1)) = voxelArray2(k).value;
        end

        scoressim = ssim(dataSample, dataOriginal);
        scoremultissim = multissim3(dataSample, dataOriginal);
        ssimResultMatrix(i, j) = scoressim;
        mssimResultMatrix(i, j) = scoremultissim;
        sampleCounter = sampleCounter + 1;
    end 
end