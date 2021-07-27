originalNameArray = ["abstractObject", "city", "crystal", "crystalCluster", "cubes", "rocks", "simpleForest", "sinWaves", "spyrals", "trees"];
ssimResultMatrix = zeros(length(originalNameArray), length(originalNameArray));
mssimResultMatrix = zeros(length(originalNameArray), length(originalNameArray));
for i=1:length(originalNameArray)-1
    originalName = originalNameArray(i);
    for j=i+1:length(originalNameArray)
        toCompareName = originalNameArray(j);
        originalFolderName = upper(extractBefore(originalName,2)) + extractAfter(originalName,1);
        originalPath = "C:\Users\Dva\Documents\Uni\Tesi\MasterThesis\SinGAN-master-custom\Evaluation\" + originalFolderName + "\" + originalName + ".json";
        toCompareFolderName = upper(extractBefore(toCompareName,2)) + extractAfter(toCompareName,1);
        toComparePath = "C:\Users\Dva\Documents\Uni\Tesi\MasterThesis\SinGAN-master-custom\Evaluation\" + toCompareFolderName + "\" + toCompareName + ".json";
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
        ssimResultMatrix(i,j) = scoressim;
        mssimResultMatrix(i,j) = scoremultissim;
    end
end