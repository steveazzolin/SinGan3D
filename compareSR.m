ssimResult = 0;
mssimResult = 0;
originalName = "crystalCluster2original";
toCompareName = "crystalCluster2";
originalPath = "C:\Users\marcu\Desktop\Uni\Tesi\MasterThesis\SinGAN-master-custom\Evaluation\SR\crystalCluster\" + originalName + ".json";
toComparePath = "C:\Users\marcu\Desktop\Uni\Tesi\MasterThesis\SinGAN-master-custom\Evaluation\SR\crystalCluster\" + toCompareName + ".json";
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