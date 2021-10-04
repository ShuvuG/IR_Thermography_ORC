%% 
% Shuvechchha Ghimire
% 18/03/2021
% Program for Empirical Evaluation of Heat Wasted in an Organic Rankine Cycle Using MATLAB Image Processing Toolbox and Machine Learning based OCR Models


% Further scope: 
% Geometric segmentation of different images 
% No consideration for environmental temperature 
% Overcompensation of images 
% Manual temperature and pixel value detection - could have solved it all 
% with the image segmenter app  
% The higher the HCF value, the more accuracy we will get. Changing HCF 
% value will mean more manual changes of Pixel value 
 
%% 
% Create an image datastore for the bulk FLIR images 
location=fullfile('/','MATLAB Drive','Thermal_Imaging','Thermal_images'); 
ds=imageDatastore({location}, "FileExtensions",{'.jpg'}); 
 
%Read image from datastore 
originalImage=readimage(ds,8); 
[rows columns numberOfColorBands] = size(originalImage); 
GrayImage=im2gray(originalImage); 
[rows_G columns_G numberOfColorBands_G] = size(GrayImage); 
imshowpair(originalImage,GrayImage,"montage"); 
 
%% 
% Main function calls 
 
% Isolate region of interest by removing unnecessary background 
Isolated_Image=Geometry_Isolation(originalImage); 
 
% Get temperature value from the image 
Temperature_Text= Temperature(originalImage) 
 
% Segment images into blocks  
% Potential values include Highest Common Factor of 1440 and 1080 
Pixel_Image=imseg(Isolated_Image,180, 1); 
 
 
%% 
% MANUAL INTERVENTIONS REQUIRED HERE! 
 
% Might need to adjust the temperature manually - quite sad! 
TT=[Temperature_Text(1) 86.7 Temperature_Text(2) 0] 
 
% Color map extraction from the regions of interest 
% Identify pixel containing the values of interests manually - quite sad! 
asd=mean2(Pixel_Image{4,2,1}); 
asd1=mean2(Pixel_Image{6,3,1}); 
asd2=mean2(Pixel_Image{6,4,1}); 
XY=[asd asd1 asd2 0] 
 
% Calculate temperature of the overall image 
temp=temperature_Calc(Pixel_Image,XY,TT); 
 
 
% Export data to excel - TO DO 
 
%% 
% Function to isolate background from irrelevant background components 
function Isolated_Image=Geometry_Isolation(colorImage) 
    % Isolate Geometry of Image# 
    Grayscale_Image=im2gray(colorImage); 
    Grayscale_Image34=im2gray(colorImage); 
    Grayscale_Image=im2double(Grayscale_Image); 
    Binarized_Image= imbinarize(Grayscale_Image); 
    Grayscale_Image2=imfill(Binarized_Image,4,"holes"); 
    Grayscale_Image2 = bwareaopen(Grayscale_Image2,400);  
    Grayscale_Image3 = activecontour(Grayscale_Image34,Grayscale_Image2,"Chan-Vese","ContractionBias",-.375); 
    Grayscale_Image3 = imfill(Grayscale_Image3,"holes"); 
    SE = strel("disk",20); 
    Grayscale_Image3 = imopen(Grayscale_Image3,SE); 
    r = colorImage(:,:,1); 
    g = colorImage(:,:,2); 
b = colorImage(:,:,3); 
r(~Grayscale_Image3) = 0; 
g(~Grayscale_Image3) = 0; 
b(~Grayscale_Image3) = 0; 
% Reconstruct the RGB image: 
Isolated_Image = cat(3,r,g,b); 
%imshow(Isolated_Image); 
end 
 
%% 
% Function to extract temperature value from the image 
% Recognise text using optical character recognition 
% Change temperature value to decimal 
function temperature=Temperature(colorImage) 
% Function to pre-process image to only show temperature data 
Grayscale_Image=im2gray(colorImage); 
Grayscale_Image=imadjust(Grayscale_Image); 
SE=strel("disk",3); 
data1=imtophat(Grayscale_Image,SE); 
data1=imfill(data1,"holes"); 
data=imcomplement(data1); 
sharpCoeff = [0 0 0;0 1 0;0 0 0]-fspecial('laplacian',0.5); 
data = imfilter(data,sharpCoeff,'symmetric'); 
% Detect MSER regions. 
[mserRegions, mserConnComp] = detectMSERFeatures(data, ...  
'RegionAreaRange',[5 450],'ThresholdDelta',3); 
% Use regionprops to measure MSER properties 
mserStats = regionprops(mserConnComp, 'BoundingBox', 'Eccentricity', ... 
'Solidity', 'Extent', 'Euler', 'Image'); 
% Compute the aspect ratio using bounding box data. 
bbox = vertcat(mserStats.BoundingBox); 
w = bbox(:,3); 
h = bbox(:,4); 
aspectRatio = w./h; 
% Threshold the data to determine which regions to remove. These thresholds 
% may need to be tuned for other images. 
filterIdx = aspectRatio' > 2;  
filterIdx = filterIdx | [mserStats.Eccentricity] > .95; 
filterIdx = filterIdx | [mserStats.Solidity] < .2; 
filterIdx = filterIdx | [mserStats.Extent] < 0.05 | [mserStats.Extent] > 0.95; 
filterIdx = filterIdx | [mserStats.EulerNumber] < -4; 
% Remove regions 
mserStats(filterIdx) = []; 
mserRegions(filterIdx) = []; 
% Get a binary image of the a region, and pad it to avoid boundary effects 
% during the stroke width computation. 
regionImage = mserStats(6).Image; 
regionImage = padarray(regionImage, [1 1]); 
% Compute the stroke width image. 
distanceImage = bwdist(~regionImage,'quasi-euclidean');  
skeletonImage = bwmorph(regionImage, 'thin', inf); 
strokeWidthImage = distanceImage; 
strokeWidthImage(~skeletonImage) = 0; 
% Compute the stroke width variation metric  
strokeWidthValues = distanceImage(skeletonImage);  
strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues); 
% Threshold the stroke width variation metric 
strokeWidthThreshold = 0.2; 
strokeWidthFilterIdx = strokeWidthMetric > strokeWidthThreshold; 
for j = 1:numel(mserStats)  
regionImage = mserStats(j).Image; 
regionImage = padarray(regionImage, [1 1], 0); 
distanceImage = bwdist(~regionImage); 
skeletonImage = bwmorph(regionImage, 'thin', inf); 
strokeWidthValues = distanceImage(skeletonImage); 
strokeWidthMetric = std(strokeWidthValues)/mean(strokeWidthValues); 
strokeWidthFilterIdx(j) = strokeWidthMetric > strokeWidthThreshold; 
end 
% Get bounding boxes for all the regions 
bboxes = vertcat(mserStats.BoundingBox); 
% Convert from the [x y width height] bounding box format to the [xmin ymin 
% xmax ymax] format for convenience. 
xmin = bboxes(:,1); 
ymin = bboxes(:,2); 
xmax = xmin + bboxes(:,3) + 1; 
ymax = ymin + bboxes(:,4) + 1; 
% Expand the bounding boxes by a small amount. 
expansionAmount = 0.02; 
xmin = (1-expansionAmount) * xmin; 
ymin = (1-expansionAmount) * ymin; 
xmax = (1+expansionAmount) * xmax; 
ymax = (1+expansionAmount) * ymax; 
% Clip the bounding boxes to be within the image bounds 
xmin = max(xmin, 1); 
ymin = max(ymin, 1); 
xmax = min(xmax, size(data,2)); 
ymax = min(ymax, size(data,1)); 
% Compute the overlap ratio 
expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1]; 
overlapRatio = bboxOverlapRatio(expandedBBoxes, expandedBBoxes); 
% Set the overlap ratio between a bounding box and itself to zero to 
% simplify the graph representation. 
n = size(overlapRatio,1);  
overlapRatio(1:n+1:n^2) = 0; 
% Create the graph 
g = graph(overlapRatio); 
% Find the connected text regions within the graph 
componentIndices = conncomp(g); 
% Merge the boxes based on the minimum and maximum dimensions. 
xmin = accumarray(componentIndices', xmin, [], @min); 
ymin = accumarray(componentIndices', ymin, [], @min); 
xmax = accumarray(componentIndices', xmax, [], @max); 
ymax = accumarray(componentIndices', ymax, [], @max); 
% Compose the merged bounding boxes using the [x y width height] format. 
textBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1]; 
% Remove bounding boxes that only contain one text region 
numRegionsInGroup = histcounts(componentIndices); 
textBBoxes(numRegionsInGroup == 1, :) = []; 
% Show the final text detection result. 
ITextRegion = insertShape(colorImage, 'Rectangle', textBBoxes,'LineWidth',3); 
%imshow(ITextRegion); 
% Extract text from image 
ocrtxt = ocr(data,textBBoxes); 
temperature1=[ocrtxt.Text]; 
wordsas = regexp(temperature1, '\n', 'split'); 
% manual definitions to adjust for machine learning inadequacies 
wordsas1=strrep(wordsas,'§','9'); 
word1=strrep(wordsas1,'C',' '); 
word12=strrep(word1,'’',''); 
word132=strrep(word12,'—c',' '); 
word132=strrep(word132,'1-c','.0'); 
word132=strrep(word132,'F',' '); 
word132=strrep(word132,'E','9'); 
word132=strrep(word132,' ',' '); 
word132=strrep(word132,'.oc',' '); 
word132=strrep(word132,'-c',' '); 
word132=strrep(word132,'?','8'); 
word132=strrep(word132,"£I.","8"); 
% 
temperature43=str2double(word132); 
temperature = temperature43(~isnan(temperature43)); 
end 
 
 
%% 
% Image segmentation to extract individual color intensities 
% If F=1, the function will create figures for 
% visualization. If F=0, no figures are created. 
% the function returns SEG, a cell array containing the 
% image segments. 
function seg = imseg(img,Lseg,F) 
if F 
% figure;imshow(img); axis on; title('Original Image') 
% figure; % open a figure to fill with image segments  
end 
L = size(img); 
max_row = floor(L(1)/Lseg); 
max_col = floor(L(2)/Lseg); 
seg = cell(max_row,max_col); 
r1 = 1; % current row index, initially 1 
for row = 1:max_row 
c1 = 1;  
for col = 1:max_col 
% find end rows/columns for segment 
r2 = r1+Lseg-1; 
c2 = c1+Lseg-1; 
% store segment in cell array 
seg(row,col) = {img(r1:r2,c1:c2,:)}; 
if F 
% plot segment 
subplot(max_row,max_col,(row-1)*max_col+col) 
imshow(cell2mat(seg(row,col)))  
end 
c1 = c2 +1; 
end 
r1 = r2 +1; 
end 
end 
 
%% 
% Calculate temperature of individual pixels 
function temp=temperature_Calc(Pixel_Image,BA,C) 
numPlotsR = size(Pixel_Image, 1); 
numPlotsC = size(Pixel_Image, 2); 
% Test variable 
% rg=Pixel_Image{1,2,1}; 
% imshow(rg); 
k=0; 
for i=1:numPlotsC 
for j=1:numPlotsR 
k=k+1; 
rgbBlock=Pixel_Image{j,i,1}; 
meanIntensity(k)= mean2(rgbBlock); 
end 
end 
B=meanIntensity'; % Transpose to get a n*1 matrix 
Y=interp1(BA,C, B, "linear", "extrap"); % corresponding temperatures 
scatter(B,Y); % scatter to see the temperature values in graph 
temp=Y; 
% Export to excel 
val = [B Y]; 
T = table(val); 
filename = 'surface_temperature.xlsx'; 
writetable(T,filename,'Sheet',1,'Range','A1')  
end 
 
