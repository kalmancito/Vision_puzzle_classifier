close all

%%
%   webcamlist
% % cam = webcam(1)
%    preview(cam)
% % 
%  closePreview(cam)
% %%
% %  cam.AvailableResolutions
% cam.resolution='1280x1024'
% cam. ExposureMode='auto'
% % cam.Exposure=-8
%%
%    img = snapshot(cam);
img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\puzzle_vision_prueba.png');
imshow(img)

% subplot(1,3,1);imshow(img(:,:,1))
% subplot(1,3,2);imshow(img(:,:,2))
% subplot(1,3,3);imshow(img(:,:,3))




I=rgb2gray(img);

%%
% background = imopen(I,strel('disk',15));
% 
% figure
% surf(double(background(1:8:end,1:8:end))),zlim([0 255]);
% ax = gca;
% ax.YDir = 'reverse';
% 
% I2=I-background;
% imshow(I2)
% 
% I3 = imadjust(I2);
% imshow(I3);

bw = imbinarize(I);
imshow(bw)
%%
bw = bwareaopen(bw,300);
kernel = [1;1;1];
bw = imclose(bw, kernel);
imshow(bw)
%%
% bw = imfill(bw,'holes');
% imshow(bw)
%%

[L,num]=bwlabel(bw,8);
imshow(L)
title('binarized image')
% a=(L==1);imshow(a)

%%


CC = bwconncomp(bw);
LL = labelmatrix(CC);
figure
imshow(label2rgb(LL));
%%

caract=regionprops(L,'all');
% imshow(L==3)
% caract(3).Area
% caract(3).BoundingBox
% caract(3).ConvexHull
centroids = cat(1, caract.Centroid);

%%

STATS = regionprops(L, 'Image', 'SubarrayIdx');
imMask = STATS(2).Image;
subImage = L(STATS(2).SubarrayIdx{:});
imshow(subImage)
%%

imshow(bw)
hold on
plot(centroids(:,1),centroids(:,2), 'b*')
hold off
%%
% figure(555)

for i=2:num
% ROI_cH{i}=imcrop(img,[caract(i).ConvexHull]);

% mask=poly2mask(,147,168)
% ROI_c{i}=imcrop(img,[caract(i).BoundingBox]);
% ROI_mask{i}=caract(i).ConvexImage;
II=imcrop(I,[caract(i).BoundingBox]);
M=caract(i).ConvexImage;
% Inew{i}=ROI_c{i}.*ROI_mask{i};
Inew = II(2:end,2:end,:).*uint8(repmat(M,[1,1,3]));

% figure(i*100);imshow(Inew)
ROI_c{i}=Inew;
% ang=caract(i).Orientation

%-------------------------------------------

sides = caract(i).Extrema(4,:) - caract(i).Extrema(6,:); % Returns the sides of the square triangle that completes th


ang = rad2deg(atan(-sides(2)/sides(1))) 

%------------------------------------------
ROI_c_r{i}=imrotate(ROI_c{i},-ang,'bilinear');
% caract(i).Solidity
% caract(i).Centroid
% caract(i).Area
% figure
% subplot(2,round(num/2),i)
figure;imshow(ROI_c_r{i})
title(num2str(i))
end
%%
stop

%%

ROI_c=ROI_c_r;
%%
nt=8;
imshow(rgb2gray(ROI_c_r{nt}))
I=rgb2gray(ROI_c_r{nt});
% [~, threshold] = edge(I, 'canny');
% fudgeFactor = .5;
BWs = edge(I,'canny');
figure, imshow(BWs), title('binary gradient mask');
%%
for i=2:num-9
II=ROI_c{i};
% imshow(rgb2gray(II))
I=rgb2gray(II);
% [~, threshold] = edge(I, 'canny');
% fudgeFactor = .5;


% I=rangefilt(I);
entropy(I)

I= edge(I,'canny');
% I=imbinarize(BWs,'adaptive');
%
% se90 = strel('line', 3, 90);
% se0 = strel('line', 3, 0);
% BWsdil = imdilate(BWs);
% figure, imshow(BWsdil), title('dilated gradient mask');
% BWdfill = imfill(BWsdil, 'holes');
% figure, imshow(BWdfill);
% title('binary image with filled holes');
% BWnobord = imclearborder(BWdfill, 4);
% figure, imshow(BWdfill), title('cleared border image');
[H,theta,rho] = hough(I);
%
subplot(1,num,i)
imshow(I);
end


figure
imshow(imadjust(mat2gray(H)),[],...
       'XData',theta,...
       'YData',rho,...
       'InitialMagnification','fit');
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal
hold on
colormap(hot)

P = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));

x = theta(P(:,2));
y = rho(P(:,1));
plot(x,y,'s','color','black');

lines = houghlines(I,theta,rho,P,'FillGap',5,'MinLength',7);
figure, imshow(I), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end
% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','red');


%%
ROIp4=ROI_c{5};
ROIp7=ROI_c{6};
ROIp8=ROI_c{7};

test=zeros(1,num);
for i=4:num
  ROIp_bw{i}=imbinarize(rgb2gray(ROI_c{i}));
[Lp,nump]=bwlabel(ROIp_bw{i},4);  
test(i)=nump;
end

test
% %%
% cc = bwconncomp(bw, 8)
% 
% %%
% figure(111)
% labeled = labelmatrix(cc);
% whos labeled
% RGB_label = label2rgb(labeled, @spring, 'c', 'shuffle');
% imshow(RGB_label)
% % % % 
% % % % %%
% % % % 
% % % % object = false(size(bw));
% % % % object(cc.PixelIdxList{3}) = true;
% % % % imshow(object);
% % % % 
% % % % %%
% % % % ROI=I(cc.PixelIdxList{2});
% % % % imshow(ROI)
% %%
% BWoutline = bwperim(object);
% Segout = I;
% Segout(BWoutline) = 255;
% figure, imshow(Segout), title('outlined original image');


%%


c3=caract(5).Area
c3=caract(5).Solidity
figure; imshow(ROIp4)

ROIp4_bw=imbinarize(rgb2gray(ROIp4),'adaptive');
ROIp8_bw=imbinarize(rgb2gray(ROIp8),'adaptive');
%%
I=ROIp4_bw;
[~, threshold] = edge(I, 'sobel');
fudgeFactor = .5;
BWs = edge(I,'sobel', threshold * fudgeFactor);
figure, imshow(BWs), title('binary gradient mask');
%
se90 = strel('line', 3, 90);
se0 = strel('line', 3, 0);
BWsdil = imdilate(BWs, [se90 se0]);
figure, imshow(BWsdil), title('dilated gradient mask');
BWdfill = imfill(BWsdil, 'holes');
figure, imshow(BWdfill);
title('binary image with filled holes');
% BWnobord = imclearborder(BWdfill, 4);
figure, imshow(BWdfill), title('cleared border image');
% imshow(ROIp4_bw)
[L4,num4]=bwlabel(BWdfill,4);
kk=centroide(BWdfill)
%%
%%
% I=ROIp7_bw;
% [~, threshold] = edge(I, 'sobel');
% fudgeFactor = .5;
% BWs = edge(I,'sobel', threshold * fudgeFactor);
% figure, imshow(BWs), title('binary gradient mask');
% %
% se90 = strel('line', 3, 90);
% se0 = strel('line', 3, 0);
% BWsdil = imdilate(BWs, [se90 se0]);
% figure, imshow(BWsdil), title('dilated gradient mask');
% BWdfill = imfill(BWsdil, 'holes');
% figure, imshow(BWdfill);
% title('binary image with filled holes');
% % BWnobord = imclearborder(BWdfill, 4);
% figure, imshow(BWdfill), title('cleared border image');
% % imshow(ROIp4_bw)
% [L7,num7]=bwlabel(BWdfill,4);
% kk=centroide(BWdfill)

%%
%%
I=ROIp8_bw;
[~, threshold] = edge(I, 'sobel');
fudgeFactor = .5;
BWs = edge(I,'sobel', threshold * fudgeFactor);
figure, imshow(BWs), title('binary gradient mask');
%
se90 = strel('line', 3, 90);
se0 = strel('line', 3, 0);
BWsdil = imdilate(BWs, [se90 se0]);
figure, imshow(BWsdil), title('dilated gradient mask');
BWdfill = imfill(BWsdil, 'holes');
figure, imshow(BWdfill);
title('binary image with filled holes');
% BWnobord = imclearborder(BWdfill, 4);
figure, imshow(BWdfill), title('cleared border image');
% imshow(ROIp4_bw)
[L8,num8]=bwlabel(BWdfill,4);
kk=centroide(BWdfill)
imshow(ROI_c{7})
%%

c3=caract(5).Area;
c3=caract(5).Solidity;
% figure;imshow(ROIp7)

% ROIp7_bw=imbinarize(rgb2gray(ROIp7),'adaptive');imshow(ROIp7_bw)
% [L7,num7]=bwlabel(ROIp7_bw,4);
close all


