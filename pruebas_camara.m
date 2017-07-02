close all
clear all
clc
 %%
%   webcamlist
% cam = webcam(1)
%    preview(cam)
% % 
% %  closePreview(cam)
% %%
% %  cam.AvailableResolutions
% cam.resolution='1280x1024'
% cam. ExposureMode='auto'
% % cam.Exposure=-8
%%
%    img = snapshot(cam);
% img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\avion\aviona1.png ');
img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\testm2.jpg ');
% img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\puzzle_vision_prueba.png ');

% subplot(1,3,1);imshow(img(:,:,1))
% subplot(1,3,2);imshow(img(:,:,2))
% subplot(1,3,3);imshow(img(:,:,3))


% img=rgb2hsv(img);
imshow(img)

I=rgb2gray(img);

%%
background = imopen(I,strel('disk',15));

figure
surf(double(background(1:8:end,1:8:end))),zlim([0 255]);
ax = gca;
ax.YDir = 'reverse';

bw = imbinarize(I,'adaptive');
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
j=0;
STATS = regionprops(L,'all');
for ii=1:length(STATS)
% STATS = regionprops(L, 'Image', 'SubarrayIdx');
        if (abs(STATS(ii).ConvexArea-(STATS(ii).Perimeter/4)^2)>7500)&&...
                STATS(ii).Area>100000
           %abs(STATS(ii).EquivDiameter-(STATS(ii).Perimeter/2))<26
        j=j+1;
        caract(j)=STATS(ii);
        imMask = STATS(ii).Image;
        subImage{j} = L(STATS(ii).SubarrayIdx{:});
        figure; imshow(subImage{j})
        end
end
num=j;
%% GIRAR EL CUADRADO
close all
for kk=1:num
% imshow(subImage{kk})
% pause
II=imcrop(rgb2gray(img),[caract(kk).BoundingBox]);
%-------------------------------------------

sides = caract(kk).Extrema(4,:) - caract(kk).Extrema(6,:); % Returns the sides of the square triangle that completes th


ang = rad2deg(atan2(-sides(2),sides(1))) ;

M=caract(kk).ConvexImage;
Inew = II(2:end,2:end,:).*uint8(repmat(M,[1,1,1]));
ROI_c{kk}=Inew;

%------------------------------------------
ROI_c_r{kk}=imrotate(ROI_c{kk},0,'bilinear');
ROI_bordes{kk}=edge(ROI_c_r{kk},'sobel');

imshow(ROI_c_r{kk})
pause
end


stop

















%%

imshow(bw)
hold on
plot(centroids(:,1),centroids(:,2), 'b*')
hold off
%%
% figure(555)
% img=rgb2hsv(img);
for i=1:num
% ROI_cH{i}=imcrop(img,[caract(i).ConvexHull]);

% mask=poly2mask(,147,168)
% ROI_c{i}=imcrop(img,[caract(i).BoundingBox]);
% ROI_mask{i}=caract(i).ConvexImage;
II=imcrop(rgb2gray(img),[caract(i).BoundingBox]);
corners{i-1}=detectSURFFeatures(II,'MetricThreshold',500);

imshow(II); hold on;
 plot(corners{i-1}.selectStrongest(5));


% II=rgb2hsv(II);
M=caract(i).ConvexImage;
% Inew = II(2:end,2:end,:).*double(repmat(M,[1,1,3])); %para HSV
Inew = II(2:end,2:end,:).*uint8(repmat(M,[1,1,3]));

% figure(i*100);imshow(Inew)
ROI_c{i-1}=Inew;
% ang=caract(i).Orientation

%-------------------------------------------

sides = caract(i).Extrema(4,:) - caract(i).Extrema(6,:); % Returns the sides of the square triangle that completes th


ang = rad2deg(atan(-sides(2)/sides(1))) ;

%------------------------------------------
ROI_c_r{i-1}=imrotate(ROI_c{i-1},-ang,'bilinear');
% caract(i).Solidity
% caract(i).Centroid
% caract(i).Area
% figure
% subplot(2,round(num/2),i)
figure;imshow(ROI_c_r{i-1})
title(num2str(i-1))
end
%%
% 
% for ii=2:num
% G=ROI_c_r{ii};
% [nrow ncol ~]=size(G)
% naranja = [.0867 .4092 0.944];%HSV
% % naranja=[237 159 121];% RGB
% for i=1:nrow
%     for j=1:ncol
% 
% pixel(1:3) = double(G(i,j,:));
% ang_thres = 15; % degrees. You should change this to suit your needs
% ang(i,j) = acosd(dot(naranja/norm(naranja),pixel/norm(pixel)));
% % mag_thres = 310; % You should change this to suit your needs
% mag_thres =0.85;
% mag(i,j) = norm(pixel);
% isnaranja(i,j) = ang(i,j) <= ang_thres & mag(i,j) >= mag_thres; % Apply both thresholds
%    
%     
%     end
% end
%  tanto(ii)=sum(sum(isnaranja))/(nrow*ncol);
%  
%  
%  figure;imshow(G)
%  title(num2str(tanto(ii)))
% end
%%


%%

ROI_c=ROI_c_r;
%%
 load binarymasks
 BWs2=BWs;
 clear BWs
for nt=2:num;
% imshow(rgb2gray(ROI_c_r{nt}))
I=rgb2gray(ROI_c_r{nt-1});
% [~, threshold] = edge(I, 'canny');
% fudgeFactor = .5;
BWs{nt-1} = edge(I,'canny');
figure, imshow(BWs{nt-1}), title('binary gradient mask');
end

%%
kk=1;
for mascara=1:num-2 %mascara=1:num-2
    for prueba=1:num-2
 
        masc=BWs2{mascara};      
[numrows numcols]=size(masc);

DD = imresize(BWs{prueba},[numrows numcols]) ;

H{prueba}=DD~=BWs2{mascara};
% H=imfilter(DD,BWs2{mascara},'conv')

% figure;imshow(H{prueba})
res(mascara,prueba)=sum(sum(H{prueba}))/sum(sum(BWs2{mascara}));
res2(mascara)=sum(sum(BWs2{mascara}));

    end
end
 close all
res=1./res;
size(res)

%  mesh(res)
imshow(H{1})


stop



%%
for i=6
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

close 


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

close all


