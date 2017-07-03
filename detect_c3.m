close all
clear all
clc
warning off
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
img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\testm7.jpg ');

I=rgb2gray(img);
I=imadjust(I);
imshow(I)
%%
background = imopen(I,strel('disk',15));
figure
surf(double(background(1:8:end,1:8:end))),zlim([0 255]);
ax = gca;
ax.YDir = 'reverse';
bw = imbinarize(I);

imshow(bw)
%%
bw = bwareaopen(bw,300);
kernel = [1;1;1];
bw = imclose(bw, kernel);
imshow(bw)
%%
bw = imfill(bw,'holes');
imshow(bw)
%%
[L,num]=bwlabel(bw,8);
imshow(L)
title('binarized image')
%%
CC = bwconncomp(bw);
LL = labelmatrix(CC);
figure
imshow(label2rgb(LL));
%%

%%
j=0;
STATS = regionprops(L,'all');
for ii=1:length(STATS)

        if (abs(STATS(ii).Area-(STATS(ii).Perimeter/4)^2)>1000)&&...
            STATS(ii).Area>100000
                j=j+1;
                caract(j)=STATS(ii);
                imMask = STATS(ii).Image;
                subImage{j} = L(STATS(ii).SubarrayIdx{:});
        end
end
num=j
clear centroids
% centroids = cat(1, caract.Centroid);
%%
% imshow(bw)
% hold on
% plot(centroids(:,1),centroids(:,2), 'b*')
% hold off
%% detectar
close all
clear II II2 II3

 j=0;
for i=1:num

II=imcrop(rgb2hsv(img),[caract(i).BoundingBox]);
% PARA DETECTAR GRIS, para las de abajo
II2=2*II(:,:,1)-II(:,:,2)-II(:,:,3);
II3=imbinarize((II2),0.6);
Ic(i)=sum(sum(II3))/caract(i).Area;

    if(Ic(i))>0.1
        j=j+1;
        kk2{j}=II;
        figure; imshow(II)
        title(['DETECTADA:' num2str(Ic(i))])
        %------------------------------------------------
        %una vez detectadas las 4 de abajo, el siguiente paso
        %es separarlas

        % PARA C3:
        II=hsv2rgb(II);
        II4=(2*II(:,:,1)-0.7*II(:,:,2)+0*II(:,:,3))./(0.57*(II(:,:,1)+II(:,:,2)+II(:,:,3)));
        II5=imbinarize(II4,0.95);
        Ic2(j)=sum(sum(II5))/caract(i).Area;
        %         figure; imshow(II5)
        %         title(num2str(Ic2(j)))
        %------------------------------------------------

        % figure; imshow(II)
        end
        
%  plot(corners{i}.selectStrongest(10));


% II=rgb2hsv(II);
M=caract(i).ConvexImage;
% Inew = II(2:end,2:end,:).*double(repmat(M,[1,1,3])); %para HSV
% Inew = II(2:end,2:end,:).*uint8(repmat(M,[1,1,3]));

% figure(i*100);imshow(Inew)
   

%-------------------------------------------
%     % ROI_c{i}=Inew;    
% 
%     sides = caract(i).Extrema(4,:) - caract(i).Extrema(6,:); % Returns the sides of the square triangle that completes th
% 
% 
% ang = rad2deg(atan(-sides(2)/sides(1))) ;

%------------------------------------------
% ROI_c_r{i}=imrotate(ROI_c{i},0,'bilinear');
% caract(i).Solidity
% caract(i).Centroid
% caract(i).Area
% figure
% subplot(2,round(num/2),i)
% figure;imshow(ROI_c_r{i})
% title(num2str(i))
end
j

% t=find(Ic2==max(Ic2)); %detect dentro de las de abajo la del ala
                          % con mas naranja
% close all
% figure;imshow(kk2{t})
stop