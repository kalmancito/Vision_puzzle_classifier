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
img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\testm8.jpg ');

I=rgb2gray(img);
% I=imadjust(I);
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
 
%     II=imcrop(rgb2hsv(img),[caract(i).BoundingBox]);
 II=imcrop(img,[caract(i).BoundingBox]);
test{i}=II;
II2=(1*II(:,:,1)-0*II(:,:,2)+0*II(:,:,3));%./(0.57*(II(:,:,1)+II(:,:,2)+II(:,:,3)));
II3=imbinarize((1-II2),0.85);
Ic(i)=sum(sum(II3))/caract(i).Area;

%         figure; imshow(II3)
%         title([num2str(Ic(i))])
        
    if(Ic(i))>0.1
        % estan detectadas las 4 de abajo aqui.
        j=j+1;
        kk2{j}=II;
%         figure; imshow(hsv2rgb(II))
%         title(['DETECTADA:' num2str(Ic(i))])
%         
        % PARA DETECTAR NARANJA
%         II=hsv2rgb(II);
        h=ones(5)/25;
%         II=imfilter(II,h);
        II4=(1*II(:,:,1)-0*II(:,:,2)+0*II(:,:,3));
        II5=imbinarize((1-II4),0.8);
       % PARA afinar regiones
        II5=bwareaopen(II5,1500);
        se=strel('disk',3);
        II5=imclose(II5,se);
        [L m(i)]=bwlabel(II5);
        Ic2(i)=sum(sum(II5))/caract(i).Area;
         
        figure; imshow(II5)
        title([num2str(Ic2(i)) 'regiones:' num2str(m(i))])       
        if  Ic2(i)>0.3&&m(i)<4
%             figure; imshow(II5)
%             title(['c3'])
        end

    end

end
j
% close all
% figure;imshow(kk2{t})
stop


%%



%     II=imcrop((img),[caract(i).BoundingBox]);
    % PARA DETECTAR GRIS, para las de abajo
%-----------------------------------------------------------------
%     [nrow ncol ~]=size(II)
    % naranja = [.0867 .4092 0.944];%HSV
%     naranja=[237 159 121];% RGB
%     for iii=1:nrow
%         for jjj=1:ncol
% %             i
% %             j
%             pixel(1:3) = double(II(iii,jjj,:));
%             ang_thres = 15; % degrees. You should change this to suit your needs
%             ang(iii,jjj) = acosd(dot(naranja/norm(naranja),pixel/norm(pixel)));
%             % mag_thres = 310; % You should change this to suit your needs
%             mag_thres =0.85;
%             mag(iii,jjj) = norm(pixel);
%             isnaranja(iii,jjj) = ang(iii,jjj) <= ang_thres & mag(iii,jjj) >= mag_thres; % Apply both thresholds
% 
%             if isnaranja
%                 II(iii,jjj,:)=[255 255 255];
%             end
%         end
%     end
%      Ic(i)=sum(sum(isnaranja))/(nrow*ncol);
%      figure;imshow(I)
%      title(num2str(tanto(ii)))
%-----------------------------------------------------------------
% II2=(2*II(:,:,1)-II(:,:,2)-II(:,:,3));