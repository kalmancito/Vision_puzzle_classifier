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
%    img = snapshot(cam);
% img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\avion\aviona1.png ');
img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\testm7.jpg ');
% img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\puzzle_vision_prueba2.png ');

% subplot(1,3,1);imshow(img(:,:,1))
% subplot(1,3,2);imshow(img(:,:,2))
% subplot(1,3,3);imshow(img(:,:,3))


% img=rgb2hsv(img);
imshow(img)

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

% caract=regionprops(L,'all');
% imshow(L==3)
% caract(3).Area
% caract(3).BoundingBox
% caract(3).ConvexHull


%%
j=0;
STATS = regionprops(L,'all');
for ii=1:length(STATS)
% STATS = regionprops(L, 'Image', 'SubarrayIdx');
        if (abs(STATS(ii).Area-(STATS(ii).Perimeter/4)^2)>1000)&&...
                STATS(ii).Area>100000
           %abs(STATS(ii).EquivDiameter-(STATS(ii).Perimeter/2))<26
        j=j+1;
        caract(j)=STATS(ii);
        imMask = STATS(ii).Image;
        subImage{j} = L(STATS(ii).SubarrayIdx{:});
%         figure; imshow(subImage{j})
        end
end
num=j
clear centroids
centroids = cat(1, caract.Centroid);
%%
imshow(bw)
hold on
plot(centroids(:,1),centroids(:,2), 'b*')
hold off
%% GIRAR EL CUADRADO
close all
% for kk=1:num
% % imshow(subImage{kk})
% % pause
% II=imcrop(rgb2gray(img),[caract(kk).BoundingBox]);
% %-------------------------------------------
% 
% sides = caract(kk).Extrema(4,:) - caract(kk).Extrema(6,:); % Returns the sides of the square triangle that completes th
% 
% 
% ang = rad2deg(atan2(-sides(2),sides(1))) ;
% 
% M=caract(kk).ConvexImage;
% Inew = II(2:end,2:end,:).*uint8(repmat(M,[1,1,1]));
% ROI_c{kk}=Inew;
% 
% %------------------------------------------
% ROI_c_r{kk}=imrotate(ROI_c{kk},0,'bilinear');
% ROI_bordes{kk}=edge(ROI_c_r{kk},'sobel');
% 
% imshow(ROI_c_r{kk})
% % pause
% end


% stop

















%%

%%
% figure(555)
% img=rgb2hsv(img);
close all
 clear II II2 II3
 j=0;
for i=1:num

II=imcrop(rgb2hsv(img),[caract(i).BoundingBox]);
% PARA DETECTAR GRIS, para las de abajo
II2=2*II(:,:,1)-II(:,:,2)-II(:,:,3);
II3=imbinarize((II2),0.6);
% PARA DETECTAR NARANJA
% II2=(2*II(:,:,1)-0.7*II(:,:,2)+0*II(:,:,3))./(0.57*(II(:,:,1)+II(:,:,2)+II(:,:,3)));
% PARA DETECTAR AZUL y naranja, para la piloto.
% II=hsv2rgb(II);
% II2=(2*II(:,:,1)-1.5*II(:,:,2)+0*II(:,:,3))./(0.57*(II(:,:,1)+II(:,:,2)+II(:,:,3)));
% II3=imbinarize(1-(II2),0.9);



Ic(i)=sum(sum(II3))/caract(i).Area;

% figure; imshow(II3)
% title(num2str(Ic(i)))
% if(Ic(i)<0.28) % detecta la piloto
%     title(['DETECTADA:' num2str(Ic(i))])
% end

% if(Ic(i)>0.7) % detecta cielo.
%     figure; imshow(II3)
%     title(['DETECTADA:' num2str(Ic(i))])
% end
% figure; imshow(II3)
% title(num2str(Ic(i)))

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

% ROI_c=ROI_c_r;
% save('masks','ROI_c')

%%

load binarymasks
load masks % mascaras en ROI_c
% originales de la foto en ROI_c_r
%%

kk=1;
for mascara=1
    for prueba=1:num
 
        masc=rgb2gray(ROI_c{mascara});      
[numrows numcols]=size(masc);
DD = imresize(rgb2gray(ROI_c_r{prueba}),[numrows numcols], 'bilinear') ;

% figure;imshow(BWs{prueba})
% H{prueba}=conv2(single(DD),single(masc));
H{prueba}=edge(masc-DD,'sobel')&(edge(masc,'sobel'));
figure;imshow(edge(masc-DD,'sobel')&(edge(masc,'sobel')))
% H=imfilter(DD,BWs2{mascara},'conv')

% figure;imshow(H{prueba})
res(mascara,prueba)=sum(sum(H{prueba}))/sum(sum(edge(masc,'sobel')));
% res(mascara,prueba)=max(H{prueba});
% res2(mascara)=sum(sum(masc));

    end
end
 close all
% res=1./res;
% size(res)
% res

% imshow(H{1})

% mesh(res)
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


