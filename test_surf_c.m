
close all
clear all
clc
warning off
%%
img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\testm2.jpg ');
% img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\puzzle_vision_prueba.png ');

I=rgb2gray(img);
% I=imadjust(I);
imshow(I)

background = imopen(I,strel('disk',15));
figure
surf(double(background(1:8:end,1:8:end))),zlim([0 255]);
ax = gca;
ax.YDir = 'reverse';
bw = imbinarize(I);

imshow(bw)

bw = bwareaopen(bw,300);
kernel = [1;1;1];
bw = imclose(bw, kernel);
imshow(bw)

bw = imfill(bw,'holes');
imshow(bw)

[L,num]=bwlabel(bw,8);
imshow(L)
title('binarized image')

CC = bwconncomp(bw);
LL = labelmatrix(CC);
figure
imshow(label2rgb(LL));



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

%%
% centroids = cat(1, caract.Centroid);
%
% imshow(bw)
% hold on
% plot(centroids(:,1),centroids(:,2), 'b*')
% hold off
%detectar
close all
clear II II2 II3

 j=0;
for i=1:num
 
%     II=imcrop(rgb2hsv(img),[caract(i).BoundingBox]);
 II=imcrop(img,[caract(i).BoundingBox]);
test2{i}=II;
end
%%
load testsurf


text_str = cell(num,1);
for ii=1:num
    text_str{ii}='-';
end
% load testsurf2
%  load testsurf3
%  test=test3;
% load testsurf4
%  test=test4;
% load testsurf5
%  test=test5;
% test2=test4;
for jj=1:num
    
    for ii=1:num
    mascara = rgb2gray(test{jj});
    mascara=imadjust(mascara);
    [nr nc]=size(mascara);
    imagentest = imresize(rgb2gray(test2{ii}),[nr,nc]);
     imagentest=imadjust(imagentest);

    points1 = detectSURFFeatures(mascara);
    points2 = detectSURFFeatures(imagentest);

    [f1,vpts1] = extractFeatures(mascara,points1);
    [f2,vpts2] = extractFeatures(imagentest,points2);

    indexPairs = matchFeatures(f1,f2);%, 'MaxRatio' ,0.5);%0.6 default
    matchedPoints1 = vpts1(indexPairs(:,1));
    matchedPoints2 = vpts2(indexPairs(:,2));

    res(ii)=max(size(indexPairs));
    end
%         figure; showMatchedFeatures(mascara,imagentest,matchedPoints1,matchedPoints2);
%         legend('matched points 1','matched points 2');
    prueba{jj}=find(res==max(res));
    if(length(prueba{jj})>1)
        kk=13
        prueba{jj}='Null';
        disp('malo')
        
    else
    kk=jj;
   
%  figure;imshow(test{jj})
%  figure;imshow(test2{prueba{jj}})
%     clear res
  
 switch kk
   case 1
      title('b1')
      text_str{prueba{jj}}='b1';
   case 2
      title('a1')
      text_str{prueba{jj}}='a1';
   case 3
      title('c1')
      text_str{prueba{jj}}='c1';
   case 4
      title('a2')
      text_str{prueba{jj}}='a2';
   case 5
      title('b2')
      text_str{prueba{jj}}='b2';
   case 6
      title('c2')
      text_str{prueba{jj}}='c2';
   case 7
      title('a3')
      text_str{prueba{jj}}='a3';
   case 8
      title('b3')
      text_str{prueba{jj}}='b3';
    case 9
      title('c3')
      text_str{prueba{jj}}='c3';
  case 10
      title('a4')
      text_str{prueba{jj}}='a4';
  case 11
      title('b4')
      text_str{prueba{jj}}='b4';
  case 12
      title('c4')
      text_str{prueba{jj}}='c4';

     otherwise
                disp('other value')
   end
 
%     pause(3)
%     close all
end 


end
prueba
close all
%%
close all
% text_str = cell(3,1);
conf_val = [85.212 98.76 78.342];
% position = [23 373;35 185;77 107];


for ii=1:num
%    text_str{ii} = ['A1'];
   position(ii,1:2)=caract(ii).Extrema(1,:);
end
%
RGB = insertText(img,position,text_str,'FontSize',102,'BoxOpacity',0.4,'TextColor','black');
imshow(RGB);    
hold on
for i=1:numel(caract)  
   rectangle('Position', caract(i).BoundingBox,'edgecolor','y')
   
end
% imshow(RGB);   