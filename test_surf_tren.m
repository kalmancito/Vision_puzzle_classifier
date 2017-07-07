
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
%5
for uu=9%uu=59:62 %uu=36:57%uu=11:35%uu=1:13%uu=21:35%
% img=imread(['C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\tren_sem (' num2str(uu) ').jpg']);
% img=imread(['C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\tren_ab (' num2str(uu) ').jpg']);
img=imread(['C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\tren_ex (' num2str(uu) ').jpg']);

% img=imread('C:\Users\Miguel\Desktop\MUAR\1_sem\vision\vision\puzzle_vision_prueba.png ');
% figure;imshow(img)
I=rgb2gray(img);
% I=histeq(I);
% imshow(I)
%%
% bw = imbinarize(I,'adaptive');
% bw = imbinarize(I,'adaptive','ForegroundPolarity','bright','Sensitivity',0.4);
bw = imbinarize(I);
bw = imclose(bw, strel('square',25));
bw = imfill(bw,4,'holes');
% imshow(bw)
%%
bw = bwareaopen(bw,60000,8);
bw = imfill(bw,'holes');
% imshow(bw)
%%
% kernel = [1;1;1];
 bw = imerode(bw,strel('square',25));

% imshow(bw)
%%

bw = imfill(bw,'holes');
% imshow(bw)
%%
 
%    bw = imerode(bw,strel('disk',15));
% bw=edge(bw,'sobel');
% bw = imfill(bw,[3 3],4);
% imshow(bw)

%%

close all
[L,num]=bwlabel(bw,4);
imshow(L)
title('binarized image')

CC = bwconncomp(bw,4);
LL = labelmatrix(CC);
% figure;imshow(label2rgb(LL));



j=0;
STATS = regionprops(L,'all');
for ii=1:length(STATS)

        if ((abs(STATS(ii).Area-(STATS(ii).Perimeter/4)^2))/STATS(ii).Area<50)&&...
            STATS(ii).Area>90000&&...
            abs(STATS(ii).MajorAxisLength-STATS(ii).MinorAxisLength)<400&...
            STATS(ii).Solidity>0.6%
            
                j=j+1;
                caract(j)=STATS(ii);
                imMask = STATS(ii).Image;
                subImage{j} = L(STATS(ii).SubarrayIdx{:});
%                 figure;imshow(subImage{j});
                I=subImage{j};
%                 I = imerode(I,strel('square',35));
%                 figure;imshow(I);
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
% load testsurf
load masctren

text_str = cell(num,1);

for ii=1:num
    text_str{ii}='-';
end
labe=text_str;

% load testsurf2
%  load testsurf3
%  test=test3;
% load testsurf4
%  test=test4;
% load testsurf5
%  test=test5;
% test2=test4;
for jj=1:num
    
    for ii=1:12
    mascara = rgb2gray(test{ii});
    mascara=imadjust(mascara);
    [nr nc]=size(mascara);
    imagentest = imresize(rgb2gray(test2{jj}),[nr,nc]);
    imagentest=imadjust(imagentest);

    points1 = detectSURFFeatures(mascara);
    points2 = detectSURFFeatures(imagentest);
 

    [f1,vpts1] = extractFeatures(mascara,points1);
    [f2,vpts2] = extractFeatures(imagentest,points2);

    indexPairs = matchFeatures(f1,f2,'unique',true);%, 'MaxRatio' ,0.5);%0.6 default
    matchedPoints1 = vpts1(indexPairs(:,1));
    matchedPoints2 = vpts2(indexPairs(:,2));

    res(ii)=length(matchedPoints2);
%             figure; showMatchedFeatures(mascara,imagentest,matchedPoints1,matchedPoints2);
%         legend('matched points 1','matched points 2');
    end

    prueba{jj}=find(res==max(res));
    if(length(prueba{jj})>1)
        kk=13;
        prueba{jj}='Null';
        disp('warning: two match')
        
    else
    kk=prueba{jj};
   
%  figure;imshow(test{jj})
%  figure;imshow(test2{prueba{jj}})
%     clear res
  
 switch kk
   case 1
      labe{jj}=('c1');
      descriptores(jj)=max(res);
      text_str{jj}=['c1:' num2str(max(res))];
   case 2
      labe{jj}=('a1');
      descriptores(jj)=max(res);
      text_str{jj}=['a1:' num2str(max(res))];
   case 3
      labe{jj}=('b1');
      descriptores(jj)=max(res);
      text_str{jj}=['b1:' num2str(max(res))];
   case 4
      labe{jj}=('c2');
      descriptores(jj)=max(res);
      text_str{jj}=['c2:' num2str(max(res))];
   case 5
      labe{jj}=('a2');
      descriptores(jj)=max(res);
      text_str{jj}=['a2:' num2str(max(res))];
   case 6
      labe{jj}=('b2');
      descriptores(jj)=max(res);
      text_str{jj}=['b2:' num2str(max(res))];
   case 7
      labe{jj}=('c3');
      descriptores(jj)=max(res);
      text_str{jj}=['c3:' num2str(max(res))];
   case 8
      labe{jj}=('a3');
      descriptores(jj)=max(res);
      text_str{jj}=['a3:' num2str(max(res))];
    case 9
       labe{jj}=('b3');
      descriptores(jj)=max(res);
      text_str{jj}=['b3:' num2str(max(res))];
  case 10
       labe{jj}=('a4');
      descriptores(jj)=max(res);
      text_str{jj}=['a4:' num2str(max(res))];
  case 11
      labe{jj}=('b4');
      descriptores(jj)=max(res);
      text_str{jj}=['b4:' num2str(max(res))];
  case 12
      labe{jj}=('c4');
      descriptores(jj)=max(res);
      text_str{jj}=['c4:' num2str(max(res))];

     otherwise
                disp('other value')
   end
 
%     pause(3)
%     close all
end 


end
% prueba
% res
close all

%%
    if num==12
        labe_bueno={'a3','c2','b1','c4','a1','b2','a2','c1','b4','a4','c3','b3','-'};

        % comprobacion extra repeticiones
        % buscar vacio y buscar repeticion;
        % repeticion solucionar con mas alto;
        % vacio solucionar con sobrante.
         [~, ind]=unique(labe);
         duplicate_ind = setdiff(1:length(labe), ind);
         if ~isempty(duplicate_ind)
                  
             duplicate_value = labe{duplicate_ind};
             pos_rep= find(strcmp(labe, duplicate_value));
             pos_min=find(min(descriptores(pos_rep))==descriptores);
             if length(find(strcmp(labe, '-')))<2&&length(pos_min)==1
                labe{pos_min}='-';
                % %
                if length(find(strcmp(labe, '-')))==1
                   labe{pos_min}=cell2mat(setdiff(labe_bueno,labe));
                   text_str{pos_min} =(labe{pos_min});
                end
             end
         end
         
         % falta solo uno por identificar
          if length(find(strcmp(labe, '-')))==1
              indx=find(strcmp(labe, '-'));
              labe{indx}=cell2mat(setdiff(labe_bueno,labe));
              text_str{indx} =labe{indx};
          end

        % labe
    end
%%
%%
close all

for ii=1:num
%    text_str{ii} = ['A1'];
   position(ii,1:2)=caract(ii).Extrema(1,:);
end
%
RGB = insertText(img,position,labe,'FontSize',102,'BoxOpacity',0.4,'TextColor','black');
h=figure(555);imshow(RGB); 
hold on
for i=1:numel(caract)  
   rectangle('Position', caract(i).BoundingBox,'edgecolor','y')
   
end
% imshow(RGB);   
saveas(h,['soluciones_antes_examen' num2str(uu)],'bmp')
pause(2)
clear position labe
end

%%
% [H,T,R] = hough(bw);
% imshow(H,[],'XData',T,'YData',R,...
%             'InitialMagnification','fit');
% xlabel('\theta'), ylabel('\rho');
% axis on, axis normal, hold on;
% P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
% x = T(P(:,2)); y = R(P(:,1));
% plot(x,y,'s','color','white');
% lines = houghlines(bw,T,R,P,'FillGap',5,'MinLength',7);
% figure, imshow(bw), hold on
% max_len = 0;
% for k = 1:length(lines)
%    xy = [lines(k).point1; lines(k).point2];
%    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
% 
%    % Plot beginnings and ends of lines
%    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
% 
%    % Determine the endpoints of the longest line segment
%    len = norm(lines(k).point1 - lines(k).point2);
%    if ( len > max_len)
%       max_len = len;
%       xy_long = xy;
%    end
% end
% plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','cyan');
