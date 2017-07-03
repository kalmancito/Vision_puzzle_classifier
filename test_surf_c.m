
close all
clear all
clc

load testsurf
% load testsurf2
% load testsurf3
load testsurf4
test2=test4;
for jj=1:12
    
    for ii=1:12
    mascara = rgb2gray(test{jj});
    [nr nc]=size(mascara);
    imagentest = imresize(rgb2gray(test2{ii}),[nr,nc]);

    points1 = detectSURFFeatures(mascara);
    points2 = detectSURFFeatures(imagentest);

    [f1,vpts1] = extractFeatures(mascara,points1);
    [f2,vpts2] = extractFeatures(imagentest,points2);

    indexPairs = matchFeatures(f1,f2);
    matchedPoints1 = vpts1(indexPairs(:,1));
    matchedPoints2 = vpts2(indexPairs(:,2));

    res(ii)=max(size(indexPairs));
    end
%         figure; showMatchedFeatures(mascara,imagentest,matchedPoints1,matchedPoints2);
%         legend('matched points 1','matched points 2');
    prueba{jj}=find(res==max(res));
    if(length(prueba{jj})>1)
        kk=13;
        prueba{jj}='Null'
    else
    kk=jj;
    figure;imshow(test2{prueba{jj}})
    figure;imshow(test{jj})
    pause(1)
    close all
    
    res;
%     clear res
    

switch kk
   case 1
      title('b1')
   case 2
      title('a1')
   case 3
      title('c1')
   case 4
      title('a2')
   case 5
      title('b2')
   case 6
      title('c2')
   case 7
      title('a3')
   case 8
      title('b3')
    case 9
      title('c3')
  case 10
      title('a4')
  case 11
      title('b4')
  case 12
      title('c4')
          
    otherwise
        disp('other value')
end
end 


end
prueba
close all
% size(matchedPoints1)