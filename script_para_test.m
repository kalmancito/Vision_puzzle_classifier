% I = imread('3617ca70.png');
Ibw = ~im2bw(I,graythresh(I));
Ifill = imfill(Ibw,'holes');
Iarea = bwareaopen(Ifill,100);
Ifinal = bwlabel(Iarea);
stat = regionprops(Ifinal,'boundingbox');
imshow(I); hold on;
for cnt = 1 : numel(stat)
    bb = stat(cnt).BoundingBox;
    rectangle('position',bb,'edgecolor','r','linewidth',2);
end
%%
sides = caract(3).Extrema(4,:) - caract(3).Extrema(6,:); % Returns the sides of the square triangle that completes th


OrientationAngle = rad2deg(atan(-sides(2)/sides(1))) ; % Note the 'minus' sign compensates for the inverted y-values in image coordinates

%%
H=ROI_c{9};
mesh(conv2(rgb2gray(masc),rgb2gray(H)))
verdad=max(max(conv2(rgb2gray(masc),rgb2gray(H))));
%%

for ii=2:num
H=ROI_c{ii};
figure;
mesh(conv2(rgb2gray(masc),rgb2gray(H)))
kk(ii)=max(max(conv2(rgb2gray(masc),rgb2gray(H))))/verdad;
end


%%
%A es el original
%B es la mascara
%C es uno muy parecido
%D es la mascara del muy parecido
[numrows numcols]=size(B);
DD = imresize(D,[numrows numcols]) ;

H=DD|B;
sum(sum(H))
sum(sum(B))
imshow(H)
% imshow(H-B)
% figure; imshow(A)
% figure; imshow(B)
% figure; imshow(C)
% figure; imshow(D)

