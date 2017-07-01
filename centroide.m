function centro = centroide(B)
%CENTROIDE Extrae del centro de gravedad de una region en img. binaria.
%   centro = centroide(B)
%   Devuelve un vector 1x2 con coordenadas x, y del centro de gravedad  
%   -------->Y
%   |
%   |
%   |
%   v  X
%   La imagen B debe contener solo un objeto y este debe estar en blanco.

%______________________________________________________________________%
%                                                                      %
% NOMBRE DEL FICHERO:   centroide.m                                    %
%                                                                      %
% RECIBE: Imagen binaria con un solo objeto                            %
% DEVUELVE: Las coordenadas del centro de gravedad en coord X,Y        %
% ALGORITMO EMPLEADO: calculo momento de primer orden                  %
% VARIABLES:   B = imagen binaria (0s y 1s    o   0s y 255s)           %
%              filas = numero de filas de la imagen B                  %
%              columnas = numero de columnas de la imagen B            %
%              i = variable contador para recorrer las filas de la img.% 
%              j = variable contador para recorrer las columnas        %
%              sum_i = sumatorio de las i                              %
%              sum_j = sumatorio de las j                              %
%              ci = fila del centro de gravedad                        %
%              cj = columna del centro de gravedad                     %
%              centro = (ci, cj)                                       %
%              numPixeles = area del objeto en pixeles                 %
% BUGS: Solo un objeto en la imagen, sino da el c.g. del conjunto      %
% SUPOSICIONES : fondo negro y objeto en blanco. Hay 1 objeto en img   %
% TEMA 4 IMAGENES BINARIAS                                             %
% AUTORES: Eusebio de la Fuente y Félix Miguel Trespaderne             %
%______________________________________________________________________%
%  Comandos de ejecucion:                                              %
%----------------------------------------------------------------------%                                                            
% >> I = imread('fija1.bmp');                                        
% >> B = binauto(I);        %binariza con nuestra funcion binauto
% >> B = ~B;                %pone objeto en blanco
% >> centro = centroide(B);
% >> imshow(B);
% >> hold on
% >> plot(centro(2),centro(1),'r+') %Con plot invertir coordcoord
% >> hold off; 
%______________________________________________________________________%
% Con funciones de matlab                                      
% >> Ient = imread('corona2.bmp');                                            
% >> BW = Ient > 128;        %binariza 'a pelo' con umbral 128 
% >> BW = ~BW;               %pone objeto en blanco                 
% >> [I,J] = find(BW);       %guarda filas y columnas no nulas de BW  
% >> c_i = sum(I)/length(I); %coord.CG filas
% >> c_j = sum(J)/length(J); %coord.CG columnas
% >> imshow(Ient);           %visualización
% >> hold on;                %pintará siguiente gráfico sobre última imagen
% >> plot(c_j, c_i,'r+');    %pinta cruz en CG. Con plot invertir coord!!
% >> hold off; 
%______________________________________________________________________%
% Con funciones de matlab image processing toolbox   
% Ojo si hay más de un objeto en la imagen binaria falla!!
% >> Ient = imread('corona2.bmp');                                            
% >> umbralOpt=graythresh(Ient);  %busca umbral optimo
% >> BW = im2bw(Ient,umbralOpt);  %binariza
% >> BW = ~BW;                    %pone objeto en blanco                 
% >> s  = regionprops(BW, 'centroid');
% >> imshow(Ient);                %visualización
% >> hold on;                     
% >> plot(s.Centroid(1,1), s.Centroid(1,2), 'r+')
% >> hold off; 
%______________________________________________________________________%

[filas,columnas] = size(B);

sum_i =0;
sum_j =0;
num_pixeles=0;

for i = 1:filas
    for j = 1: columnas
        if (B(i,j)~=0)
            num_pixeles= num_pixeles + 1;
            sum_i = sum_i + i;
            sum_j = sum_j + j;
        end
    end
end

%coord.del centroide en (filas, columnas)
ci = sum_i/num_pixeles;
cj = sum_j/num_pixeles;
centro = [ci, cj]; 

