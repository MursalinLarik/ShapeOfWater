%% NEW FINAL
%Image Processing
clear all; clc;
filename = "Lineset_line"; %Name Series
i =44; %File Index
ext = ".jpg"; %Format
fname = sprintf('%s%i%s', filename, i, ext);
A = imread(fname);
A = imresize(A, [382 800]);
A = imcrop(A,[24, 24, 800, 382]);
a = 1.5 * log(1 + im2double(A));
figure, subplot 221
imshow(a)
b = imcomplement(a);
c = im2bw(b);
d = im2double(c);
subplot 222
imshow(d)
mod = mode(d,2);
e = repmat(mod, 1, size(d, 2));
f = d-e;
BW1 = bwareaopen(f, 200);
SE = strel('disk',2);
BW2 = imclose(BW1, SE);

SE2 = strel('line', 15, 0);
BW3 = imclose(BW2, SE2);
windowSize = 7;
kernel = ones(windowSize) / windowSize ^ 2;
blurryImage = conv2(single(BW3), kernel, 'same');
BW3 = blurryImage > 0.4; % Rethreshold
BW3 = bwareafilt(BW3,1,'largest');
subplot 223
imshow(BW3)

[labeledImage, numberOfObjects] = bwlabel(BW3);
if numberOfObjects>=1
    info = regionprops(BW3,'Boundingbox');
    [row, col] = find(BW3 == 1);
    minimumxx = min(row);
    medd = find(row == minimumxx);
    sorrt = sort(medd);
    index_xx = sorrt((round(length(sorrt)/2)));
    x_cor = col(index_xx);
    y_cor = row(index_xx);
    subplot 224;
    imshow(a)
    hold on
    plot(x_cor,y_cor,'+b','linewidth',2,'MarkerSize',10);
    hold on
    for k = 1 : length(info)
        BB = info(k).BoundingBox;
        rectangle('Position', [BB(1)-15,BB(2)-15,BB(3)+15,BB(4)+15],'EdgeColor','r','LineWidth',2) ;
    end
    hold on
    [ysize, xsize] = size(a);
    depth = 2*log2(14.5+exp(0.19))*(y_cor+20)/(ysize-120)
    caption = sprintf('Depth = %f', depth);
    text(10, 10, caption, 'FontSize', 20, 'Color','red');
else
    x_cor = NaN;
    y_cor = NaN;
    depth = NaN;
end
