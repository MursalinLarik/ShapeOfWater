clear all; clc;
filename = "Lineset_line";
ext = ".jpg";
Filenames = [];
CoordinatesAndDepth = [];
GPS_Cor = [];
for i = 1:85
    fname = sprintf('%s%i%s', filename, i, ext);
    A = imread(fname);
    A = imresize(A, [382 800]);
    A = imcrop(A,[24, 24, 800, 382]);
    a = 1.5 * log(1 + im2double(A));
    b = imcomplement(a);
    c = im2bw(b);
    d = im2double(c);
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
        [ysize, xsize, zsize] = size(a);
        depth = 2*log2(14.5+exp(0.19))*(y_cor+20)/(ysize-120)
        excel_fname = sprintf('%s%i%s', "line", i, ".csv");
        [NUM,TXT,RAW]= xlsread(excel_fname);
        [NoOfRows,NoOfColumn]= size(NUM);
        trace_apex = round(x_cor/xsize * NoOfRows);
        x_gps = NUM(trace_apex,10);
        y_gps = NUM(trace_apex,9);
        if x_gps==0
            x_gps = NaN;
            y_gps = NaN;
        end
    else
        x_cor = NaN;
        y_cor = NaN;
        depth = NaN;
        x_gps = NaN;
        y_gps = NaN;
    end
    Filenames = [Filenames; [string(fname)]];
    CoordinatesAndDepth = [CoordinatesAndDepth; [x_cor y_cor depth]];
    GPS_Cor = [GPS_Cor; [x_gps, y_gps]];
end
T = table(Filenames, CoordinatesAndDepth);
W = table(Filenames, GPS_Cor);
Sheetname = 'Eval1.xlsx';
writetable(T,Sheetname,'Sheet',1,'Range','A1')
Sheetname2 = 'Eval1_GPS.xlsx';
writetable(W,Sheetname2,'Sheet',1,'Range','A1')