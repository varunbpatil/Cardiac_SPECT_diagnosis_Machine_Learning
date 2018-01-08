function [entropy, energy, cont, homo] = texture(im)
image = imread(im);
glcm = graycomatrix(image); % color image scaled to 8 gray levels
[M, N] = size(glcm);
entropy = 0;
energy = 0;
cont = 0;
homo = 0;

% calculate entropy
for i=1 to M
    for j=1 to N
        entropy = entropy + (glcm(i, j) * log(glcm(i, j))); 
    endfor
endfor

% calculate energy
for i=1 to M
    for j=1 to N
        energy = energy + (glcm(i, j)^2); 
    endfor
endfor

% calculate contrast
for i=1 to M
    for j=1 to N
        cont = cont + (glcm(i, j) * ((i - j)^2)); 
    endfor
endfor

% calculate homogeneity
for i=1 to M
    for j=1 to N
        homo = homo + (glcm(i, j) / (1 + abs(i - j))); 
    endfor
endfor

end
