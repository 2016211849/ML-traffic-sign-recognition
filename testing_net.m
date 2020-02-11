clear all 
close all

%u�itavanje istrenirane mre�e
load('netTransfer.mat')

%u�itavanje slike za testiranje
I=imread('stop_test.png','png');
%resizeanje na potrebnu veli�inu
img = imresize(I,[227 227]);

%klasifikacija

%iscrtavanje
label = classify(netTransfer , img);
figure
imshow(img)
title(string(label))