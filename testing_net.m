clear all 
close all

%uèitavanje istrenirane mreže
load('netTransfer.mat')

%uèitavanje slike za testiranje
I=imread('stop_test.png','png');
%resizeanje na potrebnu velièinu
img = imresize(I,[227 227]);

%klasifikacija

%iscrtavanje
label = classify(netTransfer , img);
figure
imshow(img)
title(string(label))