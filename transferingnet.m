%%zatvaranje svih prozora
close all 
clear all

%%
%%namjestanje patha do dataset-a
digitDatasetPath = fullfile('C:\Users\CALIC\Desktop\strojno_seminarski\','traffic_sign_lg');

%%spremanje podataka u imageDataStore
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

%odvajanje seta za treniranje i validaciju
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.75,'randomized');

%broj podataka za treniranje
numTrainImages = numel(imdsTrain.Labels);

%%
%ucitavanje alexnet pretrenirane mreze
net = alexnet;


inputSize = net.Layers(1).InputSize

%uzimanje layera iz alexnet osim zadnja 3
layersTransfer = net.Layers(1:end-3);

%broj klasa u dataset-u
numClasses = numel(categories(imdsTrain.Labels))

%dodavanje i posljednja 3 sloja
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%pohrana podataka proširene slike tj. automatsko mijenjanje velièine slike
%za trening
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);


augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',2, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%trniranje mreže
netTransfer = trainNetwork(augimdsTrain,layers,options);

%spremanje trenirane mreže 
save('netTransfer.mat', 'netTransfer');

%%

%%provjera mreže

%klasificiranje
[YPred,scores] = classify(netTransfer,augimdsValidation);

%testiranje na 10 random uzoraka iz validacijskog seta podataka
idx = randperm(numel(imdsValidation.Files),10);
figure
for i = 1:10
    subplot(5,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation)


 