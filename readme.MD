{\rtf1\ansi\ansicpg1252\cocoartf1187\cocoasubrtf390
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww25120\viewh14860\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 This package contains the following functions. \
\

\b \ul 0. TO GENERATE TEST DATA\

\b0 //I don't know how to do this\ulnone \
\

\b \ul 1. TO PREDICT THE AUTOENCODER'S PERFORMANCE ON SOME SET OF TEST DATA AFTER TRAINING IT ON ANOTHER SET OF DATA\

\i\b0 \ulnone function [acc] = stackedAEExercisePreliminary(train_data, train_labels, test_data, test_labels, inputSize, numClasses, hiddenSizeL1, hiddenSizeL2, sparsityParam, lambda, beta)    \

\i0 \
train_data = m * n matrix of m features corresponding to n train sets\
train_labels = 1 * n matrix consisting of the numeric class of each train set\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 test_data = m * n matrix of m features corresponding to n test sets\
test_label = 1 * n matrix consisting of the numeric class of each test set\
inputSize = number of input nodes (number of pixels in patch)\
numClasses = number of possible classes (equal to 2)\
hiddenSizeL1 = number of nodes in first hidden layer\
hiddenSizeL2 = number of nodes in second hidden layer\
sparsityParam = desired average activation of hidden nodes (default to .1)\
lambda = weight decay parameter (default to 3e-3)\
beta = sparsity penalty term (default to 3)\
\
acc = accuracy on dataset\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 \
\
\

\b \ul 2. TO GET THE AUTOENCODER SOFTMAX WEIGHTS BASED ON AN ENTIRE SET OF DATA\

\i\b0 \ulnone function [stackedAEOptTheta, netconfig] = stackedAEExercise(train_data, train_labels, inputSize, numClasses, hiddenSizeL1, hiddenSizeL2, sparsityParam, lambda, beta) \

\i0 \
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 train_data = m * n matrix of m features corresponding to n train sets\
train_labels = 1 * n matrix consisting of the numeric class of each train set\
inputSize = number of input nodes (number of pixels in patch)\
numClasses = number of possible classes (equal to 2)\
hiddenSizeL1 = number of nodes in first hidden layer\
hiddenSizeL2 = number of nodes in second hidden layer\
sparsityParam = desired average activation of hidden nodes (default to .1)\
lambda = weight decay parameter (default to 3e-3)\
beta = sparsity penalty term (default to 3)\
\
stackedAEOptTheta = Trained weights from classifier\
netconfig = network properties (needed for prediction)\
\
\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 \

\b \ul 3. TO GENERATE ACTIVATIONS BASED ON THE TRAINED SOFTMAX WEIGHTS\

\i\b0 \ulnone function [activs,activs_mean,net_activs] = nn_activation_map(im,stride,window_len, stackedAEOptTheta, inputSize, hiddenSizeL2, numClasses, netconfig, maxv, minv, stackedAEOptThetaMean, netconfigMean)\

\i0 \
im = raw image\
stride = stride of sliding window\
window_len = length of window side (assumed to be square)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 stackedAEOptTheta = trained weights\
inputSize = number of input pixels (window_len^2)\
hiddenSizeL2 = number of nodes in second hidden layer
\i \

\i0 numClasses = number of possible classes (equal to 2)\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 netconfig = network properties\
maxv = scaling \
minv = scaling\
stackedAEOptThetaMean = normalized-trained weights\
netconfigMean = normalized network properties\
\

\b \ul 4. TO EVALUATE PERFORMANCE ON A SINGLE MAMMOGRAM\

\i\b0 \ulnone function [fp,tp,fn,tn,accuracy] = performance(activations, roi, abs_size, abs_prob, prob_frac, prob_diff_thresh)\

\i0 \
activations = activations from the heat map\
roi = BW image from DDSM\
abs_size = minimum size of the mass\
prob_diff_thresh = required minimum range of the activations in order to do thresholding based on the max value\
prob_frac = fraction multiplied by the max value to get the probability threshold if the range crosses the required minimum\
abs_prob = default threshold for probability used if the range is not greater than the required minimum\
\
fp = false positives\
tp = true positives\
fn = false negatives\
tn = true negatives ( always = 0)\
accuracy = accuracy\
\
\
\
\
\
}