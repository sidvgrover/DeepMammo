function [activs,activs_mean,net_activs] = nn_activation_map(im,stride,window_len, stackedAEOptTheta, inputSize, hiddenSizeL2, numClasses, netconfig, maxv, minv, stackedAEOptThetaMean, netconfigMean)
    im = double(im);
	im = im/255;
	dims = size(im);
	activs = zeros(dims);
	activs_mean = zeros(dims);
	counts = zeros(dims);

	for i=1:stride:dims(1) - window_len + 1
		display(dims(1) - window_len + 1 - i);
		for j=1:stride:dims(2) - window_len + 1
			[pred] = stackedAEPredictProb(stackedAEOptTheta, inputSize, hiddenSizeL2, numClasses, netconfig, reshape(im(i:i+window_len-1,j:j+window_len-1),window_len^2,1));
			activs(i:i+window_len-1,j:j+window_len-1) = activs(i:i+window_len-1,j:j+window_len-1) + pred(1);

			mean_patch = reshape(im(i:i+window_len-1,j:j+window_len-1),window_len^2,1);  
			mean_patch = mean_patch - mean(mean_patch);
			mean_patch = mean_patch + minv;
			mean_patch = mean_patch / maxv;
			mean_patch(mean_patch < 0) = 0;
			mean_patch(mean_patch > 1) = 1;

			[pred] = stackedAEPredictProb(stackedAEOptThetaMean, inputSize, hiddenSizeL2, ...
			              numClasses, netconfigMean, mean_patch);

			activs_mean(i:i+window_len-1,j:j+window_len-1) = activs_mean(i:i+window_len-1,j:j+window_len-1) + pred(1);
			
			counts(i:i+window_len-1,j:j+window_len-1) = counts(i:i+window_len-1,j:j+window_len-1) + 1;
		end
	end
	net_activs = activs + activs_mean; net_activs = net_activs / 2;
	net_activs = net_activs ./ counts;
	activs_mean = activs_mean ./ counts;
	activs = activs ./ counts;
end
