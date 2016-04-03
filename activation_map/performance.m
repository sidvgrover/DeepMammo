function [fp,tp,fn,tn,accuracy] = performance(activations, roi, abs_prob, abs_size, prob_frac, prob_diff_thresh)

fp = 0;
tp = 0;
fn = 0;
tn = 0;

connectivity_degree = 4;

%{
This block of code deals with a problem that occurs in which a mass is distinctly identified but the entire probability
spectrum lies in 0 to x with x << 1. First, a check is done to ensure that the range of the matrix is above a certain threshold
in order to confirm that it is not benign. 

After that, the threshold is adjusted to being some fraction of the max probability. 

In practice, this works quite well with a prob_frac ~ .7 and a prob_diff_thresh ~ .4 - .5. In most cases the max falls in the range .97 - 1
so the threshold is usually around .7.
%}
prob_diff = max(max(activations)) - min(min(activations));
if prob_diff >= prob_diff_thresh
	abs_prob = max(max(activations)) * prob_frac;
end

activations(activations < abs_prob) = 0;
activations(activations ~= 0) = 1;


activations = bwareaopen(activations, thresh_size, connectivity_degree);
connected_comps_hmap = bwconncomp(activations,connectivity_degree);
connected_comps_roi = bwconncomp(roi,connectivity_degree);

centroid_hmap = regionprops(connected_comps_hmap, 'Centroid');
centroid_roi = regionprops(connected_comps_roi, 'Centroid');

have_caught_mass = zeros([1 numel(centroid_roi)]);

for i=1:numel(centroid_hmap)
	flag = 0;
	for j=1:numel(centroid_roi)
		euclid_distance = sqrt(sum((centroid_hmap(i)-centroid_roi(j)).^2));
		if euclid_distance < thresh_dist
			flag = 1;
			have_caught_mass(j) = 1;
			tp = tp + 1;
		end
	end

	if(flag == 0)
		fp = fp + 1;
	end
end

fn = fn + numel(have_caught_mass ~= 1);
accuracy = (tp + tn)/(tp + tn + fp + fn);
end

