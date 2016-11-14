function [mu1, muAll, fail] = error_detection(loss)
% Calculate the ERROR over of the landmarks using the loss (me17).
mu1=mean(loss(loss<0.1));
muAll=mean(loss);
fail=100*length(find(loss>0.1))/length(loss);
end
