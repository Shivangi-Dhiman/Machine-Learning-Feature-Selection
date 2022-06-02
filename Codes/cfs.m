function ranking = cfs(X)

corrMatrix = abs( corr(X) );

% Ranking according to minimum correlations
scores = min(corrMatrix,[],2);


[~,ranking] = sort(scores,'ascend');

end