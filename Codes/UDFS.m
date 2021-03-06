function ranking = UDFS(X,nClass)

%======================setup===========================
gammaCandi = 10.^(-5);
lamdaCandi = 10.^(-5);
knnCandi = 1;
paramCell = fs_unsup_udfs_build_param(knnCandi, gammaCandi, lamdaCandi);
%======================================================
X = double(X);

disp('UDFS: Regularized Discriminative Feature Selection for Unsupervised Learning');
param = paramCell{1};
L = LocalDisAna(X', param);
if sum(sum(isnan(L)))>0
    ranking = [1:size(X,2)];
else
    A = X'*L*X;
    W = fs_unsup_udfs(A, nClass, param.gamma);
    Y = W;
    csvwrite('feature.csv',Y);
    [~, ranking] = sort(sum(W.*W,2),'descend');
end

end