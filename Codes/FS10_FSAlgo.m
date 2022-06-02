clear all
close all
clc;

% Include dependencies
addpath('C:\Users\PC\AppData\Roaming\MathWorks\MATLAB Add-Ons\Toolboxes\Feature Selection Library\FSLib_v7.0.1_2020_2\lib'); % dependencies
addpath('C:\Users\ PC\AppData\Roaming\MathWorks\MATLAB Add-Ons\Toolboxes\Feature Selection Library\FSLib_v7.0.1_2020_2\methods'); % FS methods
addpath(genpath('C:\Users\ PC AppData\Roaming\MathWorks\MATLAB Add-Ons\Toolboxes\Feature Selection Library\FSLib_v7.0.1_2020_2\lib\drtoolbox'));

% Select a feature selection method from the list
listFS = {'UDFS','llcfs','cfs'};

[ methodID ] = readInput( listFS );
selection_method = listFS{methodID}; % Selected


%M=readtable('D1.csv'); C:\Users\PC\AppData\Roaming\MathWorks\MATLAB Add-Ons\Toolboxes\Feature Selection Library\
M=readtable('D:\UWindsor\Studies\Machine Learning\Project\Output\D1\D1.csv');
% X=table2array(M(:,1:354));
% Y=table2array(M(:,355));

X = M(:,1:end-1);
Y = M(:,end);
X=table2array(X);
Y=table2array(Y);


d_split = 0.20;
P = cvpartition(Y,'Holdout',d_split);

X_train = X(P.training,:) ;
Y_train = Y(P.training);

X_test =  X(P.test,:) ;
Y_test =  Y(P.test);

% number of features
numF = size(X_train,2);


% feature Selection on training data
switch lower(selection_method)   
    case 'udfs'
        
        nClass = 2;
        ranking = UDFS(X_train , nClass );
        csvwrite('D:\UWindsor\Studies\Machine Learning\Project\Output\D1\D1_UDFS.csv' , ranking)
        
    case 'cfs'
       
        ranking = cfs(X_train);
        csvwrite('D:\UWindsor\Studies\Machine Learning\Project\Output\D1\D1_CFS.csv' , ranking)
        
    case 'llcfs'
        
        ranking = llcfs( X_train );
        csvwrite('D:\UWindsor\Studies\Machine Learning\Project\Output\D1\D1_LLCFS.csv' , ranking)
        
    otherwise
        disp('Unknown method.')
end


