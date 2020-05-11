clc
clear
close all
str='C:\Users\HP\Desktop\PR\PR_Project\S1_Dataset\';
folder_name=uigetdir(str);
files=dir(fullfile(folder_name,'S*'));
filestrain=files(1:40);
filestest=files(41:length(files));
curr_folder=pwd;
cd(folder_name);
dattrain={};
dattest={};
%DATA1=[];
for i=1:(length(filestrain))
    filestrain(i).name;
    dattrain(i)={importdata(filestrain(i).name)};
    %DATA1=[DATA1;importdata(files(i).name)];
      
end
for i=1:(length(filestest))
    filestest(i).name;
    dattest(i)={importdata(filestest(i).name)};
    %DATA1=[DATA1;importdata(files(i).name)];
      
end
k1=i;
str1='C:\Users\HP\Desktop\PR\PR_Project\S2_Dataset\';
folder_name=uigetdir(str1);
files=dir(fullfile(folder_name,'d2p*'));
filestrain=files(1:10);
filestest=files(11:length(files));
length(filestest);
curr_folder=pwd;
cd(folder_name);
%DATA1=[];
for i=1:(length(filestrain))
    filestrain(i).name;
    dattrain(i+k1)={importdata(filestrain(i).name)};
    %DATA1=[DATA1;importdata(files(i).name)];
      
end
for i=1:(length(filestest))
    filestest(i).name;
    dattest(i+k1)={importdata(filestest(i).name)};
    %DATA1=[DATA1;importdata(files(i).name)];
      
end
    % N1=600;
    % N2=600;
    % N3=600;
    % N4=600;
    % for i=1:length(dattrain)
    %     x=cell2mat(dattrain(i));
    %     a=x;
    %     %index=1;
    %     k=a(1,9);
    %     N1try=0;
    %     N2try=0;
    %     N3try=0;
    %     N4try=0;
    %     for j=2:length(a(:,9))
    %         if a(j,9)==1
    %             if k==1
    %             N1try=N1try+a(j,1)-a(j-1,1);
    %             if N2try~=0 && N2>N2try
    %                 N2=N2try;
    %                 N2try=0;
    %             end
    %             if N3try~=0 && N3>N3try
    %                 N3=N3try;
    %                 N3try=0;
    %             end
    %             if N4try~=0 && N4>N4try
    %                 N4=N4try;
    %                 N4try=0;
    %             end
    %             else
    %                 k=1;
    %                 N1try=0;
    %             end
    %         elseif a(j,9)==2
    %             if k==2
    %             N2try=N2try+a(j,1)-a(j-1,1);
    %             if N1try~=0 && N1>N1try
    %                 N1=N1try;
    %             end
    %             if N3try~=0 && N3>N3try
    %                 N3=N3try;
    %             end
    %             if N4try~=0 && N4>N4try
    %                 N4=N4try ;   
    %             end
    %             else
    %                 k=2;
    %                 N2try=0;
    %             end
    %         elseif a(j,9)==3
    %             if k==3
    %             N3try=N3try+a(j,1)-a(j-1,1);
    %             if N1try~=0 && N1>N1try
    %                 N1=N1try;
    %             end
    %             if N2try~=0 && N2>N2try
    %                 N2=N2try;
    %             end
    %             if N4try~=0 && N4>N4try
    %                 N4=N4try ;   
    %             end
    %             else
    %                 k=3;
    %                 N3try=0;
    %             end
    %         elseif a(j,9)==4
    %             if k==4
    %             N4try=N4try+a(j,1)-a(j-1,1);
    %             if N1try~=0 && N1>N1try
    %                 N1=N1try;
    %             end
    %             if N2try~=0 && N2>N2try
    %                 N2=N2try;
    %             end
    %             if N3try~=0 && N3>N3try
    %                 N3=N3try ;   
    %             end
    %             else
    %                 k=4;
    %                 N4try=0;
    %             end
    %         end    
    %     end
    % end
DATA=[];
for i=1:length(dattrain)
    DATA=[DATA;cell2mat(dattrain(i))];
end
testDATA=[];
for i=1:length(dattest)
    testDATA=[testDATA;cell2mat(dattest(i))];
end
clear x;
clear a;
x=[DATA];
a=x;
data=[sin(atan(a(:,2)./a(:,3))),atan(a(:,4)./a(:,2)),atan(a(:,4)./a(:,3)),a(:,6),a(:,9)];
D=[a(:,3),a(:,4),a(:,5),a(:,6),sin(atan(a(:,2)./a(:,3))),atan(a(:,4)./a(:,2)),atan(a(:,4)./a(:,3)),a(:,7)./a(:,8),sqrt(a(:,2).^2 + a(:,3).^2 + a(:,4).^2),a(:,9)];
x1=[testDATA];
a=x1;
%data=[sin(atan(a(:,2)./a(:,3))),atan(a(:,4)./a(:,2)),atan(a(:,4)./a(:,3)),a(:,6),a(:,9)];
Dtest=[a(:,3),a(:,4),a(:,5),a(:,6),sin(atan(a(:,2)./a(:,3))),atan(a(:,4)./a(:,2)),atan(a(:,4)./a(:,3)),a(:,7)./a(:,8),sqrt(a(:,2).^2 + a(:,3).^2 + a(:,4).^2),a(:,9)];

inputTable = array2table(D, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6','column_7','column_8','column_9','column_10'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6','column_7','column_8','column_9'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_10;
%%finesvmgaussian

template2 = templateSVM(...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 0.25, ...
    'BoxConstraint', 1, ...
    'Standardize', true);
classification_fineGAussian_SVM = fitcecoc(...
    predictors, ...
    response, ...
    'Learners', template2, ...
    'Coding', 'onevsone', ...
    'ClassNames', [1; 2; 3; 4]);
 classification_Medium_KNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 1, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [1; 2; 3; 4]);
template9 = templateTree(...
    'MaxNumSplits', 18669);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', template9, ...
    'ClassNames', [1; 2; 3; 4]);
y2=predict(classification_Medium_KNN,predictors);
confmat2=confusionmat(response,y2)
% cross validation
% partitionedModel = crossval(classification_fineGAussian_SVM, 'KFold', 10);
% Compute validation accuracy
% validationAccuracy = 1 - kfoldLoss(partitionedModel, 'Loss', 'ClassifError')

accuracy_FineGaussian=100*sum(diag(confmat2))/sum(sum(confmat2))
% test accuracy
pred2=predict(classification_Medium_KNN,Dtest(:,1:length(Dtest(1,:))-1));
confmat2test=confusionmat(Dtest(:,length(Dtest(1,:))),pred2)
accuracy_FineGaussian_test=100*sum(diag(confmat2test))/sum(sum(confmat2test))
metrics = MyClassifyPerf(Dtest(:,length(Dtest(1,:))), pred2);
disp("precision");
metrics.precision 
disp("recall");
metrics.recall 
disp("F1-score");
metrics.F1
% figure, plotroc(Dtest(:,length(Dtest(1,:))), pred2)
% title("Gaussian SVM ROC");
% T= D(:,length(D(1,:)));
% for i =1:length(targ)
%    if(targ(i)==1)
%       t(i,:)=[1,0,0,0];
%    elseif(targ(i)==2)
%       t(i,:)=[0,1,0,0];
%    elseif(targ(i)==3)
%       t(i,:)=[0,0,1,0];
%    else
%        t(i,:)=[0,0,0,1];
%    end
% end
% % T=t;
%  clear t;
% clear targ;
%mdlSVM = fitPosterior(classification_fineGAussian_SVM)
% [~,score] = resubPredict(classification_fineGAussian_SVM);
% [XK,YK] = perfcurve(T,score(:,1),0)
% plot(XK,YK)
targ=D(:,length(D(1,:)))';
for i =1:length(targ)
   if(targ(i)==1)
      t(i,:)=[1,0,0,0];
   elseif(targ(i)==2)
      t(i,:)=[0,1,0,0];
   elseif(targ(i)==3)
      t(i,:)=[0,0,1,0];
   else
       t(i,:)=[0,0,0,1];
   end
end
XK=t';
clear t;
clear targ;
targ=y2';
for i =1:length(targ)
   if(targ(i)==1)
      t(i,:)=[1,0,0,0];
   elseif(targ(i)==2)
      t(i,:)=[0,1,0,0];
   elseif(targ(i)==3)
      t(i,:)=[0,0,1,0];
   else
       t(i,:)=[0,0,0,1];
   end
end
YK=t';
clear t;
clear targ;
figure
plotroc(XK,YK);
title('Fine Gaussian');
%  %% fineknn
% classification_Fine_KNN = fitcknn(...
%     predictors, ...
%     response, ...
%     'Distance', 'Euclidean', ...
%     'Exponent', [], ...
%     'NumNeighbors', 100, ...
%     'DistanceWeight', 'Equal', ...
%     'Standardize', true, ...
%     'ClassNames', [1; 2; 3; 4]);
% y3=predict(classification_Fine_KNN,predictors);
% confmat3=confusionmat(response,y3)
% accuracy_FineKNN=100*sum(diag(confmat3))/sum(sum(confmat3))
% 
% %cross validation
% % partitionedModel = crossval(classification_Fine_KNN, 'KFold', 10);
% % % Compute validation accuracy
% % validationAccuracy = 1 - kfoldLoss(partitionedModel, 'Loss', 'ClassifError')
% 
% %test accuracy
% pred3=predict(classification_Fine_KNN,Dtest(:,1:length(Dtest(1,:))-1));
% confmat3test=confusionmat(Dtest(:,length(Dtest(1,:))),pred3)
% accuracy_FineKNN_test=100*sum(diag(confmat3test))/sum(sum(confmat3test))
% metrics = MyClassifyPerf(Dtest(:,length(Dtest(1,:))), pred3);
% disp("precision");
% metrics.precision 
% disp("recall");
% metrics.recall 
% disp("F1-score");
% metrics.F1
% %figure, plotroc(Dtest(:,length(Dtest(1,:))), pred3)
% %title("KNN ROC");
% % [~,score] = resubPredict(classification_Fine_KNN);
% % [XK,YK] = perfcurve(T,score(:,1),0)
% % figure,
% % plot(XK,YK)
% 
% targ=D(:,length(D(1,:)))';
% for i =1:length(targ)
%    if(targ(i)==1)
%       t(i,:)=[1,0,0,0];
%    elseif(targ(i)==2)
%       t(i,:)=[0,1,0,0];
%    elseif(targ(i)==3)
%       t(i,:)=[0,0,1,0];
%    else
%        t(i,:)=[0,0,0,1];
%    end
% end
% XK=t;
% clear t;
% clear targ;
% targ=y3';
% for i =1:length(targ)
%    if(targ(i)==1)
%       t(i,:)=[1,0,0,0];
%    elseif(targ(i)==2)
%       t(i,:)=[0,1,0,0];
%    elseif(targ(i)==3)
%       t(i,:)=[0,0,1,0];
%    else
%        t(i,:)=[0,0,0,1];
%    end
% end
% YK=t;
% clear t;
% clear targ;
% figure,
% plotroc(XK,YK);
% title('Fine KNN');
% % % %% mediumknn
% % classification_Medium_KNN = fitcknn(...
% %     predictors, ...
% %     response, ...
% %     'Distance', 'Euclidean', ...
% %     'Exponent', [], ...
% %     'NumNeighbors', 100, ...
% %     'DistanceWeight', 'Equal', ...
% %     'Standardize', true, ...
% %     'ClassNames', [1; 2; 3; 4]);
% % y4=predict(classification_Medium_KNN,predictors);
% % confmat4=confusionmat(response,y4)
% % accuracy_MediumKNN=100*sum(diag(confmat4))/sum(sum(confmat4))
% % pred4=predict(classification_Fine_KNN,Dtest(:,1:length(Dtest(1,:))-1));
% % confmat4test=confusionmat(Dtest(:,length(Dtest(1,:))),pred4)
% % accuracy_FineKNN_test=100*sum(diag(confmat4test))/sum(sum(confmat4test))
% %  metrics = MyClassifyPerf(Dtest(:,length(Dtest(1,:))), pred4);
% % disp("precision");
% % metrics.precision 
% % disp("recall");
% % metrics.recall 
% % disp("F1-score");
% % metrics.F1
% % %figure, plotroc(Dtest(:,length(Dtest(1,:))), pred3)
% % %title("KNN ROC");
% % % [~,score] = resubPredict(classification_Fine_KNN);
% % % [XK,YK] = perfcurve(T,score(:,1),0)
% % % figure,
% % % plot(XK,YK)
% % 
% % targ=D(:,length(D(1,:)))';
% % for i =1:length(targ)
% %    if(targ(i)==1)
% %       t(i,:)=[1,0,0,0];
% %    elseif(targ(i)==2)
% %       t(i,:)=[0,1,0,0];
% %    elseif(targ(i)==3)
% %       t(i,:)=[0,0,1,0];
% %    else
% %        t(i,:)=[0,0,0,1];
% %    end
% % end
% % XK=t;
% % clear t;
% % clear targ;
% % targ=y4';
% % for i =1:length(targ)
% %    if(targ(i)==1)
% %       t(i,:)=[1,0,0,0];
% %    elseif(targ(i)==2)
% %       t(i,:)=[0,1,0,0];
% %    elseif(targ(i)==3)
% %       t(i,:)=[0,0,1,0];
% %    else
% %        t(i,:)=[0,0,0,1];
% %    end
% % end
% % YK=t;
% % clear t;
% % clear targ;
% % figure,
% % plotroc(XK,YK);
% % title('Medium KNN');
% % %% cubic svm
% % % template3 = templateSVM(...
% % %     'KernelFunction', 'polynomial', ...
% % %     'PolynomialOrder', 3, ...
% % %     'KernelScale', 'auto', ...
% % %     'BoxConstraint', 1, ...
% % %     'Standardize', true);
% % % classification_Cubic_SVM = fitcecoc(...
% % %     predictors, ...
% % %     response, ...
% % %     'Learners', template3, ...
% % %     'Coding', 'onevsone', ...
% % %     'ClassNames', [1; 2; 3; 4]);
% % % y5=predict(classification_Cubic_SVM,predictors);
% % % confmat5=confusionmat(response,y5)
% % % accuracy_CubicSVM=100*sum(diag(confmat5))/sum(sum(confmat5))
% % 
% % %% fine tree
% % % classification_Fine_Tree = fitctree(...
% % %     predictors, ...
% % %     response, ...
% % %     'SplitCriterion', 'gdi', ...
% % %     'MaxNumSplits', 100, ...
% % %     'Surrogate', 'off', ...
% % %     'ClassNames', [1; 2; 3; 4]);
% % % y6=predict(classification_Fine_Tree,predictors);
% % % confmat6=confusionmat(response,y6)
% % % accuracy_FineTree=100*sum(diag(confmat6))/sum(sum(confmat6))
% % % %cross validation
% % % % partitionedModel = crossval(classification_Fine_Tree, 'KFold', 10);
% % % % % Compute validation accuracy
% % % % validationAccuracy = 1 - kfoldLoss(partitionedModel, 'Loss', 'ClassifError')
% % % 
% % % %test accuracy
% % % pred6=predict(classification_Fine_Tree,Dtest(:,1:length(Dtest(1,:))-1));
% % % confmat6test=confusionmat(Dtest(:,length(Dtest(1,:))),pred6)
% % % accuracy_FineTree_test=100*sum(diag(confmat6test))/sum(sum(confmat6test))
% % % metrics = MyClassifyPerf(Dtest(:,length(Dtest(1,:))), pred6);
% % % disp("precision");
% % % metrics.precision 
% % % disp("recall");
% % % metrics.recall 
% % % disp("F1-score");
% % % metrics.F1
% % %% Naive bayes
% % 
% classification_NaiveBayes = fitcnb(...
%     predictors, ...
%     response, ...
%     'ClassNames', [1; 2; 3; 4]);
% y8=predict(classification_NaiveBayes,predictors);
% confmat8=confusionmat(response,y8)
% accuracy_NaiveBayes=100*sum(diag(confmat8))/sum(sum(confmat8))
% %cross validation
% % partitionedModel = crossval(classification_NaiveBayes, 'KFold', 10);
% % % Compute validation accuracy
% % validationAccuracy = 1 - kfoldLoss(partitionedModel, 'Loss', 'ClassifError')
% 
% %inputs->response,y
% 
% pred8=predict(classification_NaiveBayes,Dtest(:,1:length(Dtest(1,:))-1));
% confmat8test=confusionmat(Dtest(:,length(Dtest(1,:))),pred8)
% accuracy_NaiveBayes_test=100*sum(diag(confmat8test))/sum(sum(confmat8test))
% metrics = MyClassifyPerf(Dtest(:,length(Dtest(1,:))), pred8);
% %figure, plotroc(Dtest(:,length(Dtest(1,:))), pred3)
% %title("KNN ROC");
% % [~,score] = resubPredict(classification_Fine_KNN);
% % [XK,YK] = perfcurve(T,score(:,1),0)
% % figure,
% % plot(XK,YK)
% 
% targ=response';
% for i =1:length(targ)
%    if(targ(i)==1)
%       t(i,:)=[1,0,0,0];
%    elseif(targ(i)==2)
%       t(i,:)=[0,1,0,0];
%    elseif(targ(i)==3)
%       t(i,:)=[0,0,1,0];
%    else
%        t(i,:)=[0,0,0,1];
%    end
% end
% XK=t;
% clear t;
% clear targ;
% targ=y8';
% for i =1:length(targ)
%    if(targ(i)==1)
%       t(i,:)=[1,0,0,0];
%    elseif(targ(i)==2)
%       t(i,:)=[0,1,0,0];
%    elseif(targ(i)==3)
%       t(i,:)=[0,0,1,0];
%    else
%        t(i,:)=[0,0,0,1];
%    end
% end
% YK=t;
% clear t;
% clear targ;
% figure,
% plotroc(XK,YK);
% 
% title('Naive bayes');
% %% ensemble bagged trees
% template9 = templateTree(...
%     'MaxNumSplits', 18669);
% classificationEnsemble = fitcensemble(...
%     predictors, ...
%     response, ...
%     'Method', 'Bag', ...
%     'NumLearningCycles', 30, ...
%     'Learners', template9, ...
%     'ClassNames', [1; 2; 3; 4]);
% y9=predict(classificationEnsemble,predictors);
% confmat9=confusionmat(response,y9)
% accuracy_baggedtrees=100*sum(diag(confmat9))/sum(sum(confmat9))
% %cross validation
% % partitionedModel = crossval(classificationEnsemble, 'KFold', 10);
% % % Compute validation accuracy
% % validationAccuracy = 1 - kfoldLoss(partitionedModel, 'Loss', 'ClassifError')
% 
% pred9=predict(classificationEnsemble,Dtest(:,1:length(Dtest(1,:))-1));
% confmat9test=confusionmat(Dtest(:,length(Dtest(1,:))),pred9)
% accuracy_baggedtrees_test=100*sum(diag(confmat9test))/sum(sum(confmat9test))
% 
% metrics = MyClassifyPerf(Dtest(:,length(Dtest(1,:))), pred9);
% disp("precision");
% metrics.precision 
% disp("recall");
% metrics.recall 
% disp("F1-score");
% metrics.F1
% %figure, plotroc(Dtest(:,length(Dtest(1,:))), pred9)
% %title("bgged trees ROC");
% % targ=D(:,length(D(1,:)))';
% % for i =1:length(targ)
% %    if(targ(i)==1)
% %       t(i,:)=[1,0,0,0];
% %    elseif(targ(i)==2)
% %       t(i,:)=[0,1,0,0];
% %    elseif(targ(i)==3)
% %       t(i,:)=[0,0,1,0];
% %    else
% %        t(i,:)=[0,0,0,1];
% %    end
% % end
% % XK=t';
% % clear t;
% % clear targ;
% % targ=y9';
% % for i =1:length(targ)
% %    if(targ(i)==1)
% %       t(i,:)=[1,0,0,0];
% %    elseif(targ(i)==2)
% %       t(i,:)=[0,1,0,0];
% %    elseif(targ(i)==3)
% %       t(i,:)=[0,0,1,0];
% %    else
% %        t(i,:)=[0,0,0,1];
% %    end
% % end
% % YK=t';
% % clear t;
% % clear targ;
% % figure
% % plotroc(XK,YK);
% % title('bagged trees');
% %% neural netwoks
% clear t;
% clear targ;
% Data=D(:,1:length(D(1,:))-1)';
% targ=D(:,length(D(1,:)));
% for i =1:length(targ)
%    if(targ(i)==1)
%       t(i,:)=[1,0,0,0];
%    elseif(targ(i)==2)
%       t(i,:)=[0,1,0,0];
%    elseif(targ(i)==3)
%       t(i,:)=[0,0,1,0];
%    else
%        t(i,:)=[0,0,0,1];
%    end
% end
% t=t';
% x = Data;
% t = t;
% 
% trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
% 
% % Create a Pattern Recognition Network
% hiddenLayerSize = 5;
% net = patternnet(hiddenLayerSize, trainFcn);
% 
% % Train the Network
% [net,tr] = train(net,x,t);
% clear t;
% clear targ;
% % Test the Network
% y = net(Dtest(:,1:length(Dtest(1,:))-1)');
% targ=Dtest(:,length(Dtest(1,:)));
% for i =1:length(targ)
%    if(targ(i)==1)
%       t(i,:)=[1,0,0,0];
%    elseif(targ(i)==2)
%       t(i,:)=[0,1,0,0];
%    elseif(targ(i)==3)
%       t(i,:)=[0,0,1,0];
%    else
%        t(i,:)=[0,0,0,1];
%    end
% end
% t=t';
% e = gsubtract(t,y);
% % %performance = perform(net,t,y)
% % % figure, ploterrhist(e)
% figure, plotconfusion(t,y)
% title("test ANN confusion matrix");
% % figure, plotroc(t,y)
% tind=vec2ind(t);
% yind=vec2ind(y);
% metrics = MyClassifyPerf(tind, yind);
% disp("precision");
% metrics.precision 
% disp("recall");
% metrics.recall 
% disp("F1-score");
% metrics.F1
function metrics = MyClassifyPerf(targets, outputs)
Mv = max(targets);
M = Mv(1);

prec_sum = 0;
rec_sum = 0;

for ix = 1:M
    clear hx;
    clear y;
    hx = length(find(targets == ix));
    y = length(find(outputs == ix));
    clear temp_targ;
    clear temp_out;
    
    temp_targ = find(targets == ix);
    temp_out = find(outputs == ix);
    
    clear match;
    match =  intersect(temp_targ,temp_out);
    
    prec_sum =  prec_sum+ length(match)/y;
    
    rec_sum =  rec_sum+length(match)/hx;
end

metrics.precision = prec_sum/M;
metrics.recall = rec_sum/M;
metrics.F1 =  2*(metrics.precision)*metrics.recall/(metrics.precision+metrics.recall);
return;
end