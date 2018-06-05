%%---------------------------------------------------%%
% HDM05 actions classification with knn and 10-fold crossvalidation
%%---------------------------------------------------%%

close all;
clear;

warning('off','all');
load('./initdata/chen_all_actions_joint_positions.mat');

%% 1. Select a dataset
% % All HDM05
% allActionsIndx = [1 4 5 6 7 8 9 10 11 12 14 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 34 35 36 37 39 38 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 75 74 76 77 78 80 79 81 82 83 84 85 86 87 88 90 89 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 111 110 112 113 114 115 116 117 119 118 120 121 123 122 124 125 127 126 129 130 2 3 33 128]; %list of action ids
% class = [1 2 2 3 3 4 5 6 7 8 8 9 9 10 11 12 13 14 15 15 15 16 16 16 17 17 17 18 18 19 19 20 20 21 21 22 22 23 24 24 25 25 26 26 27 27 28 28 29 30 30 31 31 32 32 33 33 34 34 35 35 36 36 37 37 38 38 39 39 40 40 41 42 42 43 43 44 44 45 46 47 48 49 49 50 50 51 51 52 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 68 69 69 70 70 71 71 72 72 73 73 74 74 75 75 76 76 77 78 78 1 79 80 77];  %labels of the classes


% % Chen dataset 65
%allActionsIndx = 1:130;
%class = [ 1 1 1 2 2 3 3 4 5 6 7 8 8 8 8 9 10 11 12 13 14 14 14 15 15 15 16 16 16 ...
%           17 17 18 18 18 18 18 19 19 19 19 20 21 21 22 22 23 23 24 24 25 25 26 ...
%           27 27 28 28 29 29 30 30 31 31 32 32 33 33 34 34 35 35 36 36 37 37 37 37 37 ...
%           38 38 38 38 39 40 41 42 43 43 44 44 44 44 45 45 46 47 48 49 50 51 52 53 ...
%           54 55 55 56 56 57 58 59 59 59 59 60 60 61 61 62 62 62 62 63 63 63 63 ...
%           64 64 64 64 65 65];
      
% % Chen dataset 40
allActionsIndx = [ 1 7 10 14 20 21 24 27 30 41 43 45 46 49 51 54 56 58 60 61 ...
                   65 69 82 83 84 85 86 93 94 96 98 99 100 101 103 104 107 108 109 129];
class = 1:40;
      
%       
% Leighteley dataset
% allActionsIndx = [4:7 24:29 41 44:51 53:60 73:77  82:85 92:93 109:130];
% class = [ones(1,length(4:7)) 2*ones(1,length(24:29)) 3 4*ones(1,length(44:51)) ...
%          5*ones(1,length(53:60)) 6*ones(1,length(73:77)) 7*ones(1,length(82:85)) ...
%          8*ones(1,length(92:93)) 9*8*ones(1,length(109:130))];

%class = 1:9;
%allActionsIndx = [4 24 41 44 53 73  82 92 109];
%allActionsIndx = [4 22 41 45 54 73  82 92 109];      

% % Ofli dataset
% allActionsIndx = [8 12 17 21 32 41 42 44 52 61 82 88 92 96 101 102]; 
% class = 1:16;
% % Repetitions for each subject - starting index
% subjectRepetitionsIndx = [ 1 7 13 19 27; ...
%                            1 7 13 19 21; ...
%                            1 5 11 17 23; ...
%                            1 13 22 31 34;...
%                            1 4 7 9 12; ...
%                            1 5 8 11 12;...
%                            1 13 25 37 41;...
%                            1 7 13 19 25; ...
%                            1 5 11 15 19; ...
%                            1 5 9 11 14; ...
%                            1 5 11 15 19; ...
%                            1 4 6 10 13; ...
%                            1 13 25 37 41; ...
%                            1 4 10 13 15; ...
%                            1 4 7 10 12; ...
%                            1 4 7 10 12];

% Skeletal Quad dataset
% allActionsIndx = [8 12 17 21 32 41 42 44 52 61 82 88 92 96 101 102]; 
% class = 1:16;
% % Repetitions for each subject - starting index
% subjectRepetitionsIndx = [ 1 7 13 19 27; ... % 1
%                            1 7 13 19 21; ... % 2
%                            1 5 11 17 23; ... % 3
%                            1 13 22 31 34;... % 4
%                            1 4 7 9 12; ...   % 5 
%                            1 5 8 11 12;...   % 6 
%                            1 13 25 37 41;... % 7
%                            1 7 13 19 25; ... % 8
%                            1 5 11 15 19; ... % 9
%                            1 5 9 11 14; ...  % 10
%                            1 5 11 15 19; ... 
%                            1 4 6 10 13; ...
%                            1 13 25 37 41; ...
%                            1 4 10 13 15; ...
%                            1 4 7 10 12; ...
%                            1 4 7 10 12];
% 
% 
% skelQuadActions = [1 2 3 4 5 8 9 10 12 13 15];
%  
% allActionsIndx = allActionsIndx(skelQuadActions);
% subjectRepetitionsIndx = subjectRepetitionsIndx(skelQuadActions,:);
      
%% 2. Compute FADE, UFADE or SVD descriptor for all actions
descriptor = 'fade'; % 'ufade' % 'svd'

% (U)FADE parameters
f_th = 10;  % Cut at 10Hz
f_s  = 60;  % Sampling frequency
K    = 500; % Desired dimensionality

actionDescriptors = []; 
actionLabels = [];
MIJ = 30;
for m=1:1
    tic;
    actionDescriptors = zeros(791,60);
    actionLabels = zeros(791,1);
    iter = 1;
for i = 1:length(allActionsIndx)
    for y = 1:size(normalized_actions_struct{allActionsIndx(i),2},1)
        angles= normalized_actions_struct{allActionsIndx(i),2}{y,2};

%     for y = 1:length(allData(allActionsIndx(i)).jointPositions)
%         angles=allData(allActionsIndx(i)).jointPositions{y};
%         
%                
      angles = angles - repmat(mean(angles),size(angles,1),1);
        
%        curr_var = var(angles);        
%        [~,sortIndex] = sort(curr_var','descend');
%         
%        angles(:,sortIndex(MIJ+1:end)) = 0;
                     
%         if strcmp(descriptor, 'fade')
%             actionDescriptors = [actionDescriptors; fade(angles, f_th, K, f_s)]; 
%         elseif strcmp(descriptor, 'ufade')
%             actionDescriptors = [actionDescriptors; ufade(angles, f_th, K, f_s)]; 
%         elseif strcmp(descriptor, 'svd')
%              [~, ~, v1] = svd(angles);
%              actionDescriptors = [actionDescriptors; v1(:,1)'];
%         end
%         actionLabels = [actionLabels; class(i)];


        if strcmp(descriptor, 'fade')
            actionDescriptors(iter,:) = fade(angles, f_th, K, f_s); 
        elseif strcmp(descriptor, 'ufade')
            actionDescriptors(iter,:) = ufade(angles, f_th, K, f_s); 
        elseif strcmp(descriptor, 'svd')
             [~, ~, v1] = svd(angles);
             actionDescriptors(iter,:) = v1(:,1)';
        end
        actionLabels(iter) = class(i);
        iter = iter+1;

    end
end 
elTime(m) = toc;
end


% % Uncomment for cross-subjects tests
% trainInd = [];
% testInd  = [];
% testIter = 1;
% for i = 1:length(allActionsIndx)
% %     for j=1:size(normalized_actions_struct{allActionsIndx(i),2},1)     
%     for j = 1:length(allData(allActionsIndx(i)).jointPositions)
% 
%       % if(j<subjectRepetitionsIndx(i,2) || j>=subjectRepetitionsIndx(i,4)) % Train
%         %if(j<subjectRepetitionsIndx(i,1) || j>=subjectRepetitionsIndx(i,3)) % Best for FADE (Ofli/Skeleton quad)
%         if(j<subjectRepetitionsIndx(i,4)) % Best for UFADE (Ofli/Skeleton quad)
%        % if(j>=subjectRepetitionsIndx(i,3)) % Train
%             trainInd = [trainInd; testIter];
%         else % Test
%             testInd = [testInd; testIter];
%         end
%         testIter = testIter + 1;
%     end
% end

%% 3. Train the classifier (kNN)

numOfNN = 1;
Kfold = 10;
timeEl = [];

mdl = fitcknn(actionDescriptors,actionLabels,'NumNeighbors',1, 'Distance', 'cityblock','standardize',0);
cvmdl = crossval(mdl,'Kfold',Kfold);
kloss = kfoldLoss(cvmdl);
% 
disp(['Accuracy: ' num2str(1-kloss)]);

% mdl = fitcknn(actionDescriptors,actionLabels,'NumNeighbors',1, 'Distance', 'cityblock','standardize',0);
% cvmdl = crossval(mdl,'Kfold',Kfold);
% kloss = kfoldLoss(cvmdl);
% % 
% disp(['Accuracy: ' num2str(1-kloss)]);
% 
% cp = cvpartition(actionLabels,'k',10); % Stratified cross-validation
% order = unique(actionLabels); 
% f = @(xtr, ytr, xte,yte)confusionmat(yte,...
%   predict(fitcknn(xtr,ytr,'NumNeighbors',1, 'Distance', 'cityblock','standardize',0), xte),'order',order);
% % 
% % cvmdl = crossval(mdl,actionDescriptors,actionLabels,'Kfold',Kfold,'partition',cp);
% cfMat = crossval(f,actionDescriptors,actionLabels,'partition',cp);

% for k=1:size(cfMat,1)
%     Conf{k} = reshape(cfMat(k,:),65,65);
% 
%     for j=1:size(Conf{k},1)
%         Conf{k}(j,:) = Conf{k}(j,:)./sum(Conf{k}(j,:));
%     end
%     sum(diag(Conf{k}))/65
% end

if(exist('testInd'))
    tic;
    mdl = fitcknn(actionDescriptors(trainInd,:),actionLabels(trainInd,:),'NumNeighbors',numOfNN, 'Distance', 'cityblock');
    predictions = predict(mdl, actionDescriptors(testInd,:));
    timeEl = toc;
    
    Conf = confusionmat(actionLabels(testInd,:), predictions);
    for j=1:size(Conf,1)
        precision(j) = Conf(j,j) / sum(Conf(:,j));
        recall(j) = Conf(j,j) / sum(Conf(j,:));
    end
    accuracy = sum(diag(Conf))/length(testInd);
    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    
    % Normalize
    for i=1:length(allActionsIndx)
        Conf(i,:) = Conf(i,:)./sum(Conf(i,:));
    end
    
    figure(1)

    imagesc(Conf)
    colormap(flipud(gray))
    colorbar;
    caxis([0,1])  
    
    disp(['Classification results using ' descriptor]);
    disp(['Accuracy ' num2str(numOfNN) '-NN: ' num2str(accuracy)])
    disp(['Precision ' num2str(numOfNN) '-NN: ' num2str(mean(precision))])
    disp(['Recall ' num2str(numOfNN) '-NN: ' num2str(mean(recall))])
    disp(['Classification time ' num2str(numOfNN) '-NN: ' num2str(timeEl) ' s'])
else
    crossValIndx = crossvalind('Kfold', length(actionLabels), Kfold);
    accuracy = zeros(1,Kfold);
    recall = zeros(1,Kfold);
    precision = zeros(1,Kfold);
    for i = 1:Kfold
        tic;
        testInd = sort(find(crossValIndx == i),'ascend');
        trainInd = 1:length(actionLabels);
        trainInd(testInd) = [];

        mdl = fitcknn(actionDescriptors(trainInd,:),actionLabels(trainInd,:),'NumNeighbors',numOfNN, 'Distance', 'cityblock');
        predictions = predict(mdl, actionDescriptors(testInd,:));
        
        % NON usare con uFADE, si impalla il PC 
        %predictions = multisvm(actionDescriptors(trainInd,:),actionLabels(trainInd),actionDescriptors(testInd,:));
        timeEl(i) = toc;

        Conf{i} = confusionmat(actionLabels(testInd,:),predictions);

        for j=1:size(Conf{i},1)
            tmpPrecision(j) = Conf{i}(j,j) / sum(Conf{i}(:,j));
            tmpRecall(j) = Conf{i}(j,j) / sum(Conf{i}(j,:));
        end
        accuracy(i) = sum(diag(Conf{i}))/length(testInd);
        tmpPrecision(isnan(tmpPrecision)) = 0;
        precision(i) = mean(tmpPrecision);
        tmpRecall(isnan(tmpRecall)) = 0;
        recall(i) = mean(tmpRecall);
        
        for j=1:size(Conf{i},1)
            Conf{i}(j,:) = Conf{i}(j,:)./sum(Conf{i}(j,:));
        end
    end
    disp(['Classification results using ' descriptor]);
    disp(['Accuracy ' num2str(numOfNN) '-NN: ' num2str(mean(accuracy)) ' +- ' num2str(std(accuracy))])
    disp(['Precision ' num2str(numOfNN) '-NN: ' num2str(mean(precision)) ' +- ' num2str(std(precision)) ])
    disp(['Recall ' num2str(numOfNN) '-NN: ' num2str(mean(recall)) ' +- ' num2str(std(recall))])
    disp(['Classification time ' num2str(numOfNN) '-NN: ' num2str(mean(timeEl)) ' +- ' num2str(std(timeEl)) ' s'])
end
mdl = fitcknn(actionDescriptors,actionLabels,'NumNeighbors',1, 'Distance', 'cityblock','standardize',0);
cvmdl = crossval(mdl,'Kfold',Kfold);
kloss = kfoldLoss(cvmdl);
% 
disp(['Accuracy: ' num2str(1-kloss)]);
% 
% predictions = kfoldPredict(cvmdl);
% confmat = confusionmat(actionLabels,predictions);   
% 
%    % Normalize
%     for i=1:65
%         confmat(i,:) = confmat(i,:)./sum(confmat(i,:));
%     end
% % Scale each row of the confusion matrix to sum up to 1
% confmat = confmat./repmat(sum(confmat,2),1,size(confmat,2));




%% Leighteley dataset
%allActionsIndx = [4:7 24:29 41 44:51 53:60 73:77  82:85 92:93 109:130];
% allActionsIndx = [4 24 41 44 53 73  82 92 109];
% allActionsIndx = [4 22 41 45 54 73  82 92 109];
% class = 1:9;

%number_of_actions = [4 3 3 3 3 6 4 4 4 6 3 6 3 3 4 4 4 3 12 4 4 11 4 4 12 4 4 2 1 3 3 3 3 2 2 2 2 4 12 3 6 3 6 3 6 3 6 3 4 6 3 6 3 6 3 6 3 4 4 4 4 4 4 4 4 4 4 4 4 3 2 3 3 3 3 3 3 3 3 4 3 4 9 3 3 3 3 3 12 3 3 7 3 4 4 4 4 3 3 3 3 3 3 1 14 6 3 6 3 3 3 3 3 2 2 3 3 3 3 3 3 1 1 1 3 3];
%number_of_actions = 100*ones(1,length(allActionsIndx)); % Use all actions

% Select actions subset
% simMat = w;
%newLabels = [];
%indxAll = [];
%testIter = 1;
% test = [];
% for i=1:length(allActionsIndx)
%     if number_of_actions(i) == 0
%         continue;
%     end
%    
%     indx = find(true_labels==allActionsIndx(i));    
%     if(number_of_actions(i)<=length(indx))
%         indxAll = [indxAll; indx(1:number_of_actions(i))];
%         newLabels(end+1:end+number_of_actions(i),1) = class(i);
%     else % use all actions
%         indxAll = [indxAll; indx];
%         newLabels(end+1:end+length(indx),1) = class(i);
%     end
% %     
% %     for j=1:length(indx)       
% %        % if(j<subjectRepetitionsIndx(i,2) || j>=subjectRepetitionsIndx(i,4)) % Train
% %         %if(j<subjectRepetitionsIndx(i,4)) % Train
% %         if(j>=subjectRepetitionsIndx(i,3)) % Train
% %             testIter = testIter + 1;
% %         else % Test
% %             test = [test; testIter];
% %             testIter = testIter + 1;
% %         end
% %     end
% 
% %% Leighteley dataset
% %     for j=1:length(indx)       
% %        % if(j<subjectRepetitionsIndx(i,2) || j>=subjectRepetitionsIndx(i,4)) % Train
% %         %if(j<subjectRepetitionsIndx(i,4)) % Train
% %         if(j==1) % Train
% %             testIter = testIter + 1;
% %         else % Test
% %             test = [test; testIter];
% %             testIter = testIter + 1;
% %         end
% %     end
%end

