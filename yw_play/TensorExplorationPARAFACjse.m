% Directory (because this code is long and rambly)

%% cleaning and closing
clear all;
close all;

%% load dataset

% [num,txt,raw] = xlsread('Copy of ANTHAfinal.xlsx');

%{
%% rearranging binary variables
% changing the binary variables so they're not ridiculous and nonsensical

% feeding number - changing 90 (unknown) to nan
% mracen = 1 - black, 0 - white
% sesn = 1 = low, 2 - lower-middle, 3 - middle, 4 - upper-middle, 5 - upper

[r,c] = size(num);

% feeding number
num(num(:,14) == 90,14) = nan; % unknown

% mracen
num(num(:,22) == 5,22) = 1; % white
num(num(:,22) == 3,22) = 0; % black

% sesn
% low to high classes
num(num(:,30) == 25,30) = 1;
num(num(:,30) == 38,30) = 2;
num(num(:,30) == 63,30) = 4;
num(num(:,30) == 75,30) = 5;
num(num(:,30) == 50,30) = 3;

%% Fixing APGAR
% some apgar scores were way above the average (in the 20s), so this is to
% fix that

for k = 1:size(num,1) % going through subjects
   if num(k,19) >= 20 % if apgar1 >= 20
       num(k,19) = num(k,19)-20;
   end
   if num(k,20) >= 20 % if apgar5 >= 20
       num(k,20) = num(k,20)-20;
   end
      
end

%% SGA kids

SGA = zeros(38465,2); % preallocate
[U,ia,ic] = unique(num(:,2),'first'); % find unique subjects
birthwaz = num(ia,7); % birth waz for unique subjects
SGA(:,1) = U; % subjects
SGA(:,2) = zeros(size(U));
SGA(birthwaz <=-2.33,2) = 1; % SGA kids

SGA_kids = SGA(SGA(:,2)==1,1); % SGA kids subj ids

%% Creating a matrix of characteristics
%{
columns, in order: (copy of antha)
 - 1 - subject
 - 2 - siteid 8 - break up
 - 3 - sexn 9
 - 4 - feedingn 10
 - 5 - gagebrth 11
 - 6 - birthwt 12
 - 7 - birthlen 13
 - 8 - apgar1 14
 - 9 - apgar5 15
 - 10 - mage 16
 - 11 - mracen 17
 - 12 - mmaritn 18
 - 13 - mcignum 19
 - 14 - parity 20
 - 15 - gravida 21
 - 16 - meducyrs 22
 - 17 - sesn 23
 - 18 - geniq
 - 19 - sysbp
 - 20 - diabp
 - 21 - SGA
%}

% copy of antha
[~,ia,ic] = unique(num(:,2),'last'); % unique subjects
chara = [num(ia,[2 11:12 14 16:22 24 26:30 32:34]) SGA(:,2)];
%   characteristics
C = setdiff(1:38508,chara(:,1)); % missing subj ids

%% uniqueness

U = unique(num(:,2)); % unique subjects
count = histc(num(:,2),U); % counts for subjects (# of time points)

%% Creating a tensor

gates_tensor = NaN(38508,20,5);
% subject features time
% if i want to use mother's characteristics with siteid condensed - 21
% if i want to use mother's characteristics with siteid broken - 33
% without siteid - 20
% if i don't - 7
time_pt = size(gates_tensor,3); % number of time points

% normum = num(:,4:6);

% normalizing growth characteristics by column
normum = zeros(size(num,1),3);
for k = 1:3
    avg = nanmean(num(:,k+3));
    stdev = nanstd(num(:,k+3));
    normum(:,k) = (num(:,k+3) - avg)/stdev;
end

% normalizing mother characteristics by column
incl_feat = num(:,[12 14 16 17 18 21 22 24 26 27 28 29 30]);% 35:46]);
norming = zeros(size(incl_feat));
for k = 1:size(incl_feat,2)
   meaning = nanmean(incl_feat(:,k));
   stdev = nanstd(incl_feat(:,k));
   norming(:,k) = (incl_feat(:,k) - meaning)/stdev;
end
% norming = incl_feat;

t = 1; % time counter
for k = [1 123 366 1462 2558]
    logical = (num(:,3) == k);
    % if i want to use mother's characteristics
    gates_tensor(num(logical,2),:,t) = [normum(logical,:) num(logical,[7:10]) norming(logical,:)];
    % if i don't
%     gates_tensor(num(logical,2),:,t) = [normum(logical,:) num(logical,7:10)];
%     gates_tensor(any(isnan(gates_tensor(:,1:6,t))),:,:) =[];
    t = t + 1; % increment time counter
end

% normalizing growth characteristics by subject
for k = 1:size(gates_tensor,1)
    for p = 1:3
        avg = nanmean(gates_tensor(k,p,:));
        stdev = nanstd(gates_tensor(k,p,:));
        norma = (gates_tensor(k,p,:)-avg)/stdev;
        gates_tensor(k,p,:) = norma;
    end
end
gates_tensor(C,:,:) = [];
% gates_tensor(count~=5,:,:)=[]; % only include those with all time
%    points, doesn't include whz

% gates_tensor = gates_tensor(SGA(:,2)==1,:,:); % only SGA kids tensor
% [~,ia]=unique(num(:,2));
% gates_tensor = gates_tensor(num(ia,16)>=37,:,:); % taking away preterm
%}

% gates_SGA = gates_tensor(birthwaz <=-2.33,:,:);

%% Yener Tensor
% tensor made for professor yener

load gatesmine.mat % tensor called "gates_tensor"

% i need whz for this - gates.mat doesn't have whz
%{
holding = num(ismember(num(:,2),gates_tensor(:,1,1)),:);
count = 0;
for k = [1 123 366 1462 2558]
    count = count + 1;
gates_tensor(:,23,count) = holding(holding(:,3)==k,10);
end
%}

% replacement of variables
%{
for k = 1:size(gates_tensor,3)
%{
% when using averages for the binary varibles - a little wonky
% feeding number
gates_tensor(gates_tensor(:,9,k)>3,9,k) = nan;

% mracen
gates_tensor(gates_tensor(:,14,k) ==3,14,k) = 0;
gates_tensor(gates_tensor(:,14,k) ~= 3,14,k) = 1;


% sesn
gates_tensor(gates_tensor(:,19,k) >= 75,19,k) = 5;
gates_tensor(gates_tensor(:,19,k) >= 63,19,k) = 4;
gates_tensor(gates_tensor(:,19,k) >= 50,19,k) = 3;
gates_tensor(gates_tensor(:,19,k) >= 38,19,k) = 2;
gates_tensor(gates_tensor(:,19,k) >= 25,19,k) = 1;
%}
% when using binary counts for variables
% mracen
gates_tensor(gates_tensor(:,14,k) == 5,14,k) = 1;
gates_tensor(gates_tensor(:,14,k) == 3,14,k) = 0;

% sesn
gates_tensor(gates_tensor(:,19,k) == 25,19,k) = 1;
gates_tensor(gates_tensor(:,19,k) == 38,19,k) = 2;
gates_tensor(gates_tensor(:,19,k) == 63,19,k) = 4;
gates_tensor(gates_tensor(:,19,k) == 75,19,k) = 5;
gates_tensor(gates_tensor(:,19,k) == 50,19,k) = 3;


end
%}
gates_keep = gates_tensor; % storing the original tensor
gates_tensor = gates_tensor(:,[2:7 9:20],1:5); % not including iq, bp & subj id & whz

% normalization
%{
for k = 8:size(gates_tensor,2) % normalizing characteristics  
   for b = 1:size(gates_tensor,3)
       meaning = nanmean(gates_tensor(:,k,1));
       stdev = nanstd(gates_tensor(:,k,1));
       gates_tensor(:,k,b) = (gates_tensor(:,k,b) - meaning)/stdev;
   end
end

for k = 1:size(gates_tensor,1) % normalizing height, weight, bmi
    for p = 1:3
        avg = nanmean(gates_tensor(k,p,:));
        stdev = nanstd(gates_tensor(k,p,:));
        gates_tensor(k,p,:) = (gates_tensor(k,p,:)-avg)/stdev;
    end
end
%}
time_pt = size(gates_tensor,3); % amount of time points
chara = gates_keep(:,:,1); % characteristics at the first time point

%% Unfolding tensor

% load yenerscores.mat
% [yscores] =xlsread('Childloadings.xlsx');
% gates_mtx = [gates_keep(:,1,1) gates_tensor(:,[1:6],1) gates_tensor(:,[1:6],2)...
%     gates_tensor(:,[1:6],3) gates_tensor(:,[1:6],4) gates_tensor(:,[1:6],5)...
%     gates_tensor(:,[7:end],1) chara(:,21:23) yener_scores(:,1) yscores A idx];
%% SGA Kids - Yener Tensor

% SGA kids
U = gates_keep(:,1,1); % subjects
birthwaz = gates_keep(:,5,1); % birth waz
SGA(:,1) = U; % subjects
SGA(:,2) = zeros(size(U)); % preallocating
SGA(birthwaz <=-2.33,2) = 1; % SGA kids
chara = [chara SGA(:,2)]; % add that characteristic

% gates_SGA = gates_keep(birthwaz <=-2.33,1:19,:);
%}
%% Clustering the Clusters - Tensor Creation

round2 = 0;

%% Choosing the amount of PARAFAC components

% let's try centering?
[gates_tensornew,means,scales]=nprocess(gates_tensor,[0 0 0],[0 0 0]);
tensoring = gates_tensornew;

err = zeros(1,3); % initializing the SSE
consistency = zeros(1,2); % initializing the consistency
for k = 1:3
    [factors,~,err(k),~] = parafac(gates_tensor,k);
    consistency(k) = corcond(tensoring,factors);
    
end

% scree-plot
figure
plot(1:3,err);
title('Scree-Plot for PARAFAC model');
xlabel('Number of Components');
ylabel('SSE');
% 2 is best

% core consistency plot
figure
plot(1:3,consistency);
title('Core Consistency Plot');
xlabel('Number of Components');
ylabel('CONCORDIA');
% displays that 2 is best


%% PARAFAC

% create parafac model
[factors,it,err,corcondia,output] = parafac(tensoring,2,[0 0 0 0 10 0]);
% if options(3) = 2, MATLAB runs out of memory at the 3rd plot :(
[A,B,C] = fac2let(factors); % get loadings

%% Variance Explained
% not entirely sure if this works - i often ignore this plot
%{
var_explained  = cumsum(output)/sum(output);
figure
subplot(2,1,1)
bar(output);
title('Eigenvalues of Data')
suptitle('Explained Variance')
subplot(2,1,2)
bar(var_explained)
set(gca,'YTick',0:0.1:1)
set(gca,'XTick',0:5:69)
title('Variance Explained for Data')
%}

%% Comparing Original to PARAFAC modeling

time_points = [1 123 366 1462 2558];

% displaying original data
figure % plot
for k = 1:time_pt % for each of the time points
    subplot(3,2,k),mesh(tensoring(:,:,k));
    title(sprintf('Time Point %g',time_points(k)));
    ylabel('Subject');
    xlabel('Feature');
end
suptitle('Original Data');

% estimate model
[gates_model] = nmodel(factors,[],0);
figure % plot model
for k = 1:time_pt % for each of the time points
    subplot(3,2,k),mesh(gates_model(:,:,k));
    title(sprintf('Time Point %g',time_points(k)));
    ylabel('Subject');
    xlabel('Feature');
end
suptitle('Modeled Data');

% error in model: residual
res = tensoring - gates_model;

figure % plot residuals
for k = 1:time_pt % for each of the time points
    subplot(3,2,k),mesh(res(:,:,k));
    title(sprintf('Time Point %g',time_points(k)));
    ylabel('Subject');
    xlabel('Feature');
end
suptitle('Residuals');

%% Components/Loadings

figure
plotfac(factors);
suptitle('Component Matrices for Each Mode');

%% Loadings/Scores

% subjects
figure
subplot(2,2,1),plot(A(:,1),A(:,2),'.');
title('Mode 1')
xlabel('Score 1')
ylabel('Score 2')
%The text command print 
% mode1_labels = num(ia,2);
% for i = 1:size(A,1)
%     cc = text(A(i,1),A(i,2),num2str(mode1_labels(i)));
% end

% features
% mode2_labels = {'weight','height','bmi','waz','haz','baz','whz','sexn',...
%     'feedingn','gagebirth','birthwt','birthlen','apgar1','apgar5','mage','mracen',...
%     'mmaritn','mcignum','parity','gravida','meducyrs','sesn'}; % former
mode2_labels = {'weight','height','bmi','waz','haz','baz','whz','sexn',...
    'feedingn','gagebirth','apgar1','apgar5','mage','mracen',...
    'mcignum','parity','gravida','meducyrs','sesn'}; % yener
subplot(2,2,2),plot(B(:,1),B(:,2),'.');
title('Mode 2')
xlabel('Score 1')
ylabel('Score 2')
for i = 1:size(B,1)
    cc = text(B(i,1),B(i,2),mode2_labels{i});
end

% time
mode3_labels = {'1','123','366','1462','2558'};
subplot(2,2,3),plot(C(:,1),C(:,2),'.');
title('Mode 3')
xlabel('Score 1')
ylabel('Score 2')
for i = 1:size(C,1)
    cc = text(C(i,1),C(i,2),mode3_labels{i});
end

%% Creating a gradient for characteristics

% original tensor
%{
figure
[~,ia,ic] = unique(num(:,2),'last'); % getting the iqs
iqs = num(ia,32);
hist(iqs)
% iqs(iqs<80,:) = NaN;
% iqs(iqs>120,:) = NaN;

figure
% colors = colormap(jet(max(num(:,32))-min(num(:,32)+1)));
scatter(A(:,1),A(:,2), 5, iqs);
set(gca,'CLim',[min(iqs) max(iqs)]);
colorbar
title('Components Colored by IQ')
xlabel('Component 1')
ylabel('Component 2')

figure
scatter(A(:,1),A(:,2), 5, chara(:,6));
set(gca,'CLim',[min(chara(:,6)) max(chara(:,6))]);
colorbar
title('Components Colored by Birth Weight')
xlabel('Component 1')
ylabel('Component 2')

figure
scatter(A(:,1),A(:,2), 5, chara(:,7));
set(gca,'CLim',[min(chara(:,7)) max(chara(:,7))]);
colorbar
title('Components Colored by Birth Length')
xlabel('Component 1')
ylabel('Component 2')

labels = {'siteid','sexn',...
    'feedingn','gagebirth','birthwt','birthlen','apgar1','apgar5','mage','mracen',...
    'mmaritn','mcignum','parity','gravida','meducyrs','sesn','geniq','sysbp','diabp'};

k1 = [1 5 9 13 17];
k2 = [4 8 12 16 19];

for p = 1:5
figure
c=1;
for k = k1(p):k2(p)
subplot(2,2,c)
scatter(A(:,1),A(:,2), 5, chara(:,k+1));
set(gca,'CLim',[min(chara(:,k+1)) max(chara(:,k+1))]);
title(sprintf('%s',labels{k}))
ylabel('Second Component')
xlabel('First Component')
c = c+1;
end
end
%}

% yener tensor
%{
mode2_labels = {'sexn',...
    'feedingn','gagebirth','apgar1','apgar5','mage','mracen',...
    'mcignum','parity','gravida','meducyrs','sesn','geniq','sysbp','diabp'};
iqs = chara(:,20);
for k = 9:size(gates_keep,2)
    figure
    scatter(A(:,1),A(:,2), 5, gates_keep(:,k,1));
    set(gca,'CLim',[min(gates_keep(:,k,1)) max(gates_keep(:,k,1))]);
    title(sprintf('%g: %s',k,mode2_labels{k-8}))
    colorbar;
ylabel('Second Component')
xlabel('First Component')
end
%}     

%% Looking at where SGA kids fall
% on the scatter plot model

figure
% colors = colormap(jet(max(num(:,32))-min(num(:,32)+1)));
gscatter(A(:,1),A(:,2), SGA(:,2));
% set(gca,'CLim',[min(iqs) max(iqs)]);
% colorbar
title('Components Colored by SGA')
xlabel('Component 1')
ylabel('Component 2')

%% Imagesc
% heat maps of each mode

figure
subplot(2,2,1),imagesc(A),colorbar
title('Mode 1')
ylabel('Subjects')
xlabel('Components')

subplot(2,2,2),imagesc(B),colorbar
title('Mode 2')
ylabel('Features')
xlabel('Components')

subplot(2,2,3),imagesc(C),colorbar
title('Mode 3')
ylabel('Time Points')
xlabel('Components')

%% Fuzzy C-Means Clustering
% [yener_scores] = xlsread('Childloadings.xlsx');

nC = 5;
analyze = gates_mtx(:,47);
% analyze = yener_scores;
[VC,UF,~,G,idx] = FuzzyCMeans(analyze,(1:38508)',nC);
% idx = kmeans(analyze,nC,'replicates',10)

%% Silhouette Plot

% visualizing fuzzy c-means
% silhouette graph
figure
silhouette(analyze,idx);
% set(get(gca,'Children'),'FaceColor',[.8 .8 1])
title ('Silhouette Values per Cluster (Fuzzy C-Means)')
xlabel('Silhouette Value')
ylabel('Cluster')

%% PCA (colored clusters)

% cluster colored PCA plot
figure
hold on
grid off
gscatter(analyze(:,1),analyze(:,1),idx)
title('Scores Scatter Plot with Colored Clusters (Fuzzy C-means)');
xlabel('Component 1');
ylabel('Component 2');
hold off

%% Determining Characteristics of Clusters
% finding the averages for each characteristic

avg_char = zeros(nC,size(chara,2)); % preallocating
for k = 1:nC
   avg_char(k,:) = nanmean(chara(idx == k,:)); % average values
end

%% Significance testing

combo = combnk(1:nC,2); % the different combinations of cluster testing
tests = zeros(size(combo,1),size(chara,2)+1); % is 95% test sucessful?
pval = zeros(size(combo,1),size(chara,2)+1); % pvalue corresponding to ttest

testcount = 1;
for k = combo' % iterating through different combinations of clusters
    for p = 1:size(chara,2); % iterating through different criteria
        group1 = chara(idx==k(1),p); % first cluster
        group1(isnan(group1)) = []; % take away any nan values
        group2 = chara(idx==k(2),p); % second cluster
        group2(isnan(group2)) = []; % take away any nan values
        % difference in means test
        [tests(testcount,p+1),pval(testcount,p+1)] = ttest2(group1,group2);
    end
    testcount = testcount + 1;
end

tests(:,1:2) = combo;
pval(:,1:2) = combo;

%% Tabulate Binary Variables
% counting binary variables

%{
tab_labels = {'siteid','sexn','feedingn','mracen','mmaritn','sesn', 'SGA'};
for k = 1:nC
    fprintf('Cluster %g\n',k);
    p = 1;
    for c = [2 3 4 11 12 17 21]
        fprintf('\n\t%s:\n\t',tab_labels{p})
        tabulate(chara(idx==k,c));
        p = p + 1;
    end
end
%}

%% IQ Distributions
% histograms for the iqs of each cluster

%{
% normal
if round2 == 0
figure
for k = 1:nC
    subplot(nC,1,k)
    hist(chara(idx == k,18),16)
    xlim([0 160])
    title(sprintf('Cluster %g',k))
    xlabel('IQ')
    ylabel('Number of Subjects')
    axis tight
end
suptitle('IQ Distributions')
else
    
figure
for k = 1:nC
    subplot(ceil(sqrt(nC)),2,k)
    hist(chara(idxc == k,18),16)
    xlim([0 160])
    title(sprintf('Cluster %g',k))
    xlabel('IQ')
    ylabel('Number of Subjects')
    axis tight
end
suptitle('IQ Distributions')
end
%}

% yener
figure
j = 1;
for k = [5 4 2 1 3]
    subplot(nC,1,j)
    hist(chara(idx == k,20),7)
    xlim([0 100])
    title(sprintf('Cluster %g',k))
    xlabel(sprintf('Gestational Age at Birth'))
    ylabel(sprintf('Number of Subjects'))
%     axis tight
    j=j+1;
end
% suptitle('IQ Distributions')

%% Characteristics vs First Component


[sorting,order] = sortrows(A,2);
labels = {'siteid','sexn',...
    'feedingn','gagebirth','birthwt','birthlen','apgar1','apgar5','mage','mracen',...
    'mmaritn','mcignum','parity','gravida','meducyrs','sesn','geniq','sysbp','diabp'};
%{
  %  normal
figure
for k = 1:12
subplot(3,4,k),plot(sorting(:,1),chara(order,k+1),'.');
title(sprintf('%s vs First Component',labels{k}))
ylabel(labels{k})
xlabel('First Component')
end

figure
for k = 1:7
subplot(3,3,k),plot(sorting(:,1),chara(order,k+13),'.');
title(sprintf('%s vs First Component',labels{k+12}))
ylabel(labels{k+12})
xlabel('First Component')
end
%}


    %yener
labels = {'sexn',...
    'feedingn','gagebirth','apgar1','apgar5','mage','mracen',...
    'mcignum','parity','gravida','meducyrs','sesn','geniq','sysbp','diabp','SGA'};
figure
for k = 1:12
subplot(3,4,k),plot(sorting(:,2),chara(order,k+8),'.');
title(sprintf('%s vs Second Component',labels{k}))
ylabel(labels{k})
xlabel('First Component')
end

figure
for k = 1:4
subplot(3,3,k),plot(sorting(:,2),chara(order,k+20),'.');
title(sprintf('%s vs Second Component',labels{k+12}))
ylabel(labels{k+12})
xlabel('First Component')
end
%}

%% IQ of Clusters


figure
hold on
gscatter(analyze(:,1),chara(:,21),idx);
% scatter(A(idx==1,2),chara(idx==1,21));
for k = 1:nC
    avg_iq = nanmean(chara(idx==k,21));
    avg_two = nanmean(analyze(idx==k,2));
    cc = text(avg_two,avg_iq,num2str(avg_iq),'HorizontalAlignment','center',...
        'FontWeight','bold');
end
% plot(model(:,1),yfit,'k-');
line([80 250],[85 85],'Color','k');
%gscatter(sorting(:,1),chara(order,18), SGA(ismember(SGA(:,1),chara(:,1)),2));
title(sprintf('%s vs Second Component (2 Component PARAFAC Model)','IQ'))
ylabel('IQ')
xlabel('Second Component')
hold off

%% IQ Percentages

for k = 1:nC
   k;
   sum(chara(idx==k,21)<85)/size(chara(idx==k,21),1);
end

%% Clusters: BMI over time

figure
j = 1;
for k = [3 2 5 4 1]
    y = nanmean(gates_tensor(idx==k,3,:));
    e = std(y)*ones(size(1:5));
    subplot(2,3,j)
    errorbar(1:5,y,e)
    title(sprintf('Cluster %g',k));
    ylabel('BMI')
    xlabel('Time Points')
    ylim([10 22])
    j = j+1;
end

%% Characteristics vs First Component colored by SGA
%{
[sorting,order] = sortrows(A,1);
labels = {'siteid','sexn',...
    'feedingn','gagebirth','birthwt','birthlen','apgar1','apgar5','mage','mracen',...
    'mmaritn','mcignum','parity','gravida','meducyrs','sesn','geniq','sysbp','diabp'};

figure
for k = 1:12
subplot(3,4,k),gscatter(sorting(:,1),chara(order,k+1),chara(order,end));
title(sprintf('%s vs First Component',labels{k}))
ylabel(labels{k})
xlabel('First Component')
end

figure
for k = 1:7
subplot(3,3,k),gscatter(sorting(:,1),chara(order,k+13),chara(order,end));
title(sprintf('%s vs First Component',labels{k+12}))
ylabel(labels{k+12})
xlabel('First Component')
end

figure
gscatter(sorting(:,1),chara(order,18),chara(order,end));
title(sprintf('%s vs First Component','IQ'))
ylabel('IQ')
xlabel('First Component')
%}

%% Creating a model for IQ based on PARAFAC modeling

%normal
%{
model = [A(:,1) chara(:,18) chara(:,end)];
model(isnan(model(:,2)),:) = []; % taking out nan values

p = polyfit(model(:,1),model(:,2),1); % calculating linear model
yfit = polyval(p,model(:,1));
yresid = model(:,2) - yfit;
SSresid = sum(yresid.^2);
SStotal = (length(model(:,2))-1) * var(model(:,2));
rsq = 1 - SSresid/SStotal;
%}

% yener
model = [A(:,2) chara(:,21) chara(:,end)];
model(isnan(model(:,2)),:) = []; % taking out nan values

p = polyfit(model(:,1),model(:,2),1); % calculating linear model
yfit = polyval(p,model(:,1));
yresid = model(:,2) - yfit;
SSresid = sum(yresid.^2);
SStotal = (length(model(:,2))-1) * var(model(:,2));
rsq = 1 - SSresid/SStotal;


%% Cretaing a 3D Gscatter Plot
% for a three component model

if size(analyze,2) == 3
    figure
    hold on
    % thank god for autosave
    uni = size(unique(idx),1);
    colors = colormap(hsv(uni));
    for k = 1:nC
        scatter3(analyze(idx==k,1),analyze(idx==k,2),analyze(idx==k,3),5,colors(k,:))
    end
    legend('toggle')
    grid on
    xlabel('Score 1')
    ylabel('Score 2')
    zlabel('Score 3')
end
break

%% Classification
%{
%% Random Permutations
seed = rng;

%% Create Training and Testing Sets

% DOESN'T INCLUDE WHZ
tot = [chara(:,1) A(:,1) chara(:,18) A(:,2) SGA(:,2)]; % subject id, then iqs, then scores
tot(isnan(tot(:,3)),:) = []; % taking out nan values
tot(tot(:,2)>0,:) = []; % taking out components greater than 0

thing = find(ismember(num(:,2),tot(:,1)));
subject_rows = num((ismember(num(:,2),tot(:,1))),:);
total = NaN(size(tot,1),55);
total(:,1) = tot(:,1);
total(:,50:51) = tot(:,[2 4]);

for k = 1:size(subject_rows,1)
    % STOP CARING ABOUT EFFICIENCY
    row = tot(:,1) == subject_rows(k,2);
    if subject_rows(k,3) == 1
        total(row,2:7) = subject_rows(k,4:9);
    elseif subject_rows(k,3) == 123
        total(row,8:13) = subject_rows(k,4:9);
    elseif subject_rows(k,3) == 366
        total(row,14:19) = subject_rows(k,4:9);
    elseif subject_rows(k,3) == 1462
        total(row,20:25) = subject_rows(k,4:9);
    elseif subject_rows(k,3) == 2558
        total(row,26:31) = subject_rows(k,4:9);
        total(row,32:49) = subject_rows(k,[11 12 14 16:22 24 26:30 33:34]);
        % doesn't include siteid or geniq
    end
end

% for k = 1:size(total,1)
%     for p = 2:4
%         avg = nanmean([total(k,p),total(k,p+6),total(k,p+12),total(k,p+18),total(k,p+24)]);
%         stdev = nanstd([total(k,p),total(k,p+6),total(k,p+12),total(k,p+18),total(k,p+24)]);
%         total(k,p) = (total(k,p)-avg)/stdev;
%         total(k,p+6) = (total(k,p+6)-avg)/stdev;
%         total(k,p+12) = (total(k,p+12)-avg)/stdev;
%         total(k,p+18) = (total(k,p+18)-avg)/stdev;
%         total(k,p+24) = (total(k,p+22)-avg)/stdev;
%     end
% end

% height velocities
for k = 1:size(total,1)
    for p = 2:5
    vel = total(k,p)-total(k,p-1);
    if p == 2
        total(k,52) = vel/(123-1);
    elseif p ==3
        total(k,53) = vel/(366-123);
    elseif p ==4
        total(k,54) = vel/(1462-366);
    elseif p ==5
        total(k,55) = vel/(2558-1462);
    end
    end
end


% -1 if below 85, 1 if above or equal to 85
class = tot(:,3)>=85;
class = +class;
class(class==0,:) = -1;

% SGA classes
% class = tot(:,5)==1;
% class = +class;
% class(class==0,:) = -1;


% 1 if below 10, 2 if below 50, 3 if below 90, 4 otherwise (SGA)

% create training (90%) (rounded up) and testing set (10%)
rng(seed);
split = randperm(size(total,1));

train = total(split(1:ceil(.9*size(total,1))),2:end);
trainclass = class(split(1:ceil(.9*size(total,1))),:);
test = total(split(ceil(.9*size(total,1))+1:end),2:end);
testclass = class(split(ceil(.9*size(total,1))+1:end),:);

fprintf('Total Sample Size: %g\n',size(train,1) + size(test,1));
fprintf('\tSize of Training Set: %g\n',size(train,1));
fprintf('\tSize of Testing Set: %g\n\n',size(test,1));

fprintf('Training Set:\n');
fprintf('\tNumber of Class 1: %g\n',sum(trainclass==1));
fprintf('\tNumber of Class -1: %g\n\n',sum(trainclass==-1));

fprintf('Testing Set:\n');
fprintf('\tNumber of Class 1: %g\n',sum(testclass==1));
fprintf('\tNumber of Class -1: %g\n\n',sum(testclass==-1));


%% Linear SVM

options.MaxIter = 1000000;
SVMstruct = svmtrain(train,trainclass,'kernel_function','linear','options',options);
train_predl = svmclassify(SVMstruct,train);
test_predl = svmclassify(SVMstruct,test);

% train error
train_err = sum(train_predl~=trainclass)/size(trainclass,1);
% class 1 Training error: in class 1 but predicted class -1
train_one_err = sum(trainclass(train_predl==-1)==1)/size(trainclass,1);
% class -1 Training error: in class -1 but predicted class 1
train_minus_err = sum(trainclass(train_predl==1)==-1)/size(trainclass,1);

fprintf('Linear SVM Training Error: %g\n',train_err);
fprintf('\tClass 1 Training Error: %g\n',train_one_err);
fprintf('\tClass -1 Training Error: %g\n\n',train_minus_err);

% train error
test_err = sum(test_predl~=testclass)/size(testclass,1);
% class 1 Training error: in class 1 but predicted class -1
test_one_err = sum(testclass(test_predl==-1)==1)/size(testclass,1);
% class -1 Training error: in class -1 but predicted class 1
test_minus_err = sum(testclass(test_predl==1)==-1)/size(testclass,1);

fprintf('Linear SVM Testing Error: %g\n',test_err);
fprintf('\tClass 1 Testing Error: %g\n',test_one_err);
fprintf('\tClass -1 Testing Error: %g\n\n',test_minus_err);
%% Other Types of SVM (for voting)

SVMstruct = svmtrain(train,trainclass,'kernel_function','rbf','options',options);
train_predr = svmclassify(SVMstruct,train);
test_predr = svmclassify(SVMstruct,test);

% train error
train_err = sum(train_predr~=trainclass)/size(trainclass,1);
% class 1 Training error: in class 1 but predicted class -1
train_one_err = sum(trainclass(train_predr==-1)==1)/size(trainclass,1);
% class -1 Training error: in class -1 but predicted class 1
train_minus_err = sum(trainclass(train_predr==1)==-1)/size(trainclass,1);

fprintf('RBF SVM Training Error: %g\n',train_err);
fprintf('\tClass 1 Training Error: %g\n',train_one_err);
fprintf('\tClass -1 Training Error: %g\n\n',train_minus_err);

% train error
test_err = sum(test_predr~=testclass)/size(testclass,1);
% class 1 Training error: in class 1 but predicted class -1
test_one_err = sum(testclass(test_predr==-1)==1)/size(testclass,1);
% class -1 Training error: in class -1 but predicted class 1
test_minus_err = sum(testclass(test_predr==1)==-1)/size(testclass,1);

fprintf('RBF SVM Testing Error: %g\n',test_err);
fprintf('\tClass 1 Testing Error: %g\n',test_one_err);
fprintf('\tClass -1 Testing Error: %g\n\n',test_minus_err);

% SVMstruct = svmtrain(train,trainclass,'kernel_function','quadratic','options',options);
% train_predq = svmclassify(SVMstruct,train);
% test_predq = svmclassify(SVMstruct,test);

SVMstruct = svmtrain(train,trainclass,'kernel_function','polynomial','options',options);
train_predp = svmclassify(SVMstruct,train);
test_predp = svmclassify(SVMstruct,test);

% train error
train_err = sum(train_predp~=trainclass)/size(trainclass,1);
% class 1 Training error: in class 1 but predicted class -1
train_one_err = sum(trainclass(train_predp==-1)==1)/size(trainclass,1);
% class -1 Training error: in class -1 but predicted class 1
train_minus_err = sum(trainclass(train_predp==1)==-1)/size(trainclass,1);

fprintf('Polynomial (3) SVM Training Error: %g\n',train_err);
fprintf('\tClass 1 Training Error: %g\n',train_one_err);
fprintf('\tClass -1 Training Error: %g\n\n',train_minus_err);

% train error
test_err = sum(test_predp~=testclass)/size(testclass,1);
% class 1 Training error: in class 1 but predicted class -1
test_one_err = sum(testclass(test_predp==-1)==1)/size(testclass,1);
% class -1 Training error: in class -1 but predicted class 1
test_minus_err = sum(testclass(test_predp==1)==-1)/size(testclass,1);

fprintf('Polynomial (3) SVM Testing Error: %g\n',test_err);
fprintf('\tClass 1 Testing Error: %g\n',test_one_err);
fprintf('\tClass -1 Testing Error: %g\n\n',test_minus_err);

%% Random Forest

% B = TreeBagger(100,train,trainclass,'oobpred','on');
% y = oobPredict(B);
% 
% figure
% oobErrorBaggedEnsemble = oobError(B);
% plot(oobErrorBaggedEnsemble)
% xlabel 'Number of grown trees';
% ylabel 'Out-of-bag classification error';

% ens = fitensemble(test,testclass,'LogitBoost',50,'tree');
% 
% figure
% plot(resubLoss(ens,'mode','cumulative'));
% xlabel('Number of decision trees');
% ylabel('Resubstitution error')

ctree = ClassificationTree.fit(train,trainclass); % create classification tree
% view(ctree,'mode','graph') % text description

imp = predictorImportance(ctree);
[SGA,ix] = sort(imp,'descend');
mode2_labels = {'weight 1','height 1','bmi 1','waz 1','haz 1','baz 1',...
    'weight 123','height 123','bmi 123','waz 123','haz 123','baz 123',...
    'weight 366','height 366','bmi 366','waz 366','haz 366','baz 366',...
    'weight 1462','height 1462','bmi 1462','waz 1462','haz 1462','baz 1462',...
    'weight 2558','height 2558','bmi 2558','waz 2558','haz 2558','baz 2558',...
    'siteid','sexn','feedingn','gagebirth','birthwt','birthlen','apgar1','apgar5','mage',...
    'mracen','mmaritn','mcignum','parity','gravida','meducyrs','sesn','sysbp','diabp',...
    'score1','score2','vel1','vel123','vel366','vel1462'};
importance = mode2_labels(ix);
importance(2,:) = num2cell(SGA);

[train_predc,train_scores] = predict(ctree,train);
[test_predc,test_scores] = predict(ctree,test);

% train error
train_err = sum(train_predc~=trainclass)/size(trainclass,1);
% class 1 Training error: in class 1 but predicted class -1
train_one_err = sum(trainclass(train_predc==-1)==1)/size(trainclass,1);
% class -1 Training error: in class -1 but predicted class 1
train_minus_err = sum(trainclass(train_predc==1)==-1)/size(trainclass,1);

fprintf('Classification Tree Training Error: %g\n',train_err);
fprintf('\tClass 1 Training Error: %g\n',train_one_err);
fprintf('\tClass -1 Training Error: %g\n\n',train_minus_err);

% train error
test_err = sum(test_predc~=testclass)/size(testclass,1);
% class 1 Training error: in class 1 but predicted class -1
test_one_err = sum(testclass(test_predc==-1)==1)/size(testclass,1);
% class -1 Training error: in class -1 but predicted class 1
test_minus_err = sum(testclass(test_predc==1)==-1)/size(testclass,1);

fprintf('Classification Tree Testing Error: %g\n',test_err);
fprintf('\tClass 1 Testing Error: %g\n',test_one_err);
fprintf('\tClass -1 Testing Error: %g\n\n',test_minus_err);

% ROC curves and stuff
[x,y,~,auc] = perfcurve(trainclass,train_scores(:,2),1);
figure
plot(x,y)
xlabel('False positive rate');ylabel('True positive rate')
title('ROC for classification: Training')
fprintf('Training AUC: %g\n\n',auc);

[x,y,~,auc] = perfcurve(testclass,test_scores(:,2),1);
figure
plot(x,y)
xlabel('False positive rate');ylabel('True positive rate')
title('ROC for classification: Testing')
fprintf('Testing AUC: %g\n\n',auc);

%% Fisher Classifier

[w,t]=FisherClassifier(trainclass);
format long;

% Training Error
Cp = trainclass == 1;
Cm = trainclass == -1;
% Calculate scalar projections.
Cp_proj = Cp*w;
Cm_proj = Cm*w;
%Calculate Training error 
err = mean([(Cp_proj <= t ); (Cm_proj >= t)]);
%Plot Histogram 
ClassHist(Cp_proj,Cm_proj,t,'Fisher Method Training Results: Dataset A',err);

% Testing Error
Cp = testclass == 1;
Cm = testclass == -1;
% Calculate scalar projections.
Cp_proj = Cp*w;
Cm_proj = Cm*w;
%Calculate Training error 
err = mean([(Cp_proj <= t ); (Cm_proj >= t)]);
%Plot Histogram 
ClassHist(Cp_proj,Cm_proj,t,'Fisher Method Testing Results: Dataset A',err);

%% Let's Vote

% SVMS:
% l - linear, r - rbf, q - quadratic, p - polynomial 3
% c - classification tree

train_census = [train_predl, train_predr, train_predp, train_predc];
test_census = [test_predl, test_predr, test_predp, test_predc];

train_predv = mode(train_census,2);
test_predv = mode(test_census,2);

% train error
train_err = sum(train_predv~=trainclass)/size(trainclass,1);
% class 1 Training error: in class 1 but predicted class -1
train_one_err = sum(trainclass(train_predv==-1)==1)/size(trainclass,1);
% class -1 Training error: in class -1 but predicted class 1
train_minus_err = sum(trainclass(train_predv==1)==-1)/size(trainclass,1);

fprintf('Voting Training Error: %g\n',train_err);
fprintf('\tClass 1 Training Error: %g\n',train_one_err);
fprintf('\tClass -1 Training Error: %g\n\n',train_minus_err);

% train error
test_err = sum(test_predv~=testclass)/size(testclass,1);
% class 1 Training error: in class 1 but predicted class -1
test_one_err = sum(testclass(test_predv==-1)==1)/size(testclass,1);
% class -1 Training error: in class -1 but predicted class 1
test_minus_err = sum(testclass(test_predv==1)==-1)/size(testclass,1);

fprintf('Voting Testing Error: %g\n',test_err);
fprintf('\tClass 1 Testing Error: %g\n',test_one_err);
fprintf('\tClass -1 Testing Error: %g\n\n',test_minus_err);
%}
%% Verifying Model
% THIS TAKES FOREVER TO RUN: RUN AT YOUR OWN RISK

% XvalResult = ncrossdecomp('parafac',tensoring,2,3,7,0,0);
% tucktest(tensoring,2); % this algorithm doesn't even work D:



