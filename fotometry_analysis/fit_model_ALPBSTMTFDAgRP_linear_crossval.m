%% Short explanation

% This script loads photometry data, recorded at 1Hz, from excell files in the /data folder.
% The analysis is done for five different experiments. All recordings here are from Y1R neurons in PBN.
% (AL: ad libitum fed mice, PBS: AL fed and neutral PBS odor,
% TMT: AL fed and aversive odor introduced, FD: mice are food-deprived, AgRPAL fed mice with optogenetic agrp stimulation,
% PEG: mice are thirsty).
% We fit first the licking behavior to the photometry signal. Then we fit
% both the behavior and two gaussians. We record the resulting Pearson
% coefficients. This procedure is described in the methods.

% A lot of figures are created, one showing the two fits for each mouse (32
% figures), bar charts summarizing the Pearson coefficients (2 figures),
% bar charts summarizing the time constants (2 figures).
% The time constant for a mouse is the one that maximizes the average Pearson correlation 
% for all other mice.

%% loop over all data and gather params

mice = ["1455_AL","1456_AL","1459_AL","1460_AL","1382_AL","1483_AL","1484_AL",...
    "1911_PBS","1912_PBS","1913_PBS","1914_PBS","1915_PBS","1916_PBS","1917_PBS",...
    "1911_TMT","1912_TMT","1913_TMT","1914_TMT","1915_TMT","1916_TMT","1917_TMT",...
    "1457_FD","1458_FD","1461_FD","1462_FD","1482_FD","1486_FD","1487_FD",...
    "1570_AgRP","1571_AgRP","1573_AgRP","1634_AgRP","mk546_PEG"];
exps = ["AL","PBS","TMT","FD","AgRP","PEG"];

time_constants = 1:25;

nb_AL = 7;
nb_PBS = 7;
nb_TMT = 7;
nb_FD = 7;
nb_AgRP = 4;
nb_PEG = 1;

coeffs = zeros(length(mice),size(time_constants,2));
coeffs_both = zeros(length(mice),size(time_constants,2));
lick_signal = zeros(length(mice),3600);
licks = zeros(length(mice),3600);
foto_signals = zeros(length(mice),3600);
gauss_params = zeros(length(mice),6);
gaussians = zeros(length(mice),3600);
offsets = zeros(length(mice),1);
baseline = zeros(length(mice),301);
controls = zeros(length(mice),3901);

avg_r = zeros(length(mice),size(time_constants,2));
avg_r_both = zeros(length(mice),size(time_constants,2));
options = optimoptions('fmincon','Display','off');

temp_file_controls = readmatrix('FP_405s.xlsx');

addpath('data') 
for h=1:size(time_constants,2)
h
for j=1:length(mice)

%% load data and set some parameters

% id of mouse
mouse_id = mice(j);
number = char(mice(j));
number = number(1:4);

% read behavior and fotometry
if contains(mouse_id,'AL')
    licking = readmatrix(strcat(mouse_id,'.csv'));
    licking = licking(2:end,6:7)/1000;
    temp_file = readmatrix('FP_AL_FD.csv');
    fotometry = temp_file(j+1,2:end);
    controls(j,:) = temp_file_controls(j,2:end);
elseif contains(mouse_id,'PBS')
    licking = readmatrix(strcat(mouse_id,'.csv'));
    licking = licking/1000;
    temp_file = readmatrix('FP.csv');
    fotometry = temp_file(str2double(number(4))+2,2:end);
    controls(j,:) = temp_file_controls(j-nb_AL+20,2:end);
elseif contains(mouse_id,'TMT')
    licking = readmatrix(strcat(mouse_id,'.csv'));
    licking = licking/1000;
    temp_file = readmatrix('FP.csv');
    fotometry = temp_file(str2double(number(4))+12,2:end);
    controls(j,:) = temp_file_controls(j-nb_AL-nb_PBS+30,2:end);
elseif contains(mouse_id,'FD')
    licking = readmatrix(strcat(mouse_id,'.csv'));
    licking = licking(2:end,6:7)/1000;
    temp_file = readmatrix('FP_AL_FD.csv');
    fotometry = temp_file(j-20,2:end);
    controls(j,:) = temp_file_controls(j-nb_AL-nb_PBS-nb_TMT+10,2:end);
elseif contains(mouse_id,'AgRP')
    licking = readmatrix(strcat(mouse_id,'.csv'));
    licking = licking/1000;
    temp_file = readmatrix('Y1_NPYstim_n4_FP.csv');
    fotometry = temp_file(j-28,2:end);
    controls(j,:) = temp_file_controls(j-nb_AL-nb_PBS-nb_TMT-nb_FD+59,2:end);
elseif contains(mouse_id,'PEG')
    temp_file = readmatrix(strcat(mouse_id,'.csv'));
    licking = temp_file(4:5,3:50)';
    fotometry = temp_file(3,3:end);
    controls(j,:) = temp_file(2,3:end);
end
baseline(j,:) = fotometry(1:301);
fotometry = fotometry(302:end); %no baseline
foto_signals(j,:) = fotometry;

m = 1; %number of stimulus dimensions

T = length(fotometry); % number of stimulus frames
stimulus = zeros(size(fotometry));
idx = 1;
for i=1:T
    if i>licking(idx,1) && i<licking(idx,2)
        stimulus(i) = 1;
    end
    if i>licking(idx,2)
        idx = idx + 1;
    end
    if idx>length(licking)
        break;
    end
end
licks(j,:) = stimulus;


%% Fit behavior without any gaussian

temp = sort(fotometry);
offsets(j) = temp(36);%min(fotometry);
foto_signals(j,:) = fotometry - offsets(j);

input = convolve(stimulus,time_constants(h));
coeffs(j,h) = input'\foto_signals(j,:)';

lick_signal_temp = coeffs(j,h)*input;
temp = corrcoef(fotometry, lick_signal_temp );
r = temp(1,2);
lick_signal(j,:) = lick_signal_temp ;

avg_r(j,h) =  r;

%% Fit slow signal first on the lower enveloppe of the fotometry and then behavior

[up,lo] = envelope(foto_signals(j,:),30,'peak');

fun = @(y) sum( (lo-y(1)*exp( -((1:length(fotometry))-y(2)).^2/(2*y(3)^2) ) - y(4)*exp( -((1:length(fotometry))-y(5)).^2/(2*y(6)^2)) ).^2 );
y0 = [0,0,100,0,2000,500];
lb = [0,0,100,0,500,400];
ub = [1,1000,Inf,1,3600,Inf];
y = fmincon(fun,y0,[],[],[],[],lb,ub,[],options);

gaussians(j,:) = y(1)*exp( -((1:length(fotometry))-y(2)).^2/(2*y(3)^2) ) + y(4)*exp( -((1:length(fotometry))-y(5)).^2/(2*y(6)^2));
coeffs_both(j,h) = input'\(foto_signals(j,:)-gaussians(j,:))';
temp = corrcoef(foto_signals(j,:), coeffs_both(j,h)*input + gaussians(j,:));
r = temp(1,2);

gauss_params(j,:) = y;
avg_r_both(j,h) =  r;

end

    
end
%% visualize


% avg_r_1 = zeros(nb_AL,3);
% for i=1:nb_AL
% mean_temp = mean([avg_r(1:i-1,:);avg_r(i+1:nb_AL,:)],1);
% avg_r_1(i,2) = max(mean_temp);
% avg_r_1(i,3) = find(mean_temp==max(mean_temp));
% avg_r_1(i,1) = avg_r(i,avg_r_1(i,3));
% end
% 
% avg_r_2 = zeros(nb_PBS,3);
% for i=1:nb_PBS
% mean_temp = mean([avg_r(nb_AL+1:nb_AL+1+i-1,:);avg_r(nb_AL+1+i+1:nb_PBS+nb_AL,:)],1);
% avg_r_2(i,2) = max(mean_temp);
% avg_r_2(i,3) = find(mean_temp==max(mean_temp));
% avg_r_2(i,1) = avg_r(nb_AL+i,avg_r_2(i,3));
% end
% 
% avg_r_3 = zeros(nb_TMT,3);
% for i=1:nb_TMT
% mean_temp = mean([avg_r(nb_AL+nb_PBS+1:nb_AL+nb_PBS+1+i-1,:);avg_r(nb_AL+nb_PBS+1+i+1:nb_PBS+nb_AL+nb_TMT,:)],1);
% avg_r_3(i,2) = max(mean_temp);
% avg_r_3(i,3) = find(mean_temp==max(mean_temp));
% avg_r_3(i,1) = avg_r(nb_AL+nb_PBS+i,avg_r_3(i,3));
% end
% 
% avg_r_4 = zeros(nb_FD,3);
% for i=1:nb_FD
% mean_temp = mean([avg_r(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+1+i-1,:);avg_r(nb_AL+nb_PBS+nb_TMT+1+i+1:nb_PBS+nb_AL+nb_TMT+nb_FD,:)],1);
% avg_r_4(i,2) = max(mean_temp);
% avg_r_4(i,3) = find(mean_temp==max(mean_temp));
% avg_r_4(i,1) = avg_r(nb_AL+nb_PBS+nb_TMT+i,avg_r_4(i,3));
% end
% 
% avg_r_5 = zeros(nb_AgRP,3);
% for i=1:nb_AgRP
% mean_temp = mean([avg_r(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+1+i-1,:);avg_r(nb_AL+nb_PBS+nb_TMT+nb_FD+1+i+1:nb_PBS+nb_AL+nb_TMT+nb_FD+nb_AgRP,:)],1);
% avg_r_5(i,2) = max(mean_temp);
% avg_r_5(i,3) = find(mean_temp==max(mean_temp));
% avg_r_5(i,1) = avg_r(nb_AL+nb_PBS+nb_TMT+nb_FD+i,avg_r_5(i,3));
% end
% 
% avg_r_combo = [avg_r_1;avg_r_2;avg_r_3;avg_r_4;avg_r_5];
% 
% temp=avg_r_combo(:,1);
% 
% %pearson correlations
% figure
% y = temp;
% bar(1,mean(y(1:nb_AL,1)));
% hold on
% bar(2,mean(y(1+nb_AL:nb_AL+nb_PBS,1)));
% hold on
% bar(3,mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1)))
% hold on
% bar(4,mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1)))
% hold on
% bar(5,mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1)))
% errorbar(1,mean(y(1:nb_AL,1)),std(y(1:nb_AL,1))/sqrt(nb_AL),'LineStyle','none','Color','k','LineWidth',2)
% errorbar(2,mean(y(1+nb_AL:nb_AL+nb_PBS,1)),std(y(1+nb_AL:nb_AL+nb_PBS,1))/sqrt(nb_PBS),'LineStyle','none','Color','k','LineWidth',2)
% errorbar(3,mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1)),std(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1))/sqrt(nb_TMT),'LineStyle','none','Color','k','LineWidth',2)
% errorbar(4,mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1)),std(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1))/sqrt(nb_FD),'LineStyle','none','Color','k','LineWidth',2)
% errorbar(5,mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1)),std(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1))/sqrt(nb_AgRP),'LineStyle','none','Color','k','LineWidth',2)
% 
% scatter(1*ones(1,nb_AL),y(1:nb_AL,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
% scatter(2*ones(1,nb_PBS),y(1+nb_AL:nb_AL+nb_PBS,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
% scatter(3*ones(1,nb_TMT),y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
% scatter(4*ones(1,nb_FD),y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
% scatter(5*ones(1,nb_AgRP),y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
% xticklabels(["AL" "PBS" "TMT" "FD" "AgRP stim"])
% ylabel("Pearson correlation coefficient")
% ylim([0 1])
% box off
% 
% 
% %time constant of exponentials
% %[constants,~] = find(avg_r'==temp);
% constants = avg_r_combo(:,3);
% figure
% y = constants;
% bar(1,mean(y(1:nb_AL,1)));
% hold on
% bar(2,mean(y(1+nb_AL:nb_AL+nb_PBS,1)));
% hold on
% bar(3,mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1)))
% hold on
% bar(4,mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1)))
% hold on
% bar(5,mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1)))
% errorbar(1,mean(y(1:nb_AL,1)),std(y(1:nb_AL,1))/sqrt(nb_AL),'LineStyle','none','Color','k','LineWidth',2)
% errorbar(2,mean(y(1+nb_AL:nb_AL+nb_PBS,1)),std(y(1+nb_AL:nb_AL+nb_PBS,1))/sqrt(nb_PBS),'LineStyle','none','Color','k','LineWidth',2)
% errorbar(3,mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1)),std(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1))/sqrt(nb_TMT),'LineStyle','none','Color','k','LineWidth',2)
% errorbar(4,mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1)),std(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1))/sqrt(nb_FD),'LineStyle','none','Color','k','LineWidth',2)
% errorbar(5,mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1)),std(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1))/sqrt(nb_AgRP),'LineStyle','none','Color','k','LineWidth',2)
% 
% scatter(1*ones(1,nb_AL),y(1:nb_AL,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
% scatter(2*ones(1,nb_PBS),y(1+nb_AL:nb_AL+nb_PBS,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
% scatter(3*ones(1,nb_TMT),y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
% scatter(4*ones(1,nb_FD),y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
% scatter(5*ones(1,nb_AgRP),y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
% xticklabels(["AL" "PBS" "TMT" "FD" "AgRP stim"])
% ylabel("Time constants (s)")
% box off

%% All groups together fast

avg_r_all = zeros(nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP+nb_PEG,3);
for i=1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP+nb_PEG
mean_temp = mean([avg_r(1:i-1,:);avg_r(i+1:nb_PBS+nb_AL+nb_TMT+nb_FD+nb_AgRP+nb_PEG,:)],1);
avg_r_all(i,2) = max(mean_temp);
avg_r_all(i,3) = find(mean_temp==max(mean_temp));
avg_r_all(i,1) = avg_r(i,avg_r_all(i,3));
end

temp=avg_r_all(:,1);

%pearson correlations
figure
y = temp;
h = bar([mean(y(1:nb_AL,1));mean(y(1+nb_AL:nb_AL+nb_PBS,1));mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1));mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1));mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1))]');
hold on
errorbar(1,mean(y(1:nb_AL,1)),std(y(1:nb_AL,1))/sqrt(nb_AL),'LineStyle','none','Color','k','LineWidth',2)
errorbar(2,mean(y(1+nb_AL:nb_AL+nb_PBS,1)),std(y(1+nb_AL:nb_AL+nb_PBS,1))/sqrt(nb_PBS),'LineStyle','none','Color','k','LineWidth',2)
errorbar(3,mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1)),std(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1))/sqrt(nb_TMT),'LineStyle','none','Color','k','LineWidth',2)
errorbar(4,mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1)),std(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1))/sqrt(nb_FD),'LineStyle','none','Color','k','LineWidth',2)
errorbar(5,mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1)),std(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1))/sqrt(nb_AgRP),'LineStyle','none','Color','k','LineWidth',2)

scatter(1*ones(1,nb_AL),y(1:nb_AL,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(2*ones(1,nb_PBS),y(1+nb_AL:nb_AL+nb_PBS,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(3*ones(1,nb_TMT),y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(4*ones(1,nb_FD),y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(5*ones(1,nb_AgRP),y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
xticklabels(["AL" "PBS" "TMT" "FD" "AgRP stim"])
ylabel("Pearson correlation coefficient")
ylim([0 1])
box off
title('Licking behavior fit')

%time constant of exponentials
%[constants,~] = find(avg_r'==temp);
constants = avg_r_all(:,3);
figure
y = constants;
h = bar([mean(y(1:nb_AL,1));mean(y(1+nb_AL:nb_AL+nb_PBS,1));mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1));mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1));mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1))]');
hold on
errorbar(1,mean(y(1:nb_AL,1)),std(y(1:nb_AL,1))/sqrt(nb_AL),'LineStyle','none','Color','k','LineWidth',2)
errorbar(2,mean(y(1+nb_AL:nb_AL+nb_PBS,1)),std(y(1+nb_AL:nb_AL+nb_PBS,1))/sqrt(nb_PBS),'LineStyle','none','Color','k','LineWidth',2)
errorbar(3,mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1)),std(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1))/sqrt(nb_TMT),'LineStyle','none','Color','k','LineWidth',2)
errorbar(4,mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1)),std(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1))/sqrt(nb_FD),'LineStyle','none','Color','k','LineWidth',2)
errorbar(5,mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1)),std(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1))/sqrt(nb_AgRP),'LineStyle','none','Color','k','LineWidth',2)

scatter(1*ones(1,nb_AL),y(1:nb_AL,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(2*ones(1,nb_PBS),y(1+nb_AL:nb_AL+nb_PBS,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(3*ones(1,nb_TMT),y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(4*ones(1,nb_FD),y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(5*ones(1,nb_AgRP),y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
xticklabels(["AL" "PBS" "TMT" "FD" "AgRP stim"])
ylabel("Time constants (s)")
box off
title('Licking behavior fit')

%% All groups together slow+fast

avg_r_all_both = zeros(nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP+nb_PEG,3);
for i=1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP+nb_PEG
mean_temp = mean([avg_r_both(1:i-1,:);avg_r_both(i+1:nb_PBS+nb_AL+nb_TMT+nb_FD+nb_AgRP+nb_PEG,:)],1);
avg_r_all_both(i,2) = max(mean_temp);
avg_r_all_both(i,3) = find(mean_temp==max(mean_temp));
avg_r_all_both(i,1) = avg_r_both(i,avg_r_all_both(i,3));
end

temp=avg_r_all_both(:,1);

%pearson correlations
figure
y = temp;
h = bar([mean(y(1:nb_AL,1));mean(y(1+nb_AL:nb_AL+nb_PBS,1));mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1));mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1));mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1))]');
hold on
errorbar(1,mean(y(1:nb_AL,1)),std(y(1:nb_AL,1))/sqrt(nb_AL),'LineStyle','none','Color','k','LineWidth',2)
errorbar(2,mean(y(1+nb_AL:nb_AL+nb_PBS,1)),std(y(1+nb_AL:nb_AL+nb_PBS,1))/sqrt(nb_PBS),'LineStyle','none','Color','k','LineWidth',2)
errorbar(3,mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1)),std(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1))/sqrt(nb_TMT),'LineStyle','none','Color','k','LineWidth',2)
errorbar(4,mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1)),std(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1))/sqrt(nb_FD),'LineStyle','none','Color','k','LineWidth',2)
errorbar(5,mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1)),std(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1))/sqrt(nb_AgRP),'LineStyle','none','Color','k','LineWidth',2)

scatter(1*ones(1,nb_AL),y(1:nb_AL,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(2*ones(1,nb_PBS),y(1+nb_AL:nb_AL+nb_PBS,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(3*ones(1,nb_TMT),y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(4*ones(1,nb_FD),y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(5*ones(1,nb_AgRP),y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
xticklabels(["AL" "PBS" "TMT" "FD" "AgRP stim"])
ylabel("Pearson correlation coefficient")
ylim([0 1])
box off
title('Licking behavior + slow component fit')


%time constant of exponentials
%[constants,~] = find(avg_r'==temp);
constants_both = avg_r_all_both(:,3);
figure
y = constants_both;
h = bar([mean(y(1:nb_AL,1));mean(y(1+nb_AL:nb_AL+nb_PBS,1));mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1));mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1));mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1))]');
hold on
errorbar(1,mean(y(1:nb_AL,1)),std(y(1:nb_AL,1))/sqrt(nb_AL),'LineStyle','none','Color','k','LineWidth',2)
errorbar(2,mean(y(1+nb_AL:nb_AL+nb_PBS,1)),std(y(1+nb_AL:nb_AL+nb_PBS,1))/sqrt(nb_PBS),'LineStyle','none','Color','k','LineWidth',2)
errorbar(3,mean(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1)),std(y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1))/sqrt(nb_TMT),'LineStyle','none','Color','k','LineWidth',2)
errorbar(4,mean(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1)),std(y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1))/sqrt(nb_FD),'LineStyle','none','Color','k','LineWidth',2)
errorbar(5,mean(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1)),std(y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1))/sqrt(nb_AgRP),'LineStyle','none','Color','k','LineWidth',2)

scatter(1*ones(1,nb_AL),y(1:nb_AL,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(2*ones(1,nb_PBS),y(1+nb_AL:nb_AL+nb_PBS,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(3*ones(1,nb_TMT),y(nb_AL+nb_PBS+1:nb_AL+nb_PBS+nb_TMT,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(4*ones(1,nb_FD),y(nb_AL+nb_PBS+nb_TMT+1:nb_AL+nb_PBS+nb_TMT+nb_FD,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(5*ones(1,nb_AgRP),y(nb_AL+nb_PBS+nb_TMT+nb_FD+1:nb_AL+nb_PBS+nb_TMT+nb_FD+nb_AgRP,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
xticklabels(["AL" "PBS" "TMT" "FD" "AgRP stim"])
ylabel("Time constants (s)")
box off
title('Licking behavior + slow component fit')

figure
bar(1,mean(avg_r_all(:,1)));
hold on
bar(2,mean(avg_r_all_both(:,1)));
errorbar(1,mean(avg_r_all(:,1)),std(avg_r_all(:,1))/sqrt(nb_AL+nb_PBS+nb_FD+nb_TMT+nb_AgRP),'LineStyle','none','Color','k','LineWidth',2)
errorbar(2,mean(avg_r_all_both(:,1)),std(avg_r_all(:,1))/sqrt(nb_AL+nb_PBS+nb_FD+nb_TMT+nb_AgRP),'LineStyle','none','Color','k','LineWidth',2)
scatter(1*ones(1,nb_AL+nb_PBS+nb_FD+nb_TMT+nb_AgRP),avg_r_all(:,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(2*ones(1,nb_AL+nb_PBS+nb_FD+nb_TMT+nb_AgRP),avg_r_all_both(:,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
ylabel("Pearson correlation coefficient")
ylim([0 1])
box off
title('Licking behavior + slow component fit')

%% store fits in folder (if uncommented)

x0=10;
y0=10;
width=800;
height=500;

for i=1:nb_AL+nb_PBS+nb_FD+nb_TMT+nb_AgRP+nb_PEG
    figure
    subplot(2,1,1)
    plot(-300:3600,[baseline(i,:),foto_signals(i,:)+offsets(i)],'LineWidth',1,'Color',[0.40,0.66,0.07])
    hold on
    plot(1:3600,coeffs(i,constants(i))*convolve(licks(i,:),time_constants(constants(i)))+offsets(i),'LineWidth',1,'Color',[0.65,0.65,0.65])
    hold on
    plot(-300:3600,controls(i,:),'LineWidth',0.25,'Color',[0.2 0.5 0.9 0.25])
    xticks([-300 0 600 1200 1800 2400 3000 3600])
    xticklabels({'-5','0','10','20','30','40','50','60'})
    xlim([-300 3600])
    ylim([min(foto_signals(i,:)+offsets(i))-0.05 max(foto_signals(i,:)+offsets(i))+0.05])
    xlabel('Time (mins)')
    ylabel('Signal')
    legend('\Delta F/F_0','Fit behavior','Location','northeastoutside')
    legend boxoff 
    title('Correlation = '+string(avg_r_all(i,1)),'FontWeight','Normal')
    box off

    subplot(2,1,2)
    plot(-300:3600,[baseline(i,:),foto_signals(i,:)+offsets(i)],'LineWidth',1,'Color',[0.40,0.66,0.07])
    hold on
    plot(1:3600,coeffs_both(i,constants_both(i))*convolve(licks(i,:),time_constants(constants_both(i)))+gaussians(i,:)+offsets(i),'LineWidth',1,'Color',[0.65,0.65,0.65])
    hold on
    plot(-300:3600,controls(i,:),'LineWidth',0.25,'Color',[0.2 0.5 0.9 0.25])
    xticks([-300 0 600 1200 1800 2400 3000 3600])
    xticklabels({'-5','0','10','20','30','40','50','60'})
    xlim([-300 3600])
    ylim([min(foto_signals(i,:)+offsets(i))-0.05 max(foto_signals(i,:)+offsets(i))+0.05])
    xlabel('Time (mins)')
    ylabel('Signal')
    legend('\Delta F/F_0','Fit behavior \newline+ gaussians','Location','northeastoutside')
    legend boxoff 
    title('Correlation = '+string(avg_r_all_both(i,1)),'FontWeight','Normal')
    box off
    set(gcf,'position',[x0,y0,width,height])

    print('figures/fits_needs/fit_mouse_'+mice(i),'-dpdf','-r300','-vector')
end