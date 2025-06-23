%% Short explanation

% This script loads photometry data, recorded at 1Hz, from excell files in the /data folder.
% The analysis is done for three different experiments, all in PBN and ad
% libitum fed mice, but different subset of neurons.
% (AL: Y1R neurons, vglut: vglut neurons, nonY1: all non-Y1R neurons in PBN).
% We fit first the licking behavior to the photometry signal. Then we fit
% both the behavior and two gaussians. We record the resulting Pearson
% coefficients. The procedure is described in the methods.

% A lot of figures are created, one showing the two fits for each mouse (20
% figures), bar charts summarizing the Pearson coefficients (2 figures),
% bar charts summarizing the time constants (2 figures).
% The time constant for a mouse is the one that maximizes the average Pearson correlation 
% for all other mice in the same experimental group.

%% loop over all data and gather params

mice = ["1455_AL","1456_AL","1459_AL","1460_AL","1382_AL","1483_AL","1484_AL",...
    "1581_vglut","1582_vglut","1583_vglut","1584_vglut","1652_vglut","1653_vglut","1654_vglut",...
    "1880_nonY1","1881_nonY1","1882_nonY1","1883_nonY1","1884_nonY1","1885_nonY1"];
exps = ["AL","vglut","nonY1"];

time_constants = 1:50;

nb_AL = 7;
nb_vglut = 7;
nb_nonY1 = 6;


coeffs = zeros(length(mice),size(time_constants,2));
coeffs_both = zeros(length(mice),size(time_constants,2));
lick_signal = zeros(length(mice),3600);
licks = zeros(length(mice),3600);
foto_signals = zeros(length(mice),3600);
gauss_params = zeros(length(mice),6);
gaussians = zeros(length(mice),3600);
offsets = zeros(length(mice),1);
baseline = zeros(length(mice),301);

avg_r = zeros(length(mice),size(time_constants,2));
mse = zeros(length(mice),size(time_constants,2));
avg_r_both = zeros(length(mice),size(time_constants,2));
mse_both = zeros(length(mice),size(time_constants,2));
options = optimoptions('fmincon','Display','off');

addpath('data/photometry')  
for j=1:length(mice)
j
for h=1:size(time_constants,2)

%% load data and set some parameters

% id of mouse
mouse_id = mice(j);
number = char(mice(j));
number = number(1:4);

% read behavior

% read fotometry
if contains(mouse_id,'AL')
    licking = readmatrix(strcat(mouse_id,'.csv'));
    licking = licking(2:end,6:7)/1000;
    temp_file = readmatrix('FP_AL_FD.csv');
    fotometry = temp_file(j+1,2:end);
elseif contains(mouse_id,'vglut')
    licking = readmatrix(strcat(mouse_id,'.csv'));
    licking = licking(1:end,:)/1000;
    temp_file = readmatrix('FP_vglut.csv');
    fotometry = temp_file(j-nb_AL,2:end);
elseif contains(mouse_id,'nonY1')
    licking = readmatrix(strcat(mouse_id,'.csv'));
    licking = licking(1:end,:)/1000;
    temp_file = readmatrix('cre_off_formalin_FP.csv');
    fotometry = temp_file(j-nb_AL-nb_vglut,2:end);
end
baseline(j,:) = fotometry(1:301);
fotometry = fotometry(302:end); %no baseline

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

input = convolve(licks(j,:),time_constants(h));
coeffs(j,h) = input'\foto_signals(j,:)';

temp = corrcoef(foto_signals(j,:), coeffs(j,h)*input );
r = temp(1,2);
lick_signal(j,:) = coeffs(j,h)*input ;

avg_r(j,h) =  r;
mse(j,h) = 1-mean((foto_signals(j,:)- coeffs(j,h)*input).^2)/(mean((foto_signals(j,:)-mean(foto_signals(j,:))).^2)); 

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

% fit = coeffs_both(3,8)*convolve(licks(3,:),8) + y(1)*exp( -((1:length(fotometry))-y(2)).^2/(2*y(3)^2) ) + y(4)*exp( -((1:length(fotometry))-y(5)).^2/(2*y(6)^2));

gauss_params(j,:) = y;
avg_r_both(j,h) =  r;
mse_both(j,h) = 1-mean((foto_signals(j,:)- coeffs_both(j,h)*input - gaussians(j,:)).^2)/(mean((foto_signals(j,:)-mean(foto_signals(j,:))).^2)); 

end

    
end
%% visualize fast

avg_r_1 = zeros(nb_AL,3);
for i=1:nb_AL
mean_temp = mean([avg_r(1:i-1,:);avg_r(i+1:nb_AL,:)],1);
avg_r_1(i,2) = max(mean_temp);
avg_r_1(i,3) = find(mean_temp==max(mean_temp));
avg_r_1(i,1) = avg_r(i,avg_r_1(i,3));
end

avg_r_2 = zeros(nb_vglut,3);
for i=1:nb_vglut
mean_temp = mean([avg_r(nb_AL+1:nb_AL+1+i-1,:);avg_r(nb_AL+1+i+1:nb_vglut+nb_AL,:)],1);
avg_r_2(i,2) = max(mean_temp);
avg_r_2(i,3) = find(mean_temp==max(mean_temp));
avg_r_2(i,1) = avg_r(nb_AL+i,avg_r_2(i,3));
end

avg_r_3 = zeros(nb_nonY1,3);
for i=1:nb_nonY1
mean_temp = mean([avg_r(nb_AL+nb_vglut+1:nb_AL+nb_vglut+1+i-1,:);avg_r(nb_AL+nb_vglut+1+i+1:nb_vglut+nb_AL+nb_nonY1,:)],1);
avg_r_3(i,2) = max(mean_temp);
avg_r_3(i,3) = find(mean_temp==max(mean_temp));
avg_r_3(i,1) = avg_r(nb_AL+nb_vglut+i,avg_r_3(i,3));
end

avg_r_combo = [avg_r_1;avg_r_2;avg_r_3];

temp=avg_r_combo(:,1);
[k,p]=ttest2(temp(1:nb_AL),temp(nb_AL+1:nb_AL+nb_vglut))
[k,p]=ttest2(temp(1:nb_AL),temp(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1))

%pearson correlations
figure
y = temp;
bar(1,mean(y(1:nb_AL,1)));
hold on
bar(2,mean(y(nb_AL+1:nb_AL+nb_vglut,1)))
hold on
bar(3,mean(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1)))
errorbar(1,mean(y(1:nb_AL,1)),std(y(1:nb_AL,1))/sqrt(nb_AL),'LineStyle','none','Color','k','LineWidth',2)
errorbar(2,mean(y(nb_AL+1:nb_AL+nb_vglut,1)),std(y(nb_AL+1:nb_AL+nb_vglut,1))/sqrt(nb_vglut),'LineStyle','none','Color','k','LineWidth',2)
errorbar(3,mean(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1)),std(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1))/sqrt(nb_nonY1),'LineStyle','none','Color','k','LineWidth',2)

scatter(1*ones(1,nb_AL),y(1:nb_AL,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(2*ones(1,nb_vglut),y(nb_AL+1:nb_AL+nb_vglut,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(3*ones(1,nb_nonY1),y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
xticklabels(["Y1" "vglut" "nonY1"])
ylabel("Pearson correlation coefficient")
ylim([0 1])
box off
legend(["Y1" "vglut" "non-Y1"])
title('Licking behavior fit')

%time constant of exponentials
%[constants,~] = find(avg_r'==temp);
constants = avg_r_combo(:,3);
figure
y = constants;
h=bar([mean(y(1:nb_AL,1));mean(y(nb_AL+1:nb_AL+nb_vglut,1));mean(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1))]');
hold on
errorbar(1,mean(y(1:nb_AL,1)),std(y(1:nb_AL,1))/sqrt(nb_AL),'LineStyle','none','Color','k','LineWidth',2)
errorbar(2,mean(y(nb_AL+1:nb_AL+nb_vglut,1)),std(y(nb_AL+1:nb_AL+nb_vglut,1))/sqrt(nb_vglut),'LineStyle','none','Color','k','LineWidth',2)
errorbar(3,mean(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1)),std(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1))/sqrt(nb_nonY1),'LineStyle','none','Color','k','LineWidth',2)

scatter(1*ones(1,nb_AL),y(1:nb_AL,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(2*ones(1,nb_vglut),y(nb_AL+1:nb_AL+nb_vglut,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(3*ones(1,nb_nonY1),y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
xticklabels(["Y1" "vglut" "nonY1"])
ylabel("Time constants")
title('Licking behavior fit')

%% visualize slow+fast

avg_r_1 = zeros(nb_AL,3);
for i=1:nb_AL
mean_temp = mean([avg_r_both(1:i-1,:);avg_r_both(i+1:nb_AL,:)],1);
avg_r_1(i,2) = max(mean_temp);
avg_r_1(i,3) = find(mean_temp==max(mean_temp));
avg_r_1(i,1) = avg_r_both(i,avg_r_1(i,3));
end

avg_r_2 = zeros(nb_vglut,3);
for i=1:nb_vglut
mean_temp = mean([avg_r_both(nb_AL+1:nb_AL+1+i-1,:);avg_r_both(nb_AL+1+i+1:nb_vglut+nb_AL,:)],1);
avg_r_2(i,2) = max(mean_temp);
avg_r_2(i,3) = find(mean_temp==max(mean_temp));
avg_r_2(i,1) = avg_r_both(nb_AL+i,avg_r_2(i,3));
end

avg_r_3 = zeros(nb_nonY1,3);
for i=1:nb_nonY1
mean_temp = mean([avg_r_both(nb_AL+nb_vglut+1:nb_AL+nb_vglut+1+i-1,:);avg_r_both(nb_AL+nb_vglut+1+i+1:nb_vglut+nb_AL+nb_nonY1,:)],1);
avg_r_3(i,2) = max(mean_temp);
avg_r_3(i,3) = find(mean_temp==max(mean_temp));
avg_r_3(i,1) = avg_r_both(nb_AL+nb_vglut+i,avg_r_3(i,3));
end

avg_r_combo_both = [avg_r_1;avg_r_2;avg_r_3];

temp=avg_r_combo_both(:,1);
[k,p]=ttest2(temp(1:nb_AL),temp(nb_AL+1:nb_AL+nb_vglut))
[k,p]=ttest2(temp(1:nb_AL),temp(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1))

%pearson correlations
figure
y = temp;
bar(1,mean(y(1:nb_AL,1)));
hold on
bar(2,mean(y(nb_AL+1:nb_AL+nb_vglut,1)))
hold on
bar(3,mean(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1)))
errorbar(1,mean(y(1:nb_AL,1)),std(y(1:nb_AL,1))/sqrt(nb_AL),'LineStyle','none','Color','k','LineWidth',2)
errorbar(2,mean(y(nb_AL+1:nb_AL+nb_vglut,1)),std(y(nb_AL+1:nb_AL+nb_vglut,1))/sqrt(nb_vglut),'LineStyle','none','Color','k','LineWidth',2)
errorbar(3,mean(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1)),std(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1))/sqrt(nb_nonY1),'LineStyle','none','Color','k','LineWidth',2)

scatter(1*ones(1,nb_AL),y(1:nb_AL,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(2*ones(1,nb_vglut),y(nb_AL+1:nb_AL+nb_vglut,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(3*ones(1,nb_nonY1),y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
xticklabels(["Y1" "vglut" "nonY1"])
ylabel("Pearson correlation coefficient")
ylim([0 1])
box off
legend(["Y1" "vglut" "non-Y1"])
title('Licking behavior + slow component fit')

%time constant of exponentials
%[constants,~] = find(avg_r'==temp);
constants_both = avg_r_combo_both(:,3);
figure
y = constants_both;
h=bar([mean(y(1:nb_AL,1));mean(y(nb_AL+1:nb_AL+nb_vglut,1));mean(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1))]');
hold on
errorbar(1,mean(y(1:nb_AL,1)),std(y(1:nb_AL,1))/sqrt(nb_AL),'LineStyle','none','Color','k','LineWidth',2)
errorbar(2,mean(y(nb_AL+1:nb_AL+nb_vglut,1)),std(y(nb_AL+1:nb_AL+nb_vglut,1))/sqrt(nb_vglut),'LineStyle','none','Color','k','LineWidth',2)
errorbar(3,mean(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1)),std(y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1))/sqrt(nb_nonY1),'LineStyle','none','Color','k','LineWidth',2)

scatter(1*ones(1,nb_AL),y(1:nb_AL,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(2*ones(1,nb_vglut),y(nb_AL+1:nb_AL+nb_vglut,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
scatter(3*ones(1,nb_nonY1),y(nb_AL+nb_vglut+1:nb_AL+nb_vglut+nb_nonY1,1),60,'MarkerFaceColor','r','MarkerEdgeColor','k','LineWidth',1)
xticklabels(["Y1" "vglut" "nonY1"])
ylabel("Time constants")
title('Licking behavior + slow component fit')

%% store fits in folder (if uncommented)

x0=10;
y0=10;
width=800;
height=500;

for i=1:nb_AL+nb_vglut+nb_nonY1
    figure
    subplot(2,1,1)
    plot(-300:3600,[baseline(i,:),foto_signals(i,:)+offsets(i)],'LineWidth',1,'Color',[0.40,0.66,0.07])
    hold on
    plot(1:3600,coeffs(i,constants(i))*convolve(licks(i,:),time_constants(constants(i)))+offsets(i),'LineWidth',1,'Color',[0.65,0.65,0.65])
    xticks([-300 0 600 1200 1800 2400 3000 3600])
    xticklabels({'-5','0','10','20','30','40','50','60'})
    xlim([-300 3600])
    ylim([min(foto_signals(i,:)+offsets(i))-0.05 max(foto_signals(i,:)+offsets(i))+0.05])
    xlabel('Time (mins)')
    ylabel('Signal')
    legend('\Delta F/F_0','Fit behavior','Location','northeastoutside')
    legend boxoff 
    title('Correlation = '+string(avg_r_combo(i,1)),'FontWeight','Normal')
    box off

    subplot(2,1,2)
    plot(-300:3600,[baseline(i,:),foto_signals(i,:)+offsets(i)],'LineWidth',1,'Color',[0.40,0.66,0.07])
    hold on
    plot(1:3600,coeffs_both(i,constants_both(i))*convolve(licks(i,:),time_constants(constants_both(i)))+gaussians(i,:)+offsets(i),'LineWidth',1,'Color',[0.65,0.65,0.65])
    xticks([-300 0 600 1200 1800 2400 3000 3600])
    xticklabels({'-5','0','10','20','30','40','50','60'})
    xlim([-300 3600])
    ylim([min(foto_signals(i,:)+offsets(i))-0.05 max(foto_signals(i,:)+offsets(i))+0.05])
    xlabel('Time (mins)')
    ylabel('Signal')
    legend('\Delta F/F_0','Fit behavior \newline+ gaussians','Location','northeastoutside')
    legend boxoff 
    title('Correlation = '+string(avg_r_combo_both(i,1)),'FontWeight','Normal')
    box off
    set(gcf,'position',[x0,y0,width,height])

    print('figures/fits_neuron_types/fit_mouse_'+mice(i),'-dpdf','-r300','-vector')
end