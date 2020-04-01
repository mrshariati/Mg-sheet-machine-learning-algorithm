function [] = MgSheetLearningAlg(ti)

% Data
disp('input data:')
AllInputData = [219, 227, 231, 243, 222, 216, 239, 222, 237; 266, 271, 270, 292, 271, 267, 303, 281, 283; 6.27, 7.22, 6.56, 5.62, 7.24, 11.54, 5.37, 8.14, 8.83; 214, 227, 226, 236, 220, 217, 234, 221, 223; 267, 270, 272, 294, 273, 275, 302, 284, 273; 2.59, 1.93, 2.57, 1.55, 1.79, 4.85, 2.34, 3.27, 2.95; 261, 260, 252, 274, 252, 240, 284, 258, 250; 279, 281, 277, 305, 280, 273, 323, 292, 290; 1.76, 4.86, 6.72, 3.38, 3.46, 10.57, 1.91, 2.77, 4.99; 350, 400, 450, 350, 400, 450, 350, 400, 450]
%AllOutputData = [12, 12, 9.7, 7.6, 8, 9.3, 8.5, 8, 8.7; 10.3, 9.2, 10, 10, 9.5, 7.2, 8.7, 10, 8.6];
%p1>p2
AllOutputData = [12, 12, 10, 10, 9.5, 9.3, 8.7, 10, 8.7; 10.3, 9.2, 9.7, 7.6, 8, 7.2, 8.5, 8, 8.6];
% Fixing the random seed
rng(4e5);

% Dimensionality Reduction (Third Party Toolbox)
disp('intrinsic dimension estimated by Maximum Likelihood Estimator:')
d = ceil(intrinsic_dim(AllInputData','MLE'))+1
disp(append('input data is reduced to ', num2str(d), ' with Local Linear Embedding:'))
AllInputData = compute_mapping(AllInputData','LLE', d, 5)'

% Randomly ordered datapoints
ind = [randperm(9) ti];
for i = 1:9
    if ind(i)==ti
        ind(i)=[];
        break;
    end
end
AllInputData = AllInputData(:,ind);
AllOutputData = AllOutputData(:,ind);

% Datapoints regeneration
AllInputData = [repmat(AllInputData(:,1:8),1,5),AllInputData(:,9)];
AllOutputData = [repmat(AllOutputData(:,1:8),1,5),AllOutputData(:,9)];
ind = [randperm(40) 41];
AllInputData = AllInputData(:,ind);
AllOutputData = AllOutputData(:,ind);

% 1% Noise
for i=1:d
    AllInputData = AllInputData + [(max(AllInputData(i,:))-min(AllInputData(i,:)))*.1.*randn(1,40) 0;];
end
AllOutputData = AllOutputData + [(max(AllOutputData(1,:))-min(AllOutputData(1,:)))*.1.*randn(1,40) 0; (max(AllOutputData(2,:))-min(AllOutputData(2,:)))*.1.*randn(1,40) 0];

% Network Setup
net = feedforwardnet(4, 'trainbr');
net.biasConnect = [1;0];

% Training Setup
net.trainParam.showWindow = false;
net.trainParam.epochs = 1000;
net.layers{1}.transferFcn = 'tansig';

% Bayesian Regularization Training Method Gives Different Solutions in Each
% Run, Therefore It Is More Convinient To take best solution Over 100 Runs
minErr = 10.*ones(2,1);
for i=1:500
    net = train(net, AllInputData(:,1:40), AllOutputData(:,1:40));
    Output = net(AllInputData(:,41));
    Err = abs(Output-AllOutputData(:,41));
    if (max(Err(1,1),Err(2,1))<=max(minErr(1,1),minErr(2,1)))
        minErr = Err;
        PredictedOutput = Output;
        RandCoreStep = i;
    end
end

% Number of Unkowns in the Network (In Total 26)
% net.IW
% net.LW
% net.b
% view(net)

% Minimum Error of Solution in All Runs
disp(append('predicted,', ' targeted,', ' relative error,', ' absolute error:'))
sol = [PredictedOutput(1,1), AllOutputData(1,41), minErr(1,1)/(max(AllOutputData(1,:))-min(AllOutputData(1,:))), minErr(1,1);  PredictedOutput(2,1), AllOutputData(2,41), minErr(2,1)/(max(AllOutputData(2,:))-min(AllOutputData(2,:))), minErr(2,1);]
disp(append('minimum attained at ', num2str(RandCoreStep), ' out of 500 steps'))

% Plotting
% Gaussian
x = 0:0.01:2;
y = 0:0.01:3;
[X,Y] = meshgrid(x,y);
PredictedPole_1 = PredictedOutput(1,1).*exp(-(((X-1).^2)./.25 + ((Y-1.75).^2)./.5));
PredictedPole_2 = PredictedOutput(2,1).*exp(-(((X-1).^2)./.25 + ((Y-1.25).^2)./.5));
TargetedPole_1 = AllOutputData(1,41).*exp(-(((X-1).^2)./.25 + ((Y-1.75).^2)./.5));
TargetedPole_2 = AllOutputData(2,41).*exp(-(((X-1).^2)./.25 + ((Y-1.25).^2)./.5));

PredictedZ = max(PredictedPole_1,PredictedPole_2);
TargetedZ = max(TargetedPole_1,TargetedPole_2);

c = gray(8);
c = flipud(c);

figure(ti);

subplot(1,2,1)
surf(X,Y,TargetedZ)
colormap(c)
shading interp
caxis([0 12])
set(gca,'XColor', 'white','YColor','white')
%colorbar('position', [0.496599468870236 0.108531571218796 0.0293520882231955 0.817557513460611], 'Ticks', [0,1.5,3,4.5,6,7.5,9,10.5,12], 'TickLabels',{'0','1.5','3','4.5','6','7.5','9','10.5','12'}, 'FontSize', 15)

subplot(1,2,2)
surf(X,Y,PredictedZ)
colormap(c)
shading interp
caxis([0 12])
set(gca,'XColor', 'white','YColor','white')

end
