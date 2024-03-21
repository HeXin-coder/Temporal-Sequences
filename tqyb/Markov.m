clear
clc

% 导入数据
df1 = readtable('g15_hepad_s15_4s_20170321_20170321.csv');
df2 = readtable('g15_hepad_s15_4s_20170322_20170322.csv');

% 选择需要的列
df1 = df1(:, {'time_tag', 'S1_COUNT_RATE'});
df2 = df2(:, {'time_tag', 'S1_COUNT_RATE'});

% 合并两个数据框
df = [df1; df2];

% 获取训练数据个数
n_train = height(df1);

% 提取'time_tag'和'S1_COUNT_RATE'列
time_tag = df.time_tag;
S1 = df.S1_COUNT_RATE;

% 使用MinMaxScaler进行归一化处理
max_value = max(S1);
min_value = min(S1);
% S1 = (S1 - min(S1)) / (max(S1) - min(S1));
S1 = (S1 - min_value) / (max_value - min_value);

% 创建特征和标签
tau = 3;
features = zeros(length(time_tag) - tau, tau);
for i = 1:tau
    features(:, i) = S1(i:end - tau + i - 1);
end
labels = S1(tau + 1:end);

% 创建神经网络
net = feedforwardnet(10);
net.layers{1}.transferFcn = 'logsig';
net.trainFcn = 'trainlm';
net.performFcn = 'mse';
net.divideParam.trainRatio = n_train / length(S1);
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 1 - net.divideParam.trainRatio;

% 检查net.mat文件是否已存在
if exist('model.mat', 'file') == 2
    % 如果存在，加载神经网络
    load('model.mat', 'net');
else
    % 训练神经网络
    net = train(net, features', labels');
    save('model.mat', 'net');
end

% 预测
onestep_preds = net(features');

% 进行归一化的逆处理
% onestep_preds = onestep_preds * (max(S1) - min(S1)) + min(S1);
% S1 = S1 * (max(S1) - min(S1)) + min(S1);
onestep_preds = onestep_preds * (max_value - min_value) + min_value;
S1 = S1 * (max_value - min_value) + min_value;


% 绘图
plot(time_tag, S1, time_tag(tau + 1:end), onestep_preds);
legend('data', 'preds');
