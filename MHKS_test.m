function [ Group ] = MHKS_test( MatStruct,test_data_final,label_one,label_two )
%MatStruct.w为训练权重,测试样本一行一个样本
%   Detailed explanation goes here

samp=size(test_data_final,1);%样本个数
test_samp=[test_data_final ones(samp,1)];
Result=test_samp*MatStruct.w;
Result(find(Result>=0))=label_one;
Result(find(Result<0))=label_two;
Group=Result;
end

