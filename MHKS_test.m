function [ Group ] = MHKS_test( MatStruct,test_data_final,label_one,label_two )
%MatStruct.wΪѵ��Ȩ��,��������һ��һ������
%   Detailed explanation goes here

samp=size(test_data_final,1);%��������
test_samp=[test_data_final ones(samp,1)];
Result=test_samp*MatStruct.w;
Result(find(Result>=0))=label_one;
Result(find(Result<0))=label_two;
Group=Result;
end

