warning off;
C=[0.01,0.1,1,10];
ensemble=5;
k_gmm=[2,3,4,5];%gmm个数
fea_samp=5;%特征bagging数量
sample_ratio=0.6;%特征采样倍率
featyre_num=18;
dataset='Imbalanced_data';
dataset_imported=load(['..\imbalanced_fcv\',dataset,'.mat']);
Basic_para.b=10^(-6);
Basic_para.rho=0.99;
Basic_para.xi=10^(-4);
%----------------------------------------------

for i_data=1:64
Basic_para.datanName=dataset_imported.Imbalanced_data{i_data,1};
Basic_para.vail_type='FCV';
Basic_para.ktimes=5;%交叉验证轮数，也可以设置成跟数据集自适应
Basic_para.label=dataset_imported.Imbalanced_data{i_data,3}{1,1}(:,end);
Basic_para.std_vec=dataset_imported.Imbalanced_data{i_data,3}{1,1}(1,1:(end-1));
feature_length=length(Basic_para.std_vec);%特征长度;
Basic_para.IR=dataset_imported.Imbalanced_data{i_data,2};
for i_gmm=1:length(k_gmm)
                Basic_para.k=k_gmm(i_gmm);
all_dim=2:(2*Basic_para.k*feature_length);
Basic_para.samp_ratio=sample_ratio;
Basic_para.fea_sam=fea_samp;
% Basic_para.dim=round(2*Basic_para.k*feature_length*sample_ratio);%fea_samp;;
Basic_para.label(find(Basic_para.label==0))=2;
% dataset_struct=load(['..\imbalanced_fcv\',dataset,'.mat']);
dataset_struct=dataset_imported.Imbalanced_data{i_data,3};
%-------------
for i_dim=1:length(all_dim)
    for i_C=1:length(C)
        
                
                Basic_para.dim=all_dim(i_dim);
                BFSVM_main(dataset_struct,Basic_para,ensemble,C(i_C));
        
 
    end%end C
end%end k
end
end