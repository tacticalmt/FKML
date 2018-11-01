function [ sample_matrix ] = fisherVector_vlv( data,gmm_par )
%FISHERVECTOR_VLV Summary of this function goes here
%   Detailed explanation goes here
data_temp=data(:,1:(end-1))';%data_temp为不带类标的样本,并且一列为一个样本
vector_mu=[];
vector_sigma=[];
k=4;
[fea,N]=size(data_temp);%N为样本数，fea为特征数
I=ones(fea,1);
sample_matrix=[];
% par=get_fisher_fun(data_temp,k,initial_par);
% [means_g,sigma_g,prior_g]=vl_gmm(data_temp,k);
means_g=gmm_par.means;
sigma_g=gmm_par.sigma;
prior_g=gmm_par.prior;
for i_sam=1:N
    temp_vector=vl_fisher(data_temp(:,i_sam),means_g,sigma_g,prior_g);
    sample_matrix=[sample_matrix temp_vector];
end

end

