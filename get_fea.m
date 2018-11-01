function [ feature_bagg ] = get_fea( dataset,sum_samp,fea_sam )
%GET_FEA Summary of this function goes here
%   Detailed explanation goes here
c_slice_mat = [];
for i_c=1:sum_samp
    now_c=randperm(dataset);
    c_slice_mat=[c_slice_mat;now_c(1:fea_sam)];
    clear now_c;
end
feature_bagg=c_slice_mat;
end

