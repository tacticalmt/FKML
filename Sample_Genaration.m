function [ train_binary_data,train_binary_label,gmm_par ] = Sample_Genaration( dataset,vail_type,ktimes,i_classone,i_classtwo,bias_class,para_sampling )
%SAMPLE_GENARATION Summary of this function goes here
%   Detailed explanation goes here
%一列为一个样本
%----------------采样函数------------
%-----------------------------------
% if tp=='FCV'
    label=dataset{ktimes,1}(:,end);
    gmm_train=dataset{ktimes,1}(:,1:(end-1));
    fisher_train=dataset{ktimes,1};
    k=para_sampling.k;%gmm成分个数
    [gmm_par.means,gmm_par.sigma,gmm_par.prior]=vl_gmm(gmm_train',para_sampling.k);
    gmm_par.k=para_sampling.k;
    fisher_mat=fisherVector_vlv(fisher_train,gmm_par);
    all_data=[fisher_mat' label];
    ord_sample_oneclass=find(label==1);
%     set_sample_oneclass=dataset{ktimes,1}(ord_sample_oneclass,1:end);
    set_sample_oneclass=all_data(ord_sample_oneclass,1:end);
    ord_sample_twoclass=find(label==0);
%     set_sample_twoclass=dataset{ktimes,1}(ord_sample_twoclass,1:end);
    set_sample_twoclass=all_data(ord_sample_twoclass,1:end);
    nage_1=length(ord_sample_oneclass);
    nage_2=length(ord_sample_twoclass);
    if nage_1>nage_2
        neg_sample=set_sample_oneclass;
        pos_sample=set_sample_twoclass;
        neg_num=nage_1;
        pos_num=nage_2;
    else
        neg_sample=set_sample_twoclass;
        pos_sample=set_sample_oneclass;
        neg_num=nage_2;
        pos_num=nage_1;
    end
    sampling_num=randperm(neg_num);
    neg_data=neg_sample(sampling_num(1:pos_num),1:(end-1));
    pos_data=pos_sample(:,1:(end-1));
    neg_label=neg_sample(sampling_num(1:pos_num),end);
    pos_label=pos_sample(:,end);
    if nage_1>nage_2
    data_label=[neg_label;pos_label];
    data_label(find(data_label==0))=2;
    data_sample=[neg_data;pos_data];
    else
    data_label=[pos_label;neg_label];
    data_label(find(data_label==0))=2;
    data_sample=[pos_data;neg_data];
    end
% end


%---------------采样结束--------------
%-----------------------------------
%-------------生成训练样本------------
switch vail_type
    case 'FCV'
    
    train_binary_label=data_label;
    train_binary_data=data_sample;
%     train_binary_data=train_binary_data';
    

    case 'mccv'
     sum_classone_sample = length(find(label==i_classone));%找出当前两类各自的样本数        
     sum_classtwo_sample = length(find(label==i_classtwo));
     train_classone_sample = round(sum_classone_sample*ratio);%四舍五入取整找出训练样本数               
     train_classtwo_sample = round(sum_classtwo_sample*ratio);%两类样本数相同
     % train_classtwo_sample = round(sum_classtwo_sample*ratio);
     bias_classone = bias_class(i_classone);%获取当前类的偏移量
     bias_classtwo = bias_class(i_classtwo);
     train_classone_data = data(:,index_struct(i_iter,i_classone).index(1:train_classone_sample)+bias_classone);%设置训练样本     
     train_classtwo_data = data(:,index_struct(i_iter,i_classtwo).index(1:train_classtwo_sample)+bias_classtwo);
    train_binary_data = [train_classone_data,train_classtwo_data]';
    train_classone_label = ones(train_classone_sample,1)*i_classone;%设置训练样本类标号     
    train_classtwo_label = ones(train_classtwo_sample,1)*i_classtwo; 
    train_binary_label = [train_classone_label;train_classtwo_label];
    otherwise
        display('You have input wrong dataset number!');
end
  

