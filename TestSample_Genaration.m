function [ test_data_final,test_label ] = TestSample_Genaration( dataset,vail_type,ktimes,test_par )
%TESTSAMPLE_GENARATION Summary of this function goes here
%   Detailed explanation goes here
switch vail_type
    case 'FCV'
        
gmm_par=test_par.gmm;
label=dataset{ktimes,2}(:,end);
data_gmm=dataset{ktimes,2};
data=fisherVector_vlv(data_gmm,gmm_par);
data=data';
label(find(label==0))=2;
index_label_1=find(label==1);
index_label_2=find(label==2);
label_1=label(index_label_1);
label_2=label(index_label_2);
data_1=data(index_label_1,:);
data_2=data(index_label_2,:);
data_all=[data_1;data_2];
label_all=[label_1;label_2];

test_label=label_all;
test_data_final=data_all;
    otherwise
        d=0;
end

end

