function [ output_args ] = BFSVM_main( dataset,Basic_para,time_sampling,C )
%BFSVM_MAIN Summary of this function goes here
%   Detailed explanation goes here
ktimes=Basic_para.ktimes;
std_vec=Basic_para.std_vec';
k=Basic_para.k;
Basic_para.c=C;
sum_length=2*k*length(std_vec);
para_sampling.k=Basic_para.k;
vail_type=Basic_para.vail_type;
label=Basic_para.label;
dataname=Basic_para.datanName;
IR=Basic_para.IR;%��ƽ����
para_sampling.IR=IR;
sam_ratio=Basic_para.samp_ratio;
fea_sam=Basic_para.fea_sam;%�����ɼ�������
sub_fea=ones(fea_sam,1);
sub=time_sampling*fea_sam;
time = zeros(ktimes,1);%������¼ѵ��ʱ�������
% [mat_sample_way,mat_sample_num]=Matrixlize_fun(sub_fea);%���󻯣�[�����Ժ����Ϸ�ʽ����Ϸ�ʽ�ĸ���]=������������Ŀ�ĺ���(temptvec_data),std_vecΪһ����������
%������ƫ��������i���ƫ�Ƽ�¼��bias_class(i)��
% para_sampling.Mways=mat_sample_num;
% T_combin=(mat_sample_num*(mat_sample_num-1))/2;%���������
% if IR>(T_combin*sub)
%     sub=sam_ratio*ceil(IR/T_combin);
% end
bias_class = zeros(length(unique(label)),1);
for i_label = 1:length(unique(label))
    i_label_tempt = find(label==unique(i_label));
    bias_class(i_label) = i_label_tempt(1);
end%for_i_label
bias_class = bias_class - 1;

feature_bagg=get_fea(sum_length,sub,Basic_para.dim);
sum_class = numel(unique(label));%��������
AUC=zeros(ktimes,1);
T_AUC=zeros(ktimes,1);
GMM=ones(ktimes,1);
accuracy = zeros(ktimes,1);%��¼��ȷ�ȵ�����

InputPar.C = C;
InputPar.curC=0;%��ǰ�ӽ��õ�C
% InputPar.lam = lam;
% InputPar.ita=ita;
InputPar.dataname= dataname;
global_covflag='succeed';
%------------------------------
%------------------------------
%----��ӡ��Ļ-----------------
disp(['Setting:Dataset-',dataname,',Cross Vaildation-',vail_type,'feature num-',num2str(Basic_para.dim),' ,k-',num2str(k),' ,C-',num2str(C)]);%��ӡ����Ļ��
disp(['--------------------------------------']);

%----------------------------
%-----------�ļ������¼-------------
%��¼ÿ�����ݼ��Ľ��
file_name_result=['..\result_all\',dataname,'of',num2str(Basic_para.dim),'_doublebaggSVM_para_result','.txt'];
file_id_result=fopen(file_name_result,'at+');
%��¼ÿ�ֽ��
file_name_mccv=['..\result_mccv\',dataname,'of',num2str(Basic_para.dim),'_doublebaggpara_',vail_type,'_cv','.txt'];
file_id_mccv=fopen(file_name_mccv,'at+');
fprintf(file_id_mccv,'the parameter k- %3d ,C1-%3.3f ,gamma-%3.3f  \r\n',k,C);
%-----------------------------------
%--------------------------------------
%-------------------------------------
%--------��¼����--------------------

%-----------------ѵ������-----------------
%---------------------------------------
for i_iter=1:ktimes
    tic;%��¼ѵ��ʱ��
    samth=1;%��i��������
    for i_classone = 1:(sum_class-1)
        for i_classtwo = (i_classone+1):sum_class
         
            

                    samp_ensem=1;
                    for i_ensemble=1:time_sampling%��������
                        [train_binary_data_outside,train_binary_label,gmm_par]=Sample_Genaration(dataset,vail_type,i_iter,i_classone,i_classtwo,bias_class,para_sampling);
                        for i_fea=1:fea_sam
                            fea_select=feature_bagg(samth,:);
                            Basic_para.feat=length(fea_select); %��������
                        train_binary_data=train_binary_data_outside(:,fea_select);
%                         option=svmsmoset('MaxIter',1000000);
                        SVM(i_classone,i_classtwo).candidate{samth}=svmtrain(train_binary_data,train_binary_label,'method','SMO','boxconstraint',C);%����ѵ��,'SMO_OPTS',option
%                         MHKS(i_classone,i_classtwo).candidate{samth}=MHKS_train(train_binary_data,train_binary_label,Basic_para);  
                    

                    samp_ensem=samp_ensem+1;
                    samth=samth+1;
                        end%fea_sam
                    end%ensemble
             
        end%end class2
    end%end class1
    time(ktimes) = toc;%end
%testing
test_par.gmm=gmm_par;
[test_data_final_outside,test_label]=TestSample_Genaration(dataset,vail_type,i_iter,test_par);
samth=1;
matrix_vote = zeros(length(test_label),1);%����ͶƱ����ÿһ����һ��Group��ѡ����1������Ʊ��ͳ��
        for i_testone = 1:(sum_class-1)
            for i_testtwo = (i_testone+1):sum_class
                %���δ���������ѵ����õ����ݽ��в���       
                
                    
                            samp_ensem=1;
                            for i_ensemble=1:time_sampling
                                for i_fea=1:fea_sam
                                    fea_select=feature_bagg(samth,:);
                                    test_data_final=test_data_final_outside(:,fea_select);
                           
                    Group = svmclassify(SVM(i_classone,i_classtwo).candidate{samth},test_data_final);
                    matrix_vote = cat(2,matrix_vote,Group);%�����
                    clear Group;
                    samp_ensem=samp_ensem+1;
                    samth=samth+1;
                                end
                           end%ensemble k
                     
            end%for_i_testtwo
        end%for_i_testone 
        
%         i_candidate = (sum_class)*(sum_class-1)/2;
%         for i_poll = 1:length(test_label)
%             vector_vote = matrix_vote(i_poll,2:(i_candidate+1));
%             matrix_vote(i_poll,1) = mode(vector_vote);
%         end%for_i_poll
        
        for i_poll = 1:length(test_label)
            vector_vote = matrix_vote(i_poll,2:end);
            matrix_vote(i_poll,1) = mode(vector_vote);
        end%for_i_poll
        
        accuracy(i_iter,1) = 100*(1-(length(find((test_label - matrix_vote(:,1))~=0))/length(test_label)));
        [~,tempt_location] = unique(test_label);%���ز��ظ�Ԫ�صĸ�����[a,b]=unique(A),a��������A�в��ظ���Ԫ�أ�ÿ��һ����b���ص�һ����ͬԪ�ص�λ��
           tempt_location1=[0;tempt_location];
           [tem_clas,~]=unique(test_label);
           for i_num=1:sum_class
               tem_num=find(test_label==tem_clas(i_num));
               class_cur_num(i_num)=length(tem_num);
           end
          for i=1:sum_class
             AUC(i_iter)=AUC(i_iter)+100*(1-(length((find((test_label(find(test_label==i))-matrix_vote(find(test_label==i),1))~=0)))/class_cur_num(i)));
%           AUC(i_iter)=AUC(i_iter)+100*(1-(length(find((test_label(1+tempt_location1(i):tempt_location1(i+1)) - matrix_vote(1+tempt_location1(i):tempt_location1(i+1),1))~=0))/length(test_label(1+tempt_location1(i):tempt_location1(i+1)))));
%              GMM(i_iter)=GMM(i_iter)*(1-(length(find((test_label(1+tempt_location1(i):tempt_location1(i+1)) - matrix_vote(1+tempt_location1(i):tempt_location1(i+1),1))~=0))/length(test_label(1+tempt_location1(i):tempt_location1(i+1)))));
             GMM(i_iter)=GMM(i_iter)*(1-(length((find((test_label(find(test_label==i))-matrix_vote(find(test_label==i),1))~=0)))/class_cur_num(i)));
          end
          AUC(i_iter)=AUC(i_iter)/sum_class;
          GMM(i_iter)=100*GMM(i_iter)^(1/sum_class);
%            [~,~,~,T_AUC(i_iter)]=perfcurve(test_label,dv,'1');
           disp(['The present accuracy of ',num2str(i_iter),' iteration in MCCV is: ',num2str(accuracy(i_iter))]);%��ӡ����Ļ��
        disp(['The present AUC of ',num2str(i_iter),' iteration in MCCV is: ',num2str(AUC(i_iter))]);
%         disp(['The present TAUC of ',num2str(i_iter),' iteration in MCCV is: ',num2str(T_AUC(i_iter))]);
        disp(['The present GMM of ',num2str(i_iter),' iteration in MCCV is: ',num2str(GMM(i_iter))]);
        disp(['------']);
        
        %------------------------------------
        fprintf(file_id_mccv,'   %3.3f   %3.3f   %3.3f\r\n',accuracy(i_iter),AUC(i_iter),GMM(i_iter));
        %------------------------------------
        
end%end ktimes
    disp(['The average accuracy is: ',num2str(mean(accuracy))]);%��ӡ����Ļ��
    disp(['The average AUC is: ',num2str(mean(AUC))]);%��ӡ����Ļ��
    disp(['The average true AUC is: ',num2str(mean(T_AUC))]);%��ӡ����Ļ��
    disp(['The average GMM is: ',num2str(mean(GMM))]);%��ӡ����Ļ��
    disp(['The std of accuracies is: ',num2str(std(accuracy))]);
    disp(['The std of AUC is: ',num2str(std(AUC))]);%��ӡ����Ļ��
    disp(['The std of GMM is: ',num2str(std(GMM))]);%��ӡ����Ļ��
    disp(['The average time(s) is: ',num2str(mean(time))]);
    disp(['--------------------------------------']);
    fprintf(file_id_result,' %3d  %3d  %3.3f  %3.3f    %3.3f  %3.3f  %3.3f  %3.3f  %3.3f  %3.3f %3.3f \r\n',Basic_para.dim,k,C,mean(accuracy),mean(AUC),mean(GMM),std(accuracy),std(AUC),std(GMM),mean(time),mean(T_AUC));
    fprintf(file_id_mccv,'the acc of mean is %3.3f, auc of mean is %3.3f\r\n',mean(accuracy),mean(AUC));
%     results_detail(1,1:end)=[k,C,C2,lam,gamma,accuracy',mean(accuracy)];

    fclose(file_id_mccv);
    delete file_id_mccv;
    fclose(file_id_result);%���ļ�
    delete file_id_result;

end

