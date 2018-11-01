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
IR=Basic_para.IR;%不平衡率
para_sampling.IR=IR;
sam_ratio=Basic_para.samp_ratio;
fea_sam=Basic_para.fea_sam;%特征采集次数数
sub_fea=ones(fea_sam,1);
sub=time_sampling*fea_sam;
time = zeros(ktimes,1);%设立记录训练时间的向量
% [mat_sample_way,mat_sample_num]=Matrixlize_fun(sub_fea);%矩阵化，[矩阵化以后的组合方式，组合方式的个数]=计算矩阵化组合数目的函数(temptvec_data),std_vec为一个样例样本
%计算类偏移量，第i类的偏移记录在bias_class(i)里
% para_sampling.Mways=mat_sample_num;
% T_combin=(mat_sample_num*(mat_sample_num-1))/2;%排列组合数
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
sum_class = numel(unique(label));%求类总数
AUC=zeros(ktimes,1);
T_AUC=zeros(ktimes,1);
GMM=ones(ktimes,1);
accuracy = zeros(ktimes,1);%记录精确度的向量

InputPar.C = C;
InputPar.curC=0;%当前视角用的C
% InputPar.lam = lam;
% InputPar.ita=ita;
InputPar.dataname= dataname;
global_covflag='succeed';
%------------------------------
%------------------------------
%----打印屏幕-----------------
disp(['Setting:Dataset-',dataname,',Cross Vaildation-',vail_type,'feature num-',num2str(Basic_para.dim),' ,k-',num2str(k),' ,C-',num2str(C)]);%打印在屏幕上
disp(['--------------------------------------']);

%----------------------------
%-----------文件输出记录-------------
%记录每个数据集的结果
file_name_result=['..\result_all\',dataname,'of',num2str(Basic_para.dim),'_doublebaggSVM_para_result','.txt'];
file_id_result=fopen(file_name_result,'at+');
%记录每轮结果
file_name_mccv=['..\result_mccv\',dataname,'of',num2str(Basic_para.dim),'_doublebaggpara_',vail_type,'_cv','.txt'];
file_id_mccv=fopen(file_name_mccv,'at+');
fprintf(file_id_mccv,'the parameter k- %3d ,C1-%3.3f ,gamma-%3.3f  \r\n',k,C);
%-----------------------------------
%--------------------------------------
%-------------------------------------
%--------记录结束--------------------

%-----------------训练部分-----------------
%---------------------------------------
for i_iter=1:ktimes
    tic;%记录训练时间
    samth=1;%第i个分类器
    for i_classone = 1:(sum_class-1)
        for i_classtwo = (i_classone+1):sum_class
         
            

                    samp_ensem=1;
                    for i_ensemble=1:time_sampling%采样次数
                        [train_binary_data_outside,train_binary_label,gmm_par]=Sample_Genaration(dataset,vail_type,i_iter,i_classone,i_classtwo,bias_class,para_sampling);
                        for i_fea=1:fea_sam
                            fea_select=feature_bagg(samth,:);
                            Basic_para.feat=length(fea_select); %特征个数
                        train_binary_data=train_binary_data_outside(:,fea_select);
%                         option=svmsmoset('MaxIter',1000000);
                        SVM(i_classone,i_classtwo).candidate{samth}=svmtrain(train_binary_data,train_binary_label,'method','SMO','boxconstraint',C);%代入训练,'SMO_OPTS',option
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
matrix_vote = zeros(length(test_label),1);%设置投票矩阵，每一列是一个Group候选，第1列是总票数统计
        for i_testone = 1:(sum_class-1)
            for i_testtwo = (i_testone+1):sum_class
                %依次代入两两类训练获得的数据进行测试       
                
                    
                            samp_ensem=1;
                            for i_ensemble=1:time_sampling
                                for i_fea=1:fea_sam
                                    fea_select=feature_bagg(samth,:);
                                    test_data_final=test_data_final_outside(:,fea_select);
                           
                    Group = svmclassify(SVM(i_classone,i_classtwo).candidate{samth},test_data_final);
                    matrix_vote = cat(2,matrix_vote,Group);%加入候补
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
        [~,tempt_location] = unique(test_label);%返回不重复元素的个数，[a,b]=unique(A),a返回向量A中不重复的元素，每种一个；b返回第一个不同元素的位置
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
           disp(['The present accuracy of ',num2str(i_iter),' iteration in MCCV is: ',num2str(accuracy(i_iter))]);%打印在屏幕上
        disp(['The present AUC of ',num2str(i_iter),' iteration in MCCV is: ',num2str(AUC(i_iter))]);
%         disp(['The present TAUC of ',num2str(i_iter),' iteration in MCCV is: ',num2str(T_AUC(i_iter))]);
        disp(['The present GMM of ',num2str(i_iter),' iteration in MCCV is: ',num2str(GMM(i_iter))]);
        disp(['------']);
        
        %------------------------------------
        fprintf(file_id_mccv,'   %3.3f   %3.3f   %3.3f\r\n',accuracy(i_iter),AUC(i_iter),GMM(i_iter));
        %------------------------------------
        
end%end ktimes
    disp(['The average accuracy is: ',num2str(mean(accuracy))]);%打印在屏幕上
    disp(['The average AUC is: ',num2str(mean(AUC))]);%打印在屏幕上
    disp(['The average true AUC is: ',num2str(mean(T_AUC))]);%打印在屏幕上
    disp(['The average GMM is: ',num2str(mean(GMM))]);%打印在屏幕上
    disp(['The std of accuracies is: ',num2str(std(accuracy))]);
    disp(['The std of AUC is: ',num2str(std(AUC))]);%打印在屏幕上
    disp(['The std of GMM is: ',num2str(std(GMM))]);%打印在屏幕上
    disp(['The average time(s) is: ',num2str(mean(time))]);
    disp(['--------------------------------------']);
    fprintf(file_id_result,' %3d  %3d  %3.3f  %3.3f    %3.3f  %3.3f  %3.3f  %3.3f  %3.3f  %3.3f %3.3f \r\n',Basic_para.dim,k,C,mean(accuracy),mean(AUC),mean(GMM),std(accuracy),std(AUC),std(GMM),mean(time),mean(T_AUC));
    fprintf(file_id_mccv,'the acc of mean is %3.3f, auc of mean is %3.3f\r\n',mean(accuracy),mean(AUC));
%     results_detail(1,1:end)=[k,C,C2,lam,gamma,accuracy',mean(accuracy)];

    fclose(file_id_mccv);
    delete file_id_mccv;
    fclose(file_id_result);%闭文件
    delete file_id_result;

end

