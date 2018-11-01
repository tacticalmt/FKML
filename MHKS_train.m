function [ model ] = MHKS_train( train_data, train_label, para )
%para参数包含b,c,rho,xi,feat需要传递,traindata一行为一个样本,label为列向量
%   Detailed explanation goes here MHKS训练
%一行一个样本，类标为列向量

%--------initialization-----
num_samp=length(train_label); %number of samples
[class,loc]=unique(train_label);%class 为类的个数向量，loc为每个标签最大的位置数
class_num=length(class);%类的个数
sample_num=zeros(class_num,1);%每个类的样本数存储向量
% sample_num(1)=loc(1);
% for i=2:class_num
%     sample_num(i)=loc(i)-loc(i-1);
% end
Y=train_label;
Y(find(Y==class(1)))=1;
Y(find(Y==class(2)))=-1;
train_data_temp=[train_data ones(num_samp,1)];
w_y=diag(Y);
X=w_y*train_data_temp;
max_iter=200;
b=para.b*ones(num_samp,1);%initial b
rho=para.rho;
c=para.c;
feat=para.feat; %特征个数
w=zeros(feat+1,max_iter);
e=zeros(num_samp,max_iter);
I=ones(num_samp,1);
xi=para.xi;
terminal=0;
Verse_X=pinv(X'*X+c*ones(feat+1,1));%伪逆的结果
%--------------------------
Y(find(Y==class(2)))=-1;
Y_label(1)=class(1);
Y_label(2)=class(2);
[~,feat]=size(train_data);
for train_time=1:max_iter
    w(:,train_time)=Verse_X*X'*(b(:,train_time)+I);
%     a=X*w(train_time);
    e(:,train_time)=X*w(:,train_time)-I-b(train_time);
    b(:,train_time+1)=b(:,train_time)+rho*(e(:,train_time)+abs(e(:,train_time)));
    stop_tag=norm(b(:,train_time+1)-b(:,train_time),2);
    if stop_tag<xi
        terminal=train_time;
        break;
    else
        if train_time==max_iter
            terminal=max_iter;
        end
    end
end
model.w=w(:,terminal);
model.y_label=Y_label;
end

