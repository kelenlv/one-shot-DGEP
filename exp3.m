%experiments of DFDA
clear all
alldata = load('../data/tox+fib.mat'); %  tox+fib.mat   % tox+azo.mat
meas = table2array(alldata.table_total(:,2:end));
clear alldata
meas=meas';
A=meas(1:181,1:2000);
B=meas(182:end,1:2000);
% load fisheriris
% A=meas(1:50,:);
% B=meas(51:100,:);
% C=meas(101:150,:);
%方法一：先将A作为一类，BC作为一类
NA=size(A,1);NB=size(B,1);
p=1/2;
r=[1]; %r=class-1
times=10;
feats=size(A,2);
maxIter=10;
% NC=size(C,1);

%% distributed W
chunk_list=[5] ;%1,5:5:25
% E_list=[];
% E2_list=[];
Ats0_list=[];Ats1_list=[];Ats2_list=[];Ats3_list=[];
Bts0_list=[];Bts1_list=[];Bts2_list=[];Bts3_list=[];
Atr0_list=[];Atr1_list=[];Atr2_list=[];Atr3_list=[];
Btr0_list=[];Btr1_list=[];Btr2_list=[];Btr3_list=[];
T0=[];T1=[];T2=[];T3=[];
for kk=1:times
    %% shuffle
    RandIndexA = randperm(size(A,1));
    RandIndexB = randperm(size(B,1));
    A=A(RandIndexA,:);
    B=B(RandIndexB,:);
    A_train=A(1:floor(p*NA),:);%训练数据取1/2（或者1/3,3/4,1/4）
    B_train=B(1:floor(p*NB),:);
    % C_train=C(1:floor(4*NB/5),:);
    A_test=A((floor(NA*p)+1):end,:);
    B_test=B((floor(NB*p)+1):end,:);
    %% centered check
    [M0,u1,u2,Sw]=prepareM(A_train,B_train);%_train
    tic;
    [q,s,v]=svd(M0);
    t0=toc;
    T0=[T0 t0];
    % w0=Sw^(-1/2)*real(q);
    % % w0=real(w0)';
    % % w0=w0./norm(w0);
    y0=q(:,1)'*(u1+u2)'/(2);%+u3
    [Atr0,Btr0]=train2(q(:,1)',y0,A_train,B_train);
    [Ats0,Bts0]=test2(q(:,1)',y0,A_test,B_test);
    Atr0_list=[Atr0_list Atr0];
    Ats0_list=[Ats0_list Ats0];
    Btr0_list=[Btr0_list Btr0];
    Bts0_list=[Bts0_list Bts0];
for rr=1:size(r,2)
for k=1:size(chunk_list,2)
    chunks=chunk_list(k);
    pA=floor(size(A_train,1)/chunks);
     pB=floor(size(B_train,1)/chunks);
    %% local prepare data
    for i=1:chunks
       A_chunk(:,:,i)=A_train((i-1)*pA+1:i*pA,:); 
       B_chunk(:,:,i)=B_train((i-1)*pB+1:i*pB,:); 
       [M_chunk(:,:,i),~,~,~]=prepareM(A_chunk(:,:,i),B_chunk(:,:,i));
    end
    %% broadcast
    sumM=0;
    for i=1:chunks
        sumM=sumM+squeeze(M_chunk(:,:,i));
    end
%     %% DSVD
%     [U,t1]=dsvd(M_chunk,chunks,rr);
%     U=real(U);
%     y1=U(:,rr)'*(u1+u2)'/2;
%     [Atr1,Btr1]=train2(U(:,1)',y1,A_train,B_train);
%     [Ats1,Bts1]=test2(U(:,1)',y1,A_test,B_test);
%     Atr1_list=[Atr1_list Atr1];
%     Ats1_list=[Ats1_list Ats1];
%     Btr1_list=[Btr1_list Btr1];
%     Bts1_list=[Bts1_list Bts1];
%     T1=[T1 t1];
    %% DGEP-SVD
    tic;
    [qq,ss,vv]=svd(sumM);
    qq=real(qq);
    t2=toc;
%     w1=Sw^(-1/2)*qq;
%     w1=qq';
%     w1=w1./norm(w1);
%     y=qq(:,1)'*(size(A_train,1)*u1+size(B_train,1)*u2)'/(size(A_train,1)+size(B_train,1));
    y2= qq(:,rr)'*(u1+u2)'/2;%+u3
    [Atr2,Btr2]=train2(qq(:,rr)',y2,A_train,B_train);
    [Ats2,Bts2]=test2(qq(:,rr)',y2,A_test,B_test);
    T2=[T2 t2];
    Atr2_list=[Atr2_list Atr2];
    Btr2_list=[Btr2_list Btr2];
    Ats2_list=[Ats2_list Ats2];
    Bts2_list=[Bts2_list Bts2];
%     
    %% DGEP-PM
    tic;
%     U_g=(orth(rand(feats,rr)));%chunkSize \times r
    U_g = q(:,1);
    for i=1:maxIter
        for w = 1: chunks
            H1(:,:, w) = squeeze(M_chunk(:,:,w))*squeeze(M_chunk(:,:,w))'*U_g;%*randn(chunkSize, 1)+sigma*rand(feats,r)/chunkSize
        end    
        K1=zeros(feats,rr);      
        for w=1:chunks
            K1 = feats.*squeeze(H1(:,:,w))+K1;
        end
        K1=K1/(feats*chunks);
        [U1,~]=qr(K1,0);
        U_g = U1;
    end
    U_g=real(U_g);
    t3=toc;
    T3=[T2 t3];
    y3= U_g(:,rr)'*(u1+u2)'/2;%+u3
    [Atr3,Btr3]=train2(U_g(:,rr)',y3,A_train,B_train);
    [Ats3,Bts3]=test2(U_g(:,rr)',y3,A_test,B_test);
    
    Atr3_list=[Atr3_list Atr3];
    Btr3_list=[Btr3_list Btr3];
    Ats3_list=[Ats3_list Ats3];
    Bts3_list=[Bts3_list Bts3];
%     E_list=[E_list sin(subspace(q(:,rr),qq(:,rr)))];%error-q
%     E2_list=[E2_list norm(M0-sumM)];%error-M
%     plot_fda(qq,A_train,B_train)
    clear A_chunk B_chunk M_chunk H1
end
end
end
function [M,u1,u2,Sw]=prepareM(A_train,B_train)%Binary classification
    u1=mean(A_train);u2=mean(B_train);
    u=(u1+u2)/2;
    Sb=((u1-u)'*(u1-u)+(u2-u)'*(u2-u))/2;%+(u3-u)'*(u3-u)
%     Sb=(u1-u2)'*(u1-u2);
    S1=0;S2=0;
    for i=1:size(A_train,1)
        S1=S1+(A_train(i,:)-u1)'*(A_train(i,:)-u1);
    end
    for i=1:size(B_train,1)
        S2=S2+(B_train(i,:)-u2)'*(B_train(i,:)-u2);
    end
    Sw=(S1+S2)/2;%+S3
    M=Sw^(-1/2)*Sb*Sw^(-1/2);
end

function [rate_A,rate_B]=train2(w1,y0,A_test,B_test)
r1=0;
for i=1:size(B_test,1)
    if w1*B_test(i,:)'>y0
        r1=r1+1;    
    end
end
rate_B=r1/size(B_test,1);
r2=0;
for i=1:size(A_test,1)
    if w1*A_test(i,:)'<y0
        r2=r2+1;    
    end
end
rate_A=r2/size(A_test,1);

if rate_A<0.5
    rate_A=1-rate_A;
    rate_B=1-rate_B;
end
end
function [rate_A,rate_B]=test2(w1,y0,A_test,B_test)
r1=0;
for i=1:size(B_test,1)
    if w1*B_test(i,:)'>y0
        r1=r1+1;    
    end
end
rate_B=r1/size(B_test,1);
r2=0;
for i=1:size(A_test,1)
    if w1*A_test(i,:)'<y0
        r2=r2+1;    
    end
end
rate_A=r2/size(A_test,1);
if rate_A<0.5
    rate_A=1-rate_A;
    rate_B=1-rate_B;
end
end