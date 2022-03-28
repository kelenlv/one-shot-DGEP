clear all
close all

%% gene data
[X,Y]=load_genedata();
T=size(X,2);
feats1=size(X,1);
feats2=size(Y,1);
%% language data
% [X,Y]=load_languagedata();
% T=size(X,2);
% feats1=size(X,1);
% feats2=size(Y,1);

%% settings
chunk_list=[1];% 1 2 4 8
r=1;
p=0.9; %sample rate
maxIter=10;
sigma=0;
Time=[];
Error1=[];
Error2=[];
Error3=[];
AP_smp=[];
AP_res=[];
AP_smp0=[];
AP_res0=[];
 
for iter=1:20
for k=1:size(chunk_list,2)
    %% local server
    chunks=chunk_list(k);
    R_T=randperm(T);%T*p
%     X_smp = X(:,R_T(1:floor(T*p)));%round
%     X_res = X(:,R_T(floor(T*p)+1:T));
%     Y_smp= Y(:,R_T(1:floor(T*p)));
%     Y_res = Y(:,R_T(floor(T*p)+1:T));
    X_smp = X(:,R_T(1:48));
    X_res = X(:,R_T(49:T));
    Y_smp= Y(:,R_T(1:48));
    Y_res = Y(:,R_T(49:T));
    chunkSize = 48/chunks;
%     chunkSize = T*p/chunks; %n_i
    for w = 1:chunks
        X_chunk( :, :,w)=X_smp(:,floor(chunkSize*(w-1))+1:floor(chunkSize*w));
        Y_chunk( :, :,w)=Y_smp(:,floor(chunkSize*(w-1))+1:floor(chunkSize*w));
        A(:,:,w)=((squeeze(X_chunk( :, :,w))*squeeze(X_chunk( :, :,w))')^(-1/2))'*squeeze(X_chunk( :, :,w))*squeeze(Y_chunk(:,:,w))'*(squeeze(Y_chunk( :, :,w))*squeeze(Y_chunk( :, :,w))')^(-1/2);
        AA(:,:,w)=gen_block(feats1,X_smp,Y_smp);
    end
    %% calculate W_1
    sum_max=0;
    sum_max2=0;
    for w=1:chunks
        sum_max=sum_max+ A(:,:,w);
        sum_max2=sum_max2+ AA(:,:,w);
    end
    sum_max=sum_max/chunks;
    sum_max2=sum_max2/chunks;
     [W_1,ww,e]=svd(sum_max);
     [W_2,~,~]=svd(sum_max2);
     sum_trace=abs(trace(W_1(:,1:r)'*sum_max*e(:,1:r)));
     AP_smp0=[AP_smp0 AveragePrecision_for_genedata(X_smp, (X_smp*X_smp')^(-1/2)*W_1, Y_smp, (Y_smp*Y_smp')^(-1/2)*e)];
     AP_res0=[AP_res0 AveragePrecision_for_genedata(X_res, (X_res*X_res')^(-1/2)*W_1, Y_res, (Y_res*Y_res')^(-1/2)*e)];
     
     %% calculating true value   
     maxF_t=((X_smp*X_smp')^(-1/2))'*X_smp*Y_smp'*(Y_smp*Y_smp')^(-1/2);
    %     maxF_t=maxF_t/chunks;
     [W_g,ww,ee]=svd(maxF_t,'econ');
     True_trace=abs(trace(W_g(:,1:r)'*maxF_t*ee(:,1:r)));
%     True=ww(1,1); 
%      True=trace(abs(q'*maxF_t*e));%(:,1:r)(:,1:r)
%     error1=(True_trace-sum_trace)/True_trace;
    
    %% central server   
    %components
    U_g=W_1(:,1:r);%(orth(rand(feats1,r)));%chunkSize\times r
    V_g=e(:,1:r);%(orth(rand(feats2,r)));
    UU_g=W_2(:,1:r);%(orth(rand(feats1+feats2,r)));
%     U_g = W_g(:,1:r);
%     V_g = ee(:,1:r);
    convg=[];
    convg2=[];
    maxF_old = 0;
    maxF_old2 = 0;
    Err=[];
    Err2=[];
    tic
    for i=1:maxIter
%         if mod(i,500)==0
%             fprintf("#iter=%d,chunks=%d\n",i,k-1);
%         end
        for w = 1: chunks
            H1(:,:, w) = squeeze(A(:,:,w))*squeeze(A(:,:,w))'*U_g+sigma*rand(feats1,r)/chunkSize;%*randn(chunkSize, 1)
            H2(:,:,w) = squeeze(A(:,:,w))'*squeeze(A(:,:,w))*V_g+sigma*rand(feats2,r)/chunkSize;%*randn(chunkSize, 1)
            H3(:,:,w) = squeeze(AA(:,:,w))*UU_g+sigma*rand(feats1+feats2,r)/chunkSize;%*randn(chunkSize, 1)
        end
        K1=zeros(feats1,r);
        K2=zeros(feats2,r);
        K3=zeros(feats1+feats2,r);
        for w=1:chunks
            K1 = feats1.*squeeze(H1(:,:,w))+K1;
            K2 = feats2.*squeeze(H2(:,:,w))+K2;
            K3 = (feats1+feats2).*squeeze(H3(:,:,w))+K3;
        end
        K1=K1/(feats1*chunks);
        K2=K2/(feats2*chunks);
        K3=K3/((feats1+feats2)*chunks);
        [U1,~] = qr(K1,0);
        [V1,~] = qr(K2,0);
        [UU1,~] = qr(K3,0);

        maxF_trace=abs(trace(U1'*sum_max*V1));
        maxF_trace2=abs(trace(UU1'*sum_max2*UU1));
        convg=[convg abs(maxF_trace-maxF_old)];
        convg2=[convg2 abs(maxF_trace2-maxF_old)];
    %     if abs(maxF-maxF_old)<1e-4%/abs(maxF)
    %         i=maxIter+1;
    %         break;
    %     end
    %      fprintf('#F=%f\n',norm(maxF,'fro'));
       
        U_g = U1;
        V_g = V1; 
        UU_g = UU1;
       
        maxF_old=maxF_trace;  
        maxF_old2=maxF_trace2;  
       % maxF=trace(abs(U1'*maxF_t*V1));
        err_iter=(True_trace-maxF_trace)/True_trace;
        err_iter2=(True_trace-maxF_trace2)/True_trace;
        Err=[Err err_iter];
        Err2=[Err2 err_iter2];
    end
    %% evaluating
    t=toc;
%      plot(convg')
%     error2=(sum_trace-maxF_trace)/sum_trace;
%     error3=(True_trace-maxF_trace)/True_trace;
%     Time=[Time t];
%     Error1=[Error1 error1];
%     Error2=[Error2 error2];
%     Error3=[Error3 error3];
    AP_smp=[AP_smp AveragePrecision_for_genedata(X_smp, (X_smp*X_smp')^(-1/2)*U_g, Y_smp, (Y_smp*Y_smp')^(-1/2)*V_g)];
    AP_res=[AP_res AveragePrecision_for_genedata(X_res, (X_res*X_res')^(-1/2)*U_g, Y_res, (Y_res*Y_res')^(-1/2)*V_g)];
%     fprintf('#iter=%d,acc_tr=%f ',iter,AveragePrecision_for_genedata(X_smp, (X_smp*X_smp')^(-1/2)*U_g, Y_smp, (Y_smp*Y_smp')^(-1/2)*V_g))
%     fprintf('#iter=%d,acc_te=%f\n',iter,AveragePrecision_for_genedata(X_res, (X_res*X_res')^(-1/2)*U_g, Y_res, (Y_res*Y_res')^(-1/2)*V_g))
    
%     AveragePrecision(X, U1, Y, V1)
    
    clear X_chunk Y_chunk A AA H1 H2 H3 K1 K2 K3
    
end
end
%%
function [X,Y]=load_coildata()
load('./data/coil20threeview.mat')
X1=NormalizeFea(X1);
X=X1';
Y=label';
clear label X1 X2 X3 
end

% function [X,Y]=load_genedata()
% load('./data/Colon.mat')%Brain Lymphoma Colon Prostate Srbct
% % fea=NormalizeFea(fea);
% X=fea';
% Y=gnd';
% clear fea gnd 
% end

function [X,Y]=load_genedata()
load('./data/Lymphoma.mat')%Brain Leukemia Colon Prostate Srbct
% fea=NormalizeFea(fea);
X=fea';
Y=gnd';
clear fea gnd 
end
function [X,Y]=load_languagedata()
load('./data/mydata.mat')
X=a.Franch;
X=X(1:500,:)';
Y=a.English;
Y=Y(1:500,:)';
clear a
end
function [Y1,Y2]=synthetic_random(feats,chunkSize)
%     s=0.1*feats;
    Y1=randn(1).*rand(feats,chunkSize);%
%     Y1=threshold(Y1,s);
    Y2=randn(1).*rand(feats,chunkSize);%
%     Y2=threshold(Y2,s);
end
function M=gen_block(d,X,Y)
    Block0 = zeros(d,d);
    block0 = zeros(d,1);
    BlockA = [Block0,X*Y';Y*X',0];
    BlockB = [X*X',block0;block0',Y*Y'];
    M = BlockB^(-1/2)*BlockA*BlockB^(-1/2);
end
