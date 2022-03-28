clear all;
close all;
%% server setting
% distributed SVD
% number of features in each vector
feats_list = [100]; % 1k dimention 200 400 600 800 1000
r=1;%components
chunk_list=[2:2:100] ;%2
sigma=0;%2*maxIter*sqrt(4*maxIter/2*delta)/eps
T1=[];
T2=[];
Time=[];
Error=[];
w_list=[];
E_list=[];
E2_list=[];
maxIter=10;
for kk=1:size(feats_list,2)
    feats=feats_list(kk);
    [AA,BB]=synthetic_random(feats);
for k=1:size(chunk_list,2)
    chunks=chunk_list(k);
%     chunkSize = T/chunks; %n_i
% A,B generation
    for w = 1:chunks
        A(:,:,w)=AA;
        B(:,:,w)=BB;
        M(:,:,w)=(squeeze(B(:,:,w))^(-1/2))'*squeeze(A( :, :,w))*squeeze(B(:,:,w))^(-1/2);
    end
%     for w=1:chunks
%         U(:,:,w)=orth(rand(feats));
%         tri(:,:,w)=diag(rand(1,feats));
%         M(:,:,w)=U(:,:,w)'*tri(:,:,w)*U(:,:,w);
%     end
%% global true
    sumA=zeros(feats);
    sumB=zeros(feats);
    for w=1:chunks
        sumA= sumA+A( :, :,w);
        sumB= sumB+B( :, :,w);
    end
    sumM=(sumB^(-1/2))'*sumA*sumB^(-1/2);
    
    [q,w,e]=svd(sumM);

%% central true
    maxF_t=0;
    tic;
    for w=1:chunks
        maxF_t=maxF_t+ M(:,:,w);
    end
%     maxF_t=maxF_t/chunks;  
    [qq,ww,ee]=svd(maxF_t);
    t=toc;
    T1=[T1 t];
     %True=trace(abs(q'*maxF_t*e));%(:,1:r)(:,1:r)
     [U,t]=dsvd(M,chunks,r);
     T2=[T2 t];
%      E_list=[E_list sin(subspace(qq(:,1:r),q(:,1:r)))];  
     E_list=[E_list sin(subspace(qq,q))];
      E2_list=[E2_list sin(subspace(U,q))];  
%     True= ww(1,1);
%     convg=[];
%     convg2=[];%sin(subspace)
%     maxF_old = 0;   
    U_g=(orth(rand(feats,r)));%chunkSize \times r
    tic
    for i=1:maxIter
        for w = 1: chunks
            H1(:,:, w) = squeeze(M(:,:,w))*squeeze(M(:,:,w))'*U_g;%*randn(chunkSize, 1)+sigma*rand(feats,r)/chunkSize
        end    
        K1=zeros(feats,r);      
        for w=1:chunks
            K1 = feats.*squeeze(H1(:,:,w))+K1;
        end
        K1=K1/(feats*chunks);
        [U1,~]=qr(K1,0);
% 
%         maxF=trace(abs(U1'*maxF_t*U1));
%         convg=[convg abs(maxF-maxF_old)];    
%         convg2=[convg2 sin(subspace(U1,U_g))];
        U_g = U1;
%      
%         maxF_old=maxF;  
%         maxF=trace(abs(U1'*maxF_t*U1));
    end
     t=toc;
     Time=[Time t];
%      error=(True-maxF)/True;
%      Error=[Error error];
%      w_err=sin(subspace(U1,q))
%      w_list=[w_list w_err];
     clear A B %M  H1  K1  
end
end
yyplot();



function [A,B]=synthetic_random(d)
%     A_ = randn(d);
%     A = tril(A_,-1)+triu(A_',0);
    A=rand(d);  
    A=100.*(A'+A);%symmetric
    V = diag(rand(1,d));
    U = orth(rand(d));
    B = 100.*(U*V*U');%positive definite
    chol(B);
end