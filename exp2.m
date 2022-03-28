clear all;
close all;
%% server setting
% distributed SVD
% number of features in each vector
feats_list = [1000]; % 1k dimention 200 400 600 800 1000
r=1;%components
chunk_list=[2:2:100] ;%2
sigma=0;%2*maxIter*sqrt(4*maxIter/2*delta)/eps
rand('seed',20)
T1=[];
T2=[];
T3=[];
Error=[];
w_list=[];
E_list=[];
E2_list=[];
E3_list=[];

maxIter=10;
for kk=1:size(feats_list,2)
    feats=feats_list(kk);
    for k=1:size(chunk_list,2)
        convg1=[];
        convg2=[];
        chunks=chunk_list(k);
%% A,B generation
        for w = 1:chunks
            [A(:,:,w), B(:,:,w)] = synthetic_random(feats);
            M(:,:,w)=(squeeze(B(:,:,w))^(-1/2))'*squeeze(A( :, :,w))*squeeze(B(:,:,w))^(-1/2);
        end
%% global true
        sumA=zeros(feats);
        sumB=zeros(feats);
        for w=1:chunks
            sumA= sumA+A( :, :,w);
            sumB= sumB+B( :, :,w);
        end
        sumM=(sumB^(-1/2))'*sumA*sumB^(-1/2);
        [q,w,e]=svd(sumM);
%% DGEP-SVD
        maxF_t=0;
        tic;
        for w=1:chunks
            maxF_t=maxF_t+ M(:,:,w);
        end
        maxF_t = maxF_t/chunks;  
        [qq,ww,ee]=svd(maxF_t);
        t=toc;
        T1=[T1 t];
%         True=trace(abs(q'*maxF_t*e));%(:,1:r)(:,1:r)
    %      E_list=[E_list sin(subspace(qq(:,1:r),q(:,1:r)))];  
        E_list=[E_list sin(subspace(qq,q))];   
    %% DGEP-PM-1
        U_g=(orth(rand(feats,r)));%chunkSize \times r
        maxF=trace(abs(U_g'*maxF_t*U_g));
        tic
        for i=1:maxIter
%             for w = 1: chunks
%                 H1(:,:, w) = squeeze(M(:,:,w))*squeeze(M(:,:,w))'*U_g;%*randn(chunkSize, 1)+sigma*rand(feats,r)/chunkSize
%             end    
            K1=zeros(feats,r);      
            for w=1:chunks
                K1 = feats.*squeeze(M(:,:,w))+K1;
            end
            K1=K1/(feats*chunks);
            [U1,~]=qr(K1,0);
            maxF_old=maxF;  
            maxF=trace(abs(U1'*maxF_t*U1));
            convg1=[convg1 abs(maxF-maxF_old)];    
            U_g = U1; 
        end
        t=toc;
        T2=[T2 t];
        E2_list = [E2_list, sin(subspace(U1,q))];
%% DGEP-PM-2
        P_1=(orth(rand(feats,r)));%chunkSize \times r
        P_2=(orth(rand(feats,r)));
%         P_1 = qq;
%         P_2 = ee;
        maxF2=trace(abs(P_1'*maxF_t*P_2));
        tic
        for i=1:maxIter
            for w = 1: chunks
                H1(:,:, w) = squeeze(M(:,:,w))*squeeze(M(:,:,w))'*P_1;
                H2(:,:, w) = squeeze(M(:,:,w))'*squeeze(M(:,:,w))*P_2;%*randn(chunkSize, 1)+sigma*rand(feats,r)/chunkSize
            end    
            K1=zeros(feats,r);  
            K2=zeros(feats,r); 
            for w=1:chunks
                K1 = feats.*squeeze(H1(:,:,w))+K1;
                K2 = feats.*squeeze(H2(:,:,w))+K2;
            end
            K1=K1/(feats*chunks);
            [P1,~]=qr(K1,0);
            [P2,~]=qr(K2,0);
            maxF2_old = maxF2;
            maxF2=trace(abs(P1'*maxF_t*P2));
            convg2=[convg2 abs(maxF2-maxF2_old)];  
            P_1 = P1; 
            P_2 = P2;
        end
        t=toc;
        T3=[T3 t];
        E3_list = [E3_list, sin(subspace(P1,q))];        
        clear A B convg1 convg2 %M  H1  K1  
    end
end
% yyplot();


%% generating self_adjoint CCA
function [BlockA,BlockB]=synthetic_random(d)
%     A_ = randn(d);
%     A = tril(A_,-1)+triu(A_',0);
    Block0 = zeros(d/2,d/2);
    A = rand(d/2);  
    A = 1000.*(A'+A);%symmetric
    BlockA = [Block0,A;A,Block0];
    V = diag(rand(1,d/2));
    U = orth(rand(d/2));
    B = 100.*(U*V*U');%positive definite
    chol(B);
    BlockB = [B,Block0;Block0,B];   
end


