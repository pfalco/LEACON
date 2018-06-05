function [ output_args ] = fade(data, f_th, K,f_s)
% It computes the FADE descriptor. 
% data is the matrix describing the action
% f_th is the chosen frequency threshold
% K is the number of points and influences the frequency resolution
% f_s is the sampling frequency

    v1 = data;
    
    v1(find(isnan(v1(:,1))),:) = [];
            
    L=size(data,1);
    %N = 2^nextpow2(L);
    N=L;
    i_f= ceil((f_th * (N/2+1))/f_s); %it allows us to stop at f_th Hz
    
    if(i_f<3)
        i_f = 3;
    end
    
    vf = abs(fft(v1,N));
%     f = f_s*(0:(N/2))/N;
    f = f_s/2*linspace(0,1,N/2+1);
    %Df=f_s/N; %Initial Frequency resolution 
   % Df_new=f_th/K; %Desired frequency resolution
    f=f(1:i_f);
    v2=vf(1:i_f,:);  
    %v2abs = resampleNew(v2, f, 1/Df_new);
    v2abs = interp1(f, v2, linspace(0,f(end),K));
    [coeff,~,~,~,explained,~] = pca(v2abs); 
    coeffexpprod =  bsxfun(@times, coeff, explained');
    output_args = coeffexpprod(:, 1)'; %the first column explains most of the data
end
