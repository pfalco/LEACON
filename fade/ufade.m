function [ output_args ] = ufade(data, f_th, K,f_s)
% It computes the UFADE descriptor. 
% data is the matrix describing the action, discrete-time indexes as row indexes, joint variable indexes as column indexes
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
    f = f_s/2*linspace(0,1,N/2+1);
    f=f(1:i_f);
    v2=vf(1:i_f,:);  
    v2abs = interp1(f, v2/N, linspace(0,f(end),K));
    
    % Vectorize descriptor
    output_args = reshape(v2abs,[1,size(v2abs,1)*size(v2abs,2)]);
end
