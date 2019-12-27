function [sequencematrix] = sequenceEncode1(sequence)
%%对输入序列进行第一种编码

sequencematrix=zeros(length(sequence),length(sequence{1})*4);%%预分配空间
for i=1:length(sequence)   %样本数
    for j=1:length(sequence{1})   %序列长度
        if sequence{i}(j)=='A'
            sequencematrix(i,(j-1)*4+1:j*4)=[1 0 0 0];
        elseif sequence{i}(j)=='C'
            sequencematrix(i,(j-1)*4+1:j*4)=[0 1 0 0];
        elseif sequence{i}(j)=='G'
            sequencematrix(i,(j-1)*4+1:j*4)=[0 0 1 0];
        elseif sequence{i}(j)=='T'
            sequencematrix(i,(j-1)*4+1:j*4)=[0 0 0 1];
        else 
            sequencematrix(i,(j-1)*4+1:j*4)=[0 0 0 0];
        end
       
    end
end 
    
end

