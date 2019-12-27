function [sequencemat] =sequenceEncode2(sequence)
%%对序列进行第二种编码

sequencematrix=zeros(length(sequence),length(sequence{1})*4);
for i=1:length(sequence)
    for j=1:length(sequence{1})
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
    sgrna_sequence=sequencematrix(i,1:92);
    dna_sequence=sequencematrix(i,93:184);
    sequencemat(i,:)=int8(or(sgrna_sequence,dna_sequence));
end 
    
end