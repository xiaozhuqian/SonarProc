% Decoding jsf file and save it to a mat file.
% Input: *.jsf
% Output: *.mat

function jsf2mat(jsfDir, matDir)
    %jsfDir = '.\example_data\';
    %matDir = '.\outputs\mat\';
    jsfPath = dir([jsfDir '*.jsf']); 
    n_file = length(jsfPath); 
   
    if exist(matDir, 'dir')==0 
        mkdir(matDir)
    end

    for i = 1:length(jsfPath)
        tic;
        [~, index, ~]=fileparts(jsfPath(i).name);
        save_name = [index '.mat'];
        disp([num2str(i), 'th/', num2str(n_file), '; name:', index, '; time cost:']);
        jsf = decodejsf2([jsfDir jsfPath(i).name]);
        %disp('read done');
        save([matDir save_name], 'jsf');
        toc;
        %disp([num2str(i), 'th/', num2str(n_file), '; name:', index, '; time cost:', num2str(toc)]);
    end

end

    
