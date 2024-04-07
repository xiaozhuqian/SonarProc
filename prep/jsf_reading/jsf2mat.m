clear;
jsfDir = 'D:\SidescanData_202312\1-raw_data\Waterfall\southsea_60\number2_60\';
%jsfDir = 'D:\south_sea\jsf\';
jsfPath = dir([jsfDir '*.jsf']);
matDir = '.\mat\m12_number2';
for i = 1:length(jsfPath)
    tic;
    [dir1, index, ext]=fileparts(jsfPath(i).name);
    save_name = [index '.mat'];
    %disp(save_name);
    jsf = decodejsf2([jsfDir jsfPath(i).name]);
    %disp('read done');
    save([matDir save_name], 'jsf');
    toc;
    disp(['file:', index, '; time cost:', num2str(toc)]);
end

clear;
jsfDir = 'D:\SidescanData_202312\1-raw_data\Waterfall\sanshan_60\';
%jsfDir = 'D:\south_sea\jsf\';
jsfPath = dir([jsfDir '*.jsf']);
matDir = '.\mat\m12_sanshan';
for i = 1:length(jsfPath)
    tic;
    [dir1, index, ext]=fileparts(jsfPath(i).name);
    save_name = [index '.mat'];
    %disp(save_name);
    jsf = decodejsf2([jsfDir jsfPath(i).name]);
    %disp('read done');
    save([matDir save_name], 'jsf');
    toc;
    disp(['file:', index, '; time cost:', num2str(toc)]);
end

clear;
jsfDir = 'D:\SidescanData_202308\1-raw_data\Waterfall\southsea\number2\';
%jsfDir = 'D:\south_sea\jsf\';
jsfPath = dir([jsfDir '*.jsf']);
matDir = '.\mat\m08_number2';
for i = 1:length(jsfPath)
    tic;
    [dir1, index, ext]=fileparts(jsfPath(i).name);
    save_name = [index '.mat'];
    %disp(save_name);
    jsf = decodejsf2([jsfDir jsfPath(i).name]);
    %disp('read done');
    save([matDir save_name], 'jsf');
    toc;
    disp(['file:', index, '; time cost:', num2str(toc)]);
end
    
    
    
    
    
