clc;
clear;
% cd('D:\Program Files (x86)\Tencent\MatlabR2017b\toolbox');
source_path = 'F:\Datasets\WATReg_data\LPBA40\3-hist'
save_source_path = 'F:\Datasets\WATReg_data\LPBA40\WaveImg\Source\'
WD1_path = 'F:\Datasets\WATReg_data\LPBA40\WaveImg\WD1\';
WD2_path = 'F:\Datasets\WATReg_data\LPBA40\WaveImg\WD2\';
WD3_path = 'F:\Datasets\WATReg_data\LPBA40\WaveImg\WD3\';
% WD4_path = 'F:\Datasets\OASIS1\Tongtong\WaveImg-144x160x184\Train\WD4\'; 
imgDir=dir(fullfile(source_path,'*.nii.gz'));
fullname = {imgDir.name};
for i=1:length(fullname)
    file=fullfile(source_path, fullname(i));
    A = load_nii(file{1});
    %name_split = regexp(char(file), '_', 'split');%s使用“_”分割文件名字
    %name = char(name_split(2))
    name = strrep(fullname(i),'.nii.gz','')%去除后缀名
    %%%%%%%crop and save source image%%%%%%%%%
%     A.img = A.img(11:171, 13:205, 11:171);
%     A.hdr.dime.dim = [3,160,192,160,1,1,1,1];
    %source_img = make_nii(A);
    name = char(name)
    save_nii(A,strcat(save_source_path,name,'_source.nii'));
    %%%% Level decomposition %%%%
    wd1 = wavedec3(A.img,1,'db1');
    LLL1 = wd1.dec{1,1};
%     min = min(LLL1);
%     max = max(LLL1);
%     dst = max - min;
%     LLL1 = (LLL1-min)./dst;
%     A.img = LLL1;
%     A.hdr.dime = [4 91 109 91 1 1 1]
    LLL1_img = make_nii(LLL1);
    save_nii(LLL1_img,strcat(WD1_path,'LLL1\',name,'_LLL1.nii'));
    
    LLH1 = wd1.dec{2,1};
    LLH1_img = make_nii(LLH1);
    save_nii(LLH1_img,strcat(WD1_path,'LLH1\',name,'_LLH1.nii'));
    
    LHL1 = wd1.dec{3,1};
    LHL1_img = make_nii(LHL1);
    save_nii(LHL1_img,strcat(WD1_path,'LHL1\',name,'_LHL1.nii'));
    
    LHH1 = wd1.dec{4,1};
    LHH1_img = make_nii(LHH1);
    save_nii(LHH1_img,strcat(WD1_path,'LHH1\',name ,'_LHH1.nii'));
    
    HLL1 = wd1.dec{5,1};
    HLL1_img = make_nii(HLL1);
    save_nii(HLL1_img,strcat(WD1_path,'HLL1\',name ,'_HLL1.nii'));
    
    HLH1 = wd1.dec{6,1};
    HLH1_img = make_nii(HLH1);
    save_nii(HLH1_img,strcat(WD1_path,'HLH1\',name,'_HLH1.nii'));
    
    HHL1 = wd1.dec{7,1};
    HHL1_img = make_nii(HHL1);
    save_nii(HHL1_img,strcat(WD1_path,'HHL1\',name,'_HHL1.nii'));
    
    HHH1 = wd1.dec{8,1};
    HHH1_img = make_nii(HHH1);
    save_nii(HHH1_img,strcat(WD1_path,'HHH1\',name,'_HHH1.nii'));
    
    %%%% Leve2 decomposition %%%%
    wd2 = wavedec3(LLL1_img.img,1,'db1');
    LLL2 = wd2.dec{1,1};
    LLL2_img = make_nii(LLL2);
    save_nii(LLL2_img,strcat(WD2_path,'LLL2\',name,'_LLL2.nii'));
    
    LLH2 = wd2.dec{2,1};
    LLH2_img = make_nii(LLH2);
    save_nii(LLH2_img,strcat(WD2_path,'LLH2\',name,'_LLH2.nii'));
    
    LHL2 = wd2.dec{3,1};
    LHL2_img = make_nii(LHL2);
    save_nii(LHL2_img,strcat(WD2_path,'LHL2\',name,'_LHL2.nii'));
    
    LHH2 = wd2.dec{4,1};
    LHH2_img = make_nii(LHH2);
    save_nii(LHH2_img,strcat(WD2_path,'LHH2\',name,'_LHH2.nii'));
    
    HLL2 = wd2.dec{5,1};
    HLL2_img = make_nii(HLL2);
    save_nii(HLL2_img,strcat(WD2_path,'HLL2\',name,'_HLL2.nii'));
    
    HLH2 = wd2.dec{6,1};
    HLH2_img = make_nii(HLH2);
    save_nii(HLH2_img,strcat(WD2_path,'HLH2\',name,'_HLH2.nii'));
    
    HHL2 = wd2.dec{7,1};
    HHL2_img = make_nii(HHL2);
    save_nii(HHL2_img,strcat(WD2_path,'HHL2\',name,'_HHL2.nii'));
    
    HHH2 = wd2.dec{8,1};
    HHH2_img = make_nii(HHH2);
    save_nii(HHH2_img,strcat(WD2_path,'HHH2\',name,'_HHH2.nii'));
    
    %%%% Leve3 decomposition %%%%
    wd3 = wavedec3(LLL2_img.img,1,'db1');
    LLL3 = wd3.dec{1,1};
    LLL3_img = make_nii(LLL3);
    save_nii(LLL3_img,strcat(WD3_path,'LLL3\',name,'_LLL3.nii'));
    
    LLH3 = wd3.dec{2,1};
    LLH3_img = make_nii(LLH3);
    save_nii(LLH3_img,strcat(WD3_path,'LLH3\',name,'_LLH3.nii'));
    
    LHL3 = wd3.dec{3,1};
    LHL3_img = make_nii(LHL3);
    save_nii(LHL3_img,strcat(WD3_path,'LHL3\',name,'_LHL3.nii'));
    
    LHH3 = wd3.dec{4,1};
    LHH3_img = make_nii(LHH3);
    save_nii(LHH3_img,strcat(WD3_path,'LHH3\',name,'_LHH3.nii'));
    
    HLL3 = wd3.dec{5,1};
    HLL3_img = make_nii(HLL3);
    save_nii(HLL3_img,strcat(WD3_path,'HLL3\',name,'_HLL3.nii'));
    
    HLH3 = wd3.dec{6,1};
    HLH3_img = make_nii(HLH3);
    save_nii(HLH3_img,strcat(WD3_path,'HLH3\',name,'_HLH3.nii'));
    
    HHL3 = wd3.dec{7,1};
    HHL3_img = make_nii(HHL3);
    save_nii(HHL3_img,strcat(WD3_path,'HHL3\',name,'_HHL3.nii'));
    
    HHH3 = wd3.dec{8,1};
    HHH3_img = make_nii(HHH3);
    save_nii(HHH3_img,strcat(WD3_path,'HHH3\',name,'_HHH3.nii'));
    
%         %%%% Leve4 decomposition %%%%
%     wd4 = wavedec3(LLL3_img.img,1,'db1');
%     LLL4 = wd4.dec{1,1};
%     LLL4_img = make_nii(LLL4);
%     save_nii(LLL4_img,strcat(WD4_path,'LLL4\',num2str(i),'_LLL4.nii'));
%     
%     LLH4 = wd4.dec{2,1};
%     LLH4_img = make_nii(LLH4);
%     save_nii(LLH4_img,strcat(WD4_path,'LLH4\',num2str(i),'_LLH4.nii'));
%     
%     LHL4 = wd4.dec{3,1};
%     LHL4_img = make_nii(LHL4);
%     save_nii(LHL4_img,strcat(WD4_path,'LHL4\',num2str(i),'_LHL4.nii'));
%     
%     LHH4 = wd4.dec{4,1};
%     LHH4_img = make_nii(LHH4);
%     save_nii(LHH4_img,strcat(WD4_path,'LHH4\',num2str(i),'_LHH4.nii'));
%     
%     HLL4 = wd4.dec{5,1};
%     HLL4_img = make_nii(HLL4);
%     save_nii(HLL4_img,strcat(WD4_path,'HLL4\',num2str(i),'_HLL4.nii'));
%     
%     HLH4 = wd4.dec{6,1};
%     HLH4_img = make_nii(HLH4);
%     save_nii(HLH4_img,strcat(WD4_path,'HLH4\',num2str(i),'_HLH4.nii'));
%     
%     HHL4 = wd4.dec{7,1};
%     HHL4_img = make_nii(HHL4);
%     save_nii(HHL4_img,strcat(WD4_path,'HHL4\',num2str(i),'_HHL4.nii'));
%     
%     HHH4 = wd4.dec{8,1};
%     HHH4_img = make_nii(HHH4);
%     save_nii(HHH4_img,strcat(WD4_path,'HHH4\',num2str(i),'_HHH4.nii'));
end



