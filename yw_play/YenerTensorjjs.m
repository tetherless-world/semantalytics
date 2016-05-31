% @begin Yener_Tensor
% @out Gates_Tensor

% cleaning and closing
clear all;
close all;

%% @begin Load_Dataset
%% @desc Load ANTHA dataset
%% @in ANTHA_spreadsheet @uri file:ANTHAyener.xlsx
%% @out num @as num_0
[num,txt,raw] = xlsread('ANTHAyener.xlsx');
%% @end Load_Dataset

%% @begin Adjust_APGAR
%% @desc Adjust APGAR Scores: Ensure they are between 0-10
%% @in num @as num_0
%% @out num @as num_1
for k = 1:size(num,1)
   if num(k,19) >= 20
       num(k,19) = num(k,19)-20;
   end
   if num(k,20) >= 20
       num(k,20) = num(k,20)-20;
   end      
end
%% @end Adjust_APGAR

%% @begin Choose_Subjects
%% @desc Choose only subjects with all five time points
%% @in num @as num_1
%% @out subj
%% @out ib @as ib_0
%% @out final @as final_0
U = unique(num(:,2)); % get unique subjids
count = histc(num(:,2),U); % count the number of occurrences

subj = U(count==5); % take only subjects with all five time points
ib = find(ismember(num(:,2),subj));
final = num(ib,:); % only subjects with all five time points
break
%% @end Choose_Subjects

%% @begin Start_Missing_Values
%% @desc Impute or remove missing values based on how many are missing
%% @in subj
%% @in final @as final_0
%% @in ib @as ib_0
%% @out ib @as ib_1
for p = 1:size(subj,1) % interate through the total number of subjects
    k = subj(p); % what subject number?
   ib = find(ismember(final(:,2),k)); % dealing with one subject
%% @end Start:_Missing_Values

%% @begin Count_Missing
%% @desc Count missing heights, BMIs and weights
%% @in final @as final_0
%% @in ib @as ib_1
%% @out heights
%% @out bmis
%% @out weights
%% @out times
%% @out where
   heights = final(ib,5); % heights for that subject
   bmis = final(ib,6); % bmis for that subject
   weights = final(ib,4); % weights for that subject
   
   times = final(ib,3); %times for that subject
   where = isnan(final(ib,5)); % which values are missing?
   count = sum(where); % how many are missing?
%% @end Count_Missing

%% @begin Impute_One_Missing
%% @desc Use regression model to impute for one missing time point
%% @in heights
%% @in bmis
%% @in weights
%% @in times
%% @in where
%% @in final @as final_0
%% @in ib @as ib_1
%% @out final @as final_1
   if count == 1 % only 1 missing time point
     p1 = polyfit(times(~where),heights(~where),1); % linear fit
     p2 = polyfit(times(~where),heights(~where),2); % quad fit
     
     y1 = polyval(p1,times(~where));
     r1 = rsq(heights(~where),y1); % linear r^2?
     
     y2 = polyval(p2,times(~where));
     r2 = rsq(heights(~where),y2); % quad r^2?
     
     if r1>=r2 % if linear better than quad
         heights(where) = polyval(p1,times(where));
         bmis(where) = weights(where)./((heights(where)/100).^2);
     else % if quad better than linear
         heights(where) = polyval(p2,times(where));
         bmis(where) = weights(where)./((heights(where)/100).^2);
     end
     final(ib,5) = heights;
     final(ib,6) = bmis;
%% @end Impute_One_Missing

%% @begin Impute_Two_Missing
%% @desc Use linear fit to impute for two missing_time_points
%% @in heights
%% @in bmis
%% @in weights
%% @in times
%% @in where
%% @in final @as final_0
%% @in ib @as ib_1
%% @out final @as final_2
   elseif count == 2 % 2 missing time points - linear fit
       p1 = polyfit(times(~where),heights(~where),1);
       
       miss = find(where == 1); % where are the missing values
       heights(miss(1)) = polyval(p1,times(miss(1)));
       heights(miss(2)) = polyval(p1,times(miss(2)));
       
       bmis(where) = weights(where)./((heights(where)/100).^2);
       
       final(ib,5) = heights;
       final(ib,6) = bmis;
%% @end Impute_Two_Missing

%% @begin Remove_More_Than_Two_Missing
%% @desc Remove subjects with more than two missing time points
%% @in final @as final_0
%% @in ib @as ib_1
%% @out final @as final_3
   else % get rid of subjects >2 missing time points
       final(ib,:) = [];
   end
%% @end Remove_More_Than_Two_Missing
end

%% @begin Write_final
%% @in final @as final_1
%% @in final @as final_2
%% @in final @as final_3
%% @out test_spreadsheet_final @uri file:ANTHAyenerfinalmine.xlsx
%% @end Write_final

%% @begin Loading_data_for_tensor
%% @in test_spreadsheet_final @uri file:ANTHAyenerfinalmine.xlsx
%% @out num @as num_2
[num,txt,raw] = xlsread('ANTHAyenerfinalmine.xlsx');
%% @end Loading_data_for_tensor

%% @begin Getting_characteristics
%% @in num @as num_2
%% @out chara
[~,ia,ic] = unique(num(:,2),'last');
chara = num(ia,[2 11:end]);
%% @end Getting_characteristics

%% @begin Fixing_feeding_n
%% @in chara
%% @out chara @as chara_1
chara(chara(:,3)==90,3) = nan;
%% @end Fixing_feeding_n

%% @begin Fill_in_average_values
%% @in chara @as chara_1
%% @out chara @as chara_2
placeholder = sum(isnan(chara)); % where all the missing values are
%% @end Fill_in_average_values

%% @begin Going_through_all_the_columns
%% @in chara @as chara_2
%% @out chara @as chara_3
for k = 1:size(chara,2) 
%% @end Going_through_all_the_columns

%% @begin Does_column_have_missing_values
%% @in chara @as chara_3
%% @out chara @as chara_4
%% @out C
    if placeholder(k) > 0 
        C = chara(isnan(chara(:,k)),1); 
%% @end Does_column_have_missing_values

%% @begin Determine_missing_values
%% @in chara @as chara_4
%% @in C
%% @out gender
%% @out weights_1
%% @out heights_1
        for c = 1:size(C,1)
            gender = chara(chara(:,1) == C(c),2); % gender of missing value
            weights = transpose(abs(num(num(~ismember(num(:,2),C),3)==1,4)...
                - num(num(:,2)==C(c) & num(:,3)==1,4))); % weight comparison
            heights = transpose(abs(num(num(~ismember(num(:,2),C),3)==1,5)...
                - num(num(:,2)==C(c) & num(:,3)==1,5))); % height comparison
            
            together = [weights + heights; chara(~ismember(chara(:,1),C),2)';...
                chara(~ismember(chara(:,1),C),k)']; % comparison; gender; value
            
            [sorted,idx] = sort(together(1,:)); % sort by comparison value
            
            together = together(:,idx); % organize them by comparison value
            idx = find(together(2,:) == gender,10,'first'); % find the first 10 values
            if ismember(k,[2 3 8 13])
                chara(chara(:,1) == C(c),k) = mode(together(3,idx)); % take their mode
            else
            chara(chara(:,1) == C(c),k) = nanmean(together(3,idx)); % take their mean
            end
        end
%% @end Determine_missing_values
    end
end
% by the end i have a perfect mean matrix

%% @begin Creating_Gates_Tensor
%% @in num @as num_2
%% @in chara @as chara_4
%% @out gates_tensor
gates_tensor = NaN(21382,23,5); 

t = 1; % time counter
for k = [1 123 366 1462 2558]
    logical = (num(:,3) == k);
    gates_tensor(:,:,t) = [chara(:,1) num(logical,4:10) chara(:,2:end)];
    t = t + 1; % increment time counter
end
%% @end Creating_Gates_Tensor
%% @end Yener_Tensor
