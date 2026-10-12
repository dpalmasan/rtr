% Dump the official CEC 2022 DMMOP instances (all 60 environments of the 24 problems) so the
% Python port (src/aumo/problems/dmmop.py, DMMOPWorld(..., instances=DIR)) can replay them exactly.
%
% The environment sequence depends only on the problem's own stream (mt19937ar, seed 0), never on
% the algorithm, so calling ChangeDynamic 59 times reproduces what every competition entry saw.
%
% Usage (MATLAB or MATLAB Online; takes a few seconds):
%   1. git clone https://github.com/MiLabHITSZ/dmmops   (or download the zip)
%   2. dump_dmmop_instances('path/to/dmmops', 'path/to/output_dir')
%   3. copy output_dir/P01.mat ... P24.mat into the repository's data/dmmop/instances/
function dump_dmmop_instances(repo, out)
    addpath(repo);
    addpath(fullfile(repo, 'problem'));
    addpath(fullfile(repo, 'problem', 'Sub_f'));
    if ~exist(out, 'dir'); mkdir(out); end
    for fn = 1:24
        pro = DMMOP(fn);
        n = size(pro.x, 1);
        D = pro.D;
        maxEnv = pro.maxEnv;
        X = zeros(maxEnv, n, D);
        H = nan(maxEnv, n);
        W = nan(maxEnv, n);
        GN = zeros(maxEnv, 1);
        SEL = zeros(maxEnv, n);
        OF = nan(maxEnv, n);
        O = nan(maxEnv, n, D);
        if pro.fun_num > 4
            M = zeros(maxEnv, numel(pro.M), D, D);
        else
            M = zeros(0);
        end
        for e = 1:maxEnv
            if e > 1
                pro.ChangeDynamic();
            end
            X(e, :, :) = pro.x;
            if pro.fun_num <= 4
                H(e, 1:numel(pro.h)) = pro.h;
                W(e, 1:numel(pro.w)) = pro.w;
            else
                for i = 1:numel(pro.M)
                    M(e, i, :, :) = pro.M{i};
                end
            end
            GN(e) = pro.gn;
            SEL(e, 1:numel(pro.selected_idx)) = pro.selected_idx;
            OF(e, 1:numel(pro.of)) = pro.of;
            O(e, 1:size(pro.o, 1), :) = pro.o;
        end
        fun_num = pro.fun_num; change_type = pro.change_type; %#ok<NASGU>
        save(fullfile(out, sprintf('P%02d.mat', fn)), 'X', 'H', 'W', 'M', 'GN', 'SEL', 'OF', 'O', ...
             'D', 'fun_num', 'change_type', '-v7');
        fprintf('P%02d: %d environments, gn in [%d, %d]\n', fn, maxEnv, min(GN), max(GN));
    end
end
