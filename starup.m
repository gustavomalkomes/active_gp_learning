% script to load paths, assuming they are in a parent folder

if ~exist('gp.m', 'file')
    addpath(genpath('../gpml-matlab-v3.6'));
end

if ~exist('exact_inference.m', 'file')
    addpath(genpath('../gpml_extensions'));
end

if ~exist('mgp.m', 'file')
    addpath(genpath('../mgp'));
end

if ~exist('minFunc.m', 'file')
    addpath(genpath('../minFunc_2012'));
end

% add active_gp_learning files
addpath(genpath('./'));
