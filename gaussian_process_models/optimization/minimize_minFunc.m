% MINIMIZE_MINFUNC optimize GP hyperparameters with random restart.
%
% This implements GP hyperparameter optimization with random
% restart. Each optimization is accomplished using Mark Schmidt's
% minFunc function:
%
%   http://www.di.ens.fr/~mschmidt/Software/minFunc.html
%
%
% See also MINFUNC.

function [ ...
    best_theta, ...
    best_nlZ, ...
    best_HnlZ, ...
    best_L, ...
    best_posterior, ...
    best_minFunc_output...
    ] ...
    = minimize_minFunc(model, x, y, varargin)

% parse optional inputs
parser = inputParser;

addParameter(parser, 'initial_hyperparameters', []);
addParameter(parser, 'num_restarts', 3);
addParameter(parser, 'minFunc_options', ...
    struct('Display', 'off', ...
    'MaxIter', 500,   ...
    'method', 'lbfgs'));

parse(parser, varargin{:});
options = parser.Results;

if isempty(options.initial_hyperparameters)
    initial_theta = model.prior();
else
    initial_theta = options.initial_hyperparameters;
end


f = @(hyperparameter_values) gp_optimizer_wrapper(...
    hyperparameter_values, ...
    initial_theta, ...
    model.inference_method, ...
    model.mean_function, ...
    model.covariance_function, ...
    model.likelihood, ...
    x, ...
    y ...
    );

[fx, gx, hx] = f(unwrap(initial_theta));
number_of_hyperparameters = numel(unwrap(initial_theta));
assert(numel(fx) == 1)
assert(numel(gx) == number_of_hyperparameters)
assert(numel(hx) == number_of_hyperparameters^2)

best_theta_values = unwrap(initial_theta);
best_nlZ = fx;
best_HnlZ = hx;
best_L = [];
best_posterior = [];
best_minFunc_output = [];

theta = initial_theta;

for i = 1:options.num_restarts
    try
        [theta_values, ~, exitflag, minFunc_output] = ...
            minFunc(f, unwrap(theta), options.minFunc_options);
        [nlZ, ~, HnlZ, post] = f(theta_values);
        L = chol(HnlZ);
    catch ME
        switch ME.identifier
            case 'MATLAB:posdef'
                % ignore this iteration
                theta = model.prior();
                continue
            otherwise
                rethrow(ME)
        end
    end
    
    if exitflag < 0
        nlZ = +inf;
    end
    
    % saving best values
    if (nlZ < best_nlZ)
        best_theta_values = theta_values;
        best_nlZ = nlZ;
        best_HnlZ = HnlZ;
        best_L = L;
        best_posterior = post;
        best_minFunc_output = minFunc_output;
    end
    
    % sample hyperparameters
    theta = model.prior();
end

if isnan(best_nlZ) || isinf(best_nlZ)
    error('AGPL:minimize_minFunc', 'Objective NaN or Inf')
end

% rewrap hyperparameters
best_theta = rewrap(theta, best_theta_values);

end