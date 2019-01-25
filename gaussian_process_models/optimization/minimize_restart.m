function [theta,best_iter,best_nlZ] = minimize_restart(model,x,y,varargin)

% parse optional inputs
parser = inputParser;

addParameter(parser, 'initial_hyperparameters', []);
addParameter(parser, 'num_restarts', 10);
addParameter(parser, 'length', 500);
addParameter(parser, 'verbosity', 0);

parse(parser, varargin{:});
options = parser.Results;

if (isempty(options.initial_hyperparameters))
    initial_hyperparameters = model.prior();
else
    initial_hyperparameters = options.initial_hyperparameters;
end

[best_theta, best_nlZ,best_iter] = ...
    minimize(...
        initial_hyperparameters, ...
        model.prediction_method, ...
        options.length, ...
        model.inference_method, ...
        model.mean_function, ...
        model.covariance_function, ...
        model.likelihood, ...
        x, ...
        y ...
    );

for i = 1:options.num_restarts
    theta = model.prior();
    
    [theta_values, nlZ,iter] = ...
        minimize(theta, @gp, options.length, model.inference_method, ...
        model.mean_function, model.covariance_function, model.likelihood, x, y);
    
    if (nlZ(end) < best_nlZ(end))
        best_nlZ = nlZ;
        best_theta = theta_values;
        best_iter  = iter;
    end
end

theta = best_theta;
