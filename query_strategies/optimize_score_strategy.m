function [x_star, context] = optimize_score_strategy(...
    score, problem, model, x_train, y_train, context)

error('Not compatible with this version of agpl yet');

% start with a global-grid optimization
[x_star, context] = pool_max_score_strategy(...
    score, problem, model, x_train, y_train, context);

% refine the initial optimization with a local optimization
d = size(x_train,2);

options = optimoptions('fmincon', 'Display', 'none', ...
    'MaxFunctionEvaluations', 100);

A = []; b = []; Aeq = []; beq = []; non_linear_constraint = [];

score_function = @(x_pool) score(x_pool, model, x_train, y_train, context);

t = tic;
[x_star, max_val, ~, output] = fmincon(score_function, x_star, ...
    A, b, Aeq, beq, zeros(d,1), ones(d,1), non_linear_constraint, options);
time = toc(t);

context.max_acq  = max_val;

end
