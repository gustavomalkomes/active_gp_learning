function [x_star_index, context] = pool_max_score_strategy(...
    score, problem, model, x_train, y_train, context)

if isfield(context, 'used')
    used = context.used;
else
    used = problem.used;
end

% score the available points
available_points = problem.x_pool(~used,:);

if isempty(available_points)
    x_star_index = [];
    return
end

scores = score(available_points, model, x_train, y_train, context);

[max_val, index] = max(scores);
original_indices = find(~used);
x_star_index     = original_indices(index);

used(original_indices(index)) = true;

% saving data in context
context.used     = used;
context.max_acq  = max_val;
end
