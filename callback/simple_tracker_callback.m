function context = simple_tracker_callback(problem, ~, ~, y, i, context)

num_init_points = numel(problem.initial_y);
y_first = min(problem.initial_y);
y_best  = min([problem.initial_y; y]);

gap     = NaN;
if isfield(problem, 'optimum') && ~isnan(problem.optimum)
    if abs(y_first - problem.optimum) < 1e-12
        gap = 1;
    else
        gap = (y_first - y_best)/(y_first - problem.optimum);
    end
end

max_acq     = context.max_acq;
context.gap = gap;

if problem.verbose
    fprintf(...
        '%s %s iter: %3d; last: %6.5f; min: %6.5f; acq: %6.6f; gap: %1.5f\n', ...
        datestr(now,'yy-mmm-dd-HH:MM'), ...
        problem.name, ...
        i, ...
        y(num_init_points+i), ...
        y_best, ...
        max_acq, ...
        gap);
end


if isfield(context, 'all_gap')
    context.all_gap = [context.all_gap; gap];
else
    context.all_gap = gap;
end

end