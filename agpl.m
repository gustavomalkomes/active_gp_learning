% AGPL is a skeleton code for performing active learning experiments
% 
% Inputs
%   problem
%   model
%   update_models
%   query_strategy
%   label_oracle
%   callback
%
% Outputs
%   x
%   y
%   context

function [x, y, context] = agpl(...
    problem, ...
    model, ...
    query_strategy, ...
    label_oracle, ...
    callback ...
)

% set initial points
% TODO: select initial random points
if isfield(problem, 'initial_y')
    num_init_points = numel(problem.initial_y);
    initial_x = problem.initial_x;
    initial_y = problem.initial_y;
end

x = NaN(num_init_points + problem.budget, size(problem.initial_x,2));
y = NaN(num_init_points + problem.budget, size(problem.initial_y,2));

x(1:num_init_points, :) = initial_x;
y(1:num_init_points, :) = initial_y;   

    
has_pool = isfield(problem, 'has_pool') && problem.has_pool;
verbose  = isfield(problem, 'verbose') && problem.verbose;
problem.verbose = verbose;
select_random_points_if_empty = true;

total_time_update      = NaN(problem.budget,1);
total_time_acquisition = NaN(problem.budget,1);
total_time_oracle      = NaN(problem.budget,1);

context                = [];

for i = 1:problem.budget
    % update models with current observations
    x_train = x(1:num_init_points+i-1, :);
    y_train = y(1:num_init_points+i-1, :);
    
    tstart_update             = tic;
    model = model.update(x_train, y_train);
    time_update               = toc(tstart_update);
    total_time_update(i)      = time_update;
    
    % select location of next observation using the query strategy
    tstart_acquisition        = tic;
    [chosen_x_star, context]  = query_strategy(...
        problem, model, x_train, y_train, context...
    );
    time_acquisition          = toc(tstart_acquisition);
    total_time_acquisition(i) = time_acquisition;
    
    % force to choose random points if chosen_x_star is empty 
    if has_pool && isempty(chosen_x_star) && select_random_points_if_empty
        chosen_x_star = randi(size(problem.x_pool,1));
    end
    
    if has_pool
        chosen_x_star =  problem.x_pool(chosen_x_star,:);
    end    
    
    % observe label at chosen location
    tstart_oracle             = tic;
    this_chosen_label         = label_oracle(...
        problem, x_train, y_train, chosen_x_star...
    );
    time_oracle               = toc(tstart_oracle);
    total_time_oracle(i)      = time_oracle;
    
    % update data    
    x(num_init_points + i,:)  = chosen_x_star;
    y(num_init_points + i,:)  = this_chosen_label;
    
    % call callback function, if defined
    if (nargin > 4) && ~isempty(callback)
        context = callback(problem, model, x_train, y_train, i, context);
    end
end

time.total_time_update      = total_time_update;
time.total_time_acquisition = total_time_acquisition;
time.total_time_oracle      = total_time_oracle;

context.time = time;