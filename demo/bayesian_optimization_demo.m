% example of simple pool based active learning for optimization

num_points = 1000; % number of points in the pool
d = 1; % input space dimension
budget = 10; % total number points
number_of_initial_points = 5;

% model
hyperpriors = Hyperpriors();
covariance = Covariance.str2covariance('SE', hyperpriors);
model = GpModel(covariance, hyperpriors);

% query strategy
query_strategy = @(problem, models, x_train, y_train, context) ...
    pool_max_score_strategy(@expected_improvement, ...
        problem, models, x_train, y_train, context);

% label oracle
label_oracle = get_x_square_oracle();

% callback function
callback = @simple_tracker_callback;

% setup the problem
pool = sobolset(d,'Skip',1e2,'Leap',1e5);
pool = scramble(pool,'MatousekAffineOwen');
x_pool = net(pool, num_points);

initial_indices = 1:number_of_initial_points;
initial_x = x_pool(initial_indices, :);
initial_y = zeros(number_of_initial_points,1);
for i=1:number_of_initial_points
   initial_y(i) = label_oracle([], [], [], initial_x(i));
end
used = false(num_points,1);
used(initial_indices) = true;

problem.initial_x = initial_x;
problem.initial_y = initial_y;
problem.used = used;
problem.budget = budget;
problem.x_pool = x_pool;
problem.has_pool = true;

% run active gp learning
[x, y, context] = agpl(...
    problem, ...
    model, ...
    query_strategy, ...
    label_oracle, ...
    callback ...
);