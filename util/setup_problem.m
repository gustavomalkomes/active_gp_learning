function [problem,label_oracle] = setup_problem(...
    fun, budget_per_dim, grid_points_factor, num_init_points)


f            = @(x) feval(fun, x);
fun_info     = f([]);

if contains(fun,'grid')
    x_pool         = fun_info.x_pool;
    optimum        = fun_info.min;
    lb             = min(x_pool);
    ub             = max(x_pool);
    label_oracle   = @(problem,x_train, y_train, x_star) f(x_star);
    
    d              = size(lb,2);

    n_grid_points  = size(x_pool,1);
    used           = false(size(x_pool,1),1);
    init_idx       = randperm(n_grid_points,num_init_points);
    used(init_idx) = true;
    
    x              = x_pool(init_idx,:);
    y              = zeros(num_init_points,1);
    
    for k = 1:num_init_points
        y(k) = label_oracle([], [], [], init_idx(k));
    end
    
    problem.x_pool               = x_pool;
    problem.used                 = used;
    problem.initial_x            = x;
    problem.initial_y            = y;
    problem.has_pool             = true;
    problem.optimum              = optimum;
    problem.budget               = budget_per_dim*d;
    
    return
end

optimum         = fun_info.min;
lb              = fun_info.lb;
ub              = fun_info.ub;
map_to_problem  = @(x) bsxfun(@plus, bsxfun(@times, x, (ub-lb)), lb);
label_oracle    = @(problem,x,y,x_star) f(map_to_problem(x_star));

d                 = size(lb,2);
num_points        = grid_points_factor*d^2;

p                 = sobolset(d);
x_pool            = net(p, num_points);
p                 = sobolset(d,'Skip',1e2,'Leap',1e5);
p                 = scramble(p,'MatousekAffineOwen');
x_pool            = [x_pool; net(p, num_points)];

num_grid_points   = size(x_pool,1);

used              = false(num_grid_points,1);
init_idx          = randperm(num_grid_points,num_init_points);
used(init_idx)    = true;

x                 = x_pool(init_idx,:);
y                 = zeros(num_init_points,1);


for k = 1:num_init_points
    y(k) = label_oracle([], [], [], x(k,:));
end

problem.optimum   = optimum;
problem.budget    = budget_per_dim*d;
problem.x_pool    = x_pool;
problem.used      = used;
problem.initial_x = x;
problem.initial_y = y;