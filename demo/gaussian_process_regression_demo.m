%% Simple examples
x = randn(20, 1);                 % 20 training inputs
y = sin(3*x) + 0.1*randn(20, 1);  % 20 noisy training targets
xs = linspace(-3, 3, 100)';        % 61 test inputs

% first define the hyperprior
hyperpriors = Hyperpriors();
% then choose your kernel
covariance = Covariance.str2covariance('SE', hyperpriors);
% finally, create a model using the kernel
model = GpModel(covariance, hyperpriors);

model = model.train(x,y);

ys = model.predict(x,[],xs);