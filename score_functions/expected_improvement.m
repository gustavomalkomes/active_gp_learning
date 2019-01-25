function ei = expected_improvement(x_pool, model, ...
    x_train, y_train, ~)

y_min = min(y_train);

[~, ~, mu,cov] = model.predict(x_train, y_train, x_pool);

% make sure that cov > 0
cov((cov<0)) = 0;
sigma = sqrt(cov);

% compute expected improvement
delta = (y_min - mu);
u     = delta./sigma;
u_pdf = normpdf(u);
u_cdf = normcdf(u);

ei        = delta .* u_cdf + sigma .* u_pdf;
ei(ei<0) = 0; 

end