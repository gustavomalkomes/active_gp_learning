classdef GpModel < handle
    properties
        name % string to represent the model
        %
        % GPML function handles
        %
        covariance % covariance object
        covariance_function % covariance function handle
        inference_method % inference function handle
        likelihood % likelihood function handle
        mean_function % mean function handle
        prediction_method % prediction function handle
        %
        % structures
        %
        hyperpriors % used Hyperpriors
        prior % structure with cov, lik and mean priors
        hyperpriors_parameters % parameters for the hyperpriors
        %
        % attributes saved during training
        %
        theta % lastest model hyperparameters
        negative_log_likelihood % the negative log MLE/MAP
        theta_hessian % a struct containing the Hessian of the negative log MAP/MLE
        theta_hessian_chol % cholesky decomposition of HnlZ
        posterior % posterior structu as in GPML
        %
        % attributes logging time
        %
        optimization_time % total optimization time
        %
        % train options
        %
        optimization_options
    end
    methods
        % Constructor
        function obj = GpModel(covariance, hyperpriors)
            
            if nargin == 0
                hyperpriors = Hyperpriors();
                covariance = Covariance.str2covariance('SE', hyperpriors);
            end

            if isempty(hyperpriors)
                hyperpriors = Hyperpriors();
            end
            
            assert(isa(hyperpriors, 'Hyperpriors'), 'Not a Hyperprior');
            assert(isa(covariance, 'Covariance'), 'Not a Covariance');
            
            obj.name = covariance.name;
            obj.covariance = covariance;
            obj.covariance_function = covariance.function_handle;
            obj.mean_function = {@constant_mean};
            obj.likelihood = @likGauss;
            
            obj.hyperpriors = hyperpriors;
            % construct the prior for lik, mean, cov
            priors = struct();
            priors.cov = covariance.priors;
            priors.lik = {hyperpriors.gaussian_prior('lik_noise_std')};
            priors.mean = {hyperpriors.gaussian_prior('mean_offset')};
            
            % get hyperprior from each individual cov, lik, mean priors
            prior = get_prior(@independent_prior, priors);
            
            % MAP inference
            inference_method = @exact_inference;
            obj.inference_method = {@inference_with_prior, ...
                inference_method, prior};
            
            obj.prior = prior;
            obj.prediction_method = @gp;
            
            % optimization options
            obj.optimization_options.num_restarts = 3;
            obj.optimization_options.display = 2;
            obj.optimization_options.minFunc_options.Display  = 'off';
            obj.optimization_options.minFunc_options.MaxIter  = 1000;
            obj.optimization_options.minimize_options = [];
        end
        %
        % GP functions
        %
        % perform predictions using the prediction_method
        % we assume the prediction method has the same usage
        % as the gp.m function from the GPML toolkit
        % use y_train = [] to reuse the posterior
        function [varargout] = predict(obj, x_train, y_train, x_star)
            
            if isempty(obj.theta)
                error('AGPL:GpModel:prediction:thetaIsEmpty', ...
                    'You must train before predict');
            end
            
            if isempty(y_train)
                y_train = obj.posterior;
            end
            
            if nargin == 4
                % perform predictions
                [ymu, ys2, fmu, fs2, log_probabilities] = ...
                    obj.prediction_method(...
                        obj.theta, ...
                        obj.inference_method, ...
                        obj.mean_function, ...
                        obj.covariance_function, ...
                        obj.likelihood, ...
                        x_train, ...
                        y_train, ...
                        x_star ...
                    );
                varargout = {ymu, ys2, fmu, fs2, log_probabilities};
            elseif nargin == 3
                % compute negative likelihood
                [nlZ, dnlZ] = obj.prediction_method(...
                        obj.theta, ...
                        obj.inference_method, ...
                        obj.mean_function, ...
                        obj.covariance_function, ...
                        obj.likelihood, ...
                        x_train, ...
                        y_train ...
                    );
                 varargout = {nlZ, dnlZ};
            end
        end
        
        function [obj] = update(obj, x_train, y_train)
            % Update rotine for updating a GpModel. 
            % Standard behavior is to call the TRAIN method 
            % but this function can be extended to implement 
            % more complex update rotines.
            %
            % [OBJ] = UPDATE(OBJ, x_train, y_train)
            % 
            obj = obj.train(x_train, y_train);
        end
        
        function [obj] = train(obj, x_train, y_train)
            % Train the hyperparameters of a GPModel with training data
            % x_data and y_train. 
            
            if obj.optimization_options.display > 0                
                num_cov_parameters = str2num(feval(obj.covariance_function{:}));
                fprintf('\n%s data %-3d x %-3d cov_hyp: %-3d     cov_name: %-50s\n', ...
                    datestr(now,'yy-mmm-dd-HH:MM'), ...
                    size(x_train,1), size(x_train,2), ...
                    num_cov_parameters, ...
                    obj.covariance.name);
            end
            
            initial_theta = obj.theta;
            start_opt_time = tic;
            [new_theta, new_nlZ, new_HnlZ, new_L, new_posterior, ...
                optimization_output] = ...
                    minimize_minFunc(obj, x_train, y_train, ...
                    'initial_hyperparameters', initial_theta, ...
                    'num_restarts', obj.optimization_options.num_restarts, ...
                    'minFunc_options', obj.optimization_options.minFunc_options);
            obj.optimization_time = toc(start_opt_time);
            
            % updating (hyp)parameters
            obj.theta = new_theta;
            obj.negative_log_likelihood = new_nlZ;
            obj.theta_hessian = new_HnlZ;
            obj.theta_hessian_chol = new_L;
            obj.posterior = new_posterior;

            if obj.optimization_options.display > 0
                fprintf('%s lZ: %7.3f    cov_hyp: %-3d\tcov_name: %-50s\n', ...
                    datestr(now,'yy-mmm-dd-HH:MM'), -obj.negative_log_likelihood, ...
                    num_cov_parameters, ...
                    obj.covariance.name);
            end
            
            if obj.optimization_options.display > 1 ...
                    && ~isempty(optimization_output)
                
                if isfield(optimization_output, 'iterations')
                    fprintf('\t\tcov_hyp: %-3d   iter:%-4d        fun_count:%-4d \t\t %4.2f seconds\n', ...
                        num_cov_parameters, ...
                        optimization_output.iterations, ...
                        optimization_output.funcCount, ...
                        obj.optimization_time);
                end
                
                if isfield(optimization_output, 'firstorderopt')
                    fprintf('\t\tcov_hyp: %-3d   first-order optimality: %-2.8f \t %s\n\n', ...
                        num_cov_parameters, ...
                        optimization_output.firstorderopt, ...
                        optimization_output.message);
                end
            end
        end
        
        function result = log_evidence(obj)
            % compute laplace approximation of the model evidence
            % Laplace approximation to model evidence is
            %
            %  log Z ~ L(\hat{\theta}) + (d / 2) log(2\pi) - (1 / 2) log det H
            %
            % where d is the dimension of \theta and H is the
            % negative Hessian of L evaluated at \hat{\theta}

            % use the cholesky decomposition of the theta hessian matrix
            L = obj.theta_hessian_chol;
            d = size(L,1);
            
            % computing the log evidence
            %
            % applying the following trick: 
            %   log(det(H))/2 = sum(log(diag(chol(H))))
            %                      
            % log_evidence = -nlZ + (d/2)*log(2*pi) - sum(log(diag(L)));
            %
            % where L = chol(HnlZ)
            
            % precompting log(2\pi) / 2
            half_log_2pi = 0.918938533204673;
            result = -obj.negative_log_likelihood + ...
                    d * half_log_2pi - ...
                    sum(log(diag(L)) ...
                );
        end
    end
end
