classdef GpModel
    properties
        name % string to represent the model
        %
        % GPML and GPML extensions function handles
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
        L % choles
        nlZ % the negative log MLE/MAP
        HnlZ % a struct containing the Hessian of the negative log MAP/MLE
        posterior % posterior structu as in GPML
        %
        % attributes logging time
        %
        optimization_time % total optimization time
        hessian_time % total time computing the hessian
        %
        % train options
        %
        optimization_options
    end
    methods
        % Constructor
        function obj = GpModel(covariance, hyperpriors)
            
            if isempty(hyperpriors)
                hyperpriors = Hyperpriors();
            end
            
            assert(isa(hyperpriors, 'Hyperpriors'), 'Not a hyperpriors');
            
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
            obj.optimization_options.num_restarts = 2;
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
        function [ymu, ys2, fmu, fs2, log_probabilities] = ...
                predict(obj, x_train, y_train, x_star)
            
            if isempty(obj.theta)
                error('AGPL:GpModel:prediction:thetaIsEmpty', ...
                    'You must train before predict');
            end
            
            if isempty(y_train)
                y_train = obj.posterior;
            end
            
            [ymu, ys2, fmu, fs2, log_probabilities] = obj.prediction_method(...
                obj.theta, ...
                obj.inference_method, ...
                obj.mean_function, ...
                obj.covariance_function, ...
                obj.likelihood, ...
                x_train, ...
                y_train, ...
                x_star);
        end
        
        % update rotine 
        function [obj] = update(obj, x_train, y_train)
            obj = obj.train(x_train, y_train);
        end
        
        % train hyperparameters using an (external) optimization procedure
        function [obj] = train(obj, x_train, y_train)
            
            if obj.optimization_options.display > 0
                number_of_parameters = numel(obj.theta);
                fprintf('\n%s data %-3d x %-3d cov_hyp: %-3d     cov_name: %-50s\n', ...
                    datestr(now,'yy-mmm-dd-HH:MM'), ...
                    size(x_train,1), size(x_train,2), ...
                    number_of_parameters, ...
                    obj.covariance.name);
            end
            
            initial_theta = obj.theta;
            
            start_opt_time = tic;
            try
                [new_theta, new_nlZ, opt_output] = ...
                    minimize_minFunc(obj, x_train, y_train, ...
                    'initial_hyperparameters', initial_theta, ...
                    'num_restarts', obj.optimization_options.num_restarts, ...
                    'minFunc_options', obj.optimization_options.minFunc_options);
            catch ME
                switch ME.identifier
                    case 'AGPL:minimize_minFunc'
                        warning('MinFunc minimize failed. Using minimize.m'); 
                        try
                            [new_theta, new_nlZ, opt_output] = ...
                                minimize_minimize(obj, x_train, y_train, ...
                                'initial_hyperparameters', initial_theta, ...
                                'num_restarts', obj.optimization_options.num_restarts, ...
                                'minimize_options', obj.optimization_options.minimize_options);
                        catch
                            error('AGPL:GpModel:train','Optimization failed'); 
                        end
                    otherwise
                        rethrow(ME)
                end                
            end
            obj.optimization_time = toc(start_opt_time);
            
            % updating (hyp)parameters
            obj.theta                   = new_theta;
            obj.negative_log_likelihood = new_nlZ;
            obj.theta_hessian           = opt_output.HnlZ;
            obj.theta_posterior         = opt_output.post;
            
            if obj.optimization_options.display > 0
                fprintf('%s lZ: %-4.2f     cov_hyp: %-3d\tcov_name: %-50s\n', ...
                    datestr(now,'yy-mmm-dd-HH:MM'), -obj.nlZ, ...
                    str2num(feval(obj.covariance_function{:})), ...
                    obj.covariance.name);
            end
            
            if obj.optimization_options.display > 1 ...
                    && ~isempty(opt_output)
                
                if isfield(opt_output, 'iterations')
                    fprintf('\t\tcov_hyp: %-3d   iter:%-4d        fun_count:%-4d \t\t %4.2f seconds\n', ...
                        str2num(feval(obj.covariance_function{:})), ...
                        minFunc_output.iterations, ...
                        opt_output.funcCount, ...
                        opt_output);
                end
                
                fprintf('\t\tcov_hyp: %-3d   H_time:%-4.2f seconds \t\t\t total_time: %4.2f seconds\n', ...
                    str2num(feval(obj.covariance_function{:})), ...
                    hessian_computation_time, ...
                    hessian_computation_time+obj.optimization_time);
                
                if isfield(minFunc_output, 'firstorderopt')
                    fprintf('\t\tcov_hyp: %-3d   opt: %-2.8f \t\t\t\t %s\n\n', ...
                        str2num(feval(obj.covariance_function{:})), ...
                        minFunc_output.firstorderopt, ...
                        minFunc_output.message);
                end
            end
        end
        
        % compute laplace approximation of the model evidence
        % Laplace approximation to model evidence is
        %
        %  log Z ~ L(\hat{\theta}) + (d / 2) log(2\pi) - (1 / 2) log det H
        %
        % where d is the dimension of \theta and H is the
        % negative Hessian of L evaluated at \hat{\theta}
        function result = log_evidence(obj)
            
            % model L is the cholesky decomposition of the Hessian matrix
            d = size(obj.posterior.L,1);
            
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
            result = -obj.nlZ + d * half_log_2pi - ...
                sum(log(diag(obj.posterior.L)));
        end
    end
end
