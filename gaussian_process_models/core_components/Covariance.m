classdef Covariance
    properties
        name % string to represent the model
        is_base % boolean to define if it is a base covariance
        rnd_code % random numbers to figure out if two models are the same
        
        function_handle % covariance function handle
        priors % covariance prior function handle
    end
    methods
        
        function obj = Covariance(name, is_base, rnd_code, ...
                function_handle, priors)
            obj.name = name;
            obj.is_base = is_base;
            obj.rnd_code = rnd_code;
            obj.function_handle = function_handle;
            obj.priors = priors;
        end
        
        % mask kernel
        function obj = mask(o1, dimension)
            new_name = [o1.name, '_', num2str(dimension)];
            new_is_base = true;
            
            rnd_code_maximum_length = numel(o1.rnd_code);
            new_rnd_code = rand(1,rnd_code_maximum_length);
            new_rnd_code = new_rnd_code.*randi(...
                rnd_code_maximum_length,1,rnd_code_maximum_length);
            
            new_function_handle = {@mask_covariance, {dimension, ...
                o1.function_handle}};
                        
            new_priors = o1.priors;
            
            % create new covariance
            obj = Covariance(new_name, new_is_base, new_rnd_code, ...
                new_function_handle, new_priors);
        end
        
        % overload addition opertation '+'
        function obj = plus(o1, o2)
            new_name = ['(', o1.name, '+', o2.name, ')'];
            new_is_base = false;
            new_rnd_code = o1.rnd_code + o2.rnd_code;
            
            cov1 = o1.function_handle; cov2 = o2.function_handle;
            new_function_handle = {@sum_covariance,{cov1,cov2}};
            
            new_priors = {o1.priors{:}, o2.priors{:}};
            
            % create new covariance
            obj = Covariance(new_name, new_is_base, new_rnd_code, ...
                new_function_handle, new_priors);
        end
        % overload multiplication opertation '*'
        function obj = mtimes(o1,o2)
            new_name = ['(', o1.name, '*', o2.name, ')'];
            new_is_base = false;
            new_rnd_code = o1.rnd_code .* o2.rnd_code;
            
            cov1 = o1.function_handle; cov2 = o2.function_handle;
            new_function_handle = {@prod_covariance,{cov1,cov2}};
            
            new_priors = {o1.priors{:}, o2.priors{:}};
            
            % create new covariance
            obj = Covariance(new_name, new_is_base, new_rnd_code, ...
                new_function_handle, new_priors);
        end
        % overload equality operation '='
        function r = eq(o1, o2) % overload equality
            tolerance = 1e-10;
            r = max(abs(o1.rnd_code-o2.rnd_code)) < tolerance;
        end
        
        % some helper functions
        function samples = get_hyperparameters_sample(obj)
            hyperpriors.cov = obj.priors;
            covariance_hyperprior = get_prior(@independent_prior, ...
                hyperpriors);
            % sample from the covariance hyperprior
            samples = covariance_hyperprior();
            samples = samples.cov;
        end
        
        function r = is_base_kernel(obj)
            r = obj.is_base;
        end
    end
    
    
    methods (Static)
        
        function covariance = ...
                str2covariance(covariance_name, hyperpriors, varargin)
            
            if isempty(hyperpriors)
                hyperpriors = Hyperpriors();
            end
            
            assert(isa(hyperpriors, 'Hyperpriors'), 'Not a hyperpriors');
            
            rnd_code_maximum_length = 100;
            
            is_base = true;
            rnd_code = rand(1,rnd_code_maximum_length);
            rnd_code = rnd_code.*randi(...
                rnd_code_maximum_length,1,rnd_code_maximum_length);
            
            switch covariance_name
                case 'SEard'
                    cov.fun = {@ard_sqdexp_covariance};
                    d = varargin{1};
                    hyperprior_cell = cell(d+1,1);
                    for i = 1:d
                        hyperprior_cell{i} = ...
                            hyperpriors.gaussian_prior('length_scale');
                    end
                    hyperprior_cell{end} = ...
                        hyperpriors.gaussian_prior('output_scale');
                    cov.priors = hyperprior_cell;
                    
                case 'SE'
                    cov.fun = {@isotropic_sqdexp_covariance};
                    cov.priors = {...
                        hyperpriors.gaussian_prior('length_scale'), ...
                        hyperpriors.gaussian_prior('output_scale'), ...
                        };
                    
                case 'M1'
                    cov.fun = {@isotropic_matern_covariance12};
                    cov.priors = {...
                        hyperpriors.gaussian_prior('length_scale'), ...
                        hyperpriors.gaussian_prior('output_scale'), ...
                        };
                    
                case 'PER'
                    cov.fun = {@periodic_covariance};
                    cov.priors = {...
                        hyperpriors.gaussian_prior('p_length_scale'), ...
                        hyperpriors.gaussian_prior('p'), ...
                        hyperpriors.gaussian_prior('output_scale'), ...
                        };
                    
                case 'LIN'
                    cov.fun = {@linear_covariance};
                    cov.priors = {...
                        hyperpriors.gaussian_prior('output_scale')
                        };
                    
                case 'RQ'
                    cov.fun = {@isotropic_rq_covariance};
                    cov.priors = {...
                        hyperpriors.gaussian_prior('length_scale'), ...
                        hyperpriors.gaussian_prior('output_scale'), ...
                        hyperpriors.gaussian_prior('alpha'), ...
                        };
            end
            
            function_handle = cov.fun;
            priors = cov.priors;
            
            covariance = Covariance(...
                covariance_name, ...
                is_base, ...
                rnd_code, ...
                function_handle, ...
                priors...
                ); 
        end
        
    end
end