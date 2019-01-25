classdef CovarianceGrammar
    properties
        base_kernels_names
        base_kernels
        dimension
        hyperpriors
    end
    methods
        %
        % Constructor
        %
        function obj = CovarianceGrammar(base_kernels_names, ...
                dimension, hyperprior)
            
            if isempty(hyperprior)
                hyperprior = Hyperpriors();
            end
            
            assert(isa(hyperprior, 'Hyperpriors'))
            
            
            n = numel(base_kernels_names);
            base_kernels = cell(1,n);
            for i = 1:n
                base_kernels{i} = ...
                    Covariance.str2covariance(base_kernels_names{i}, ...
                    hyperprior);
            end
            
            if dimension > 1
                total_new_covariances = n*dimension;
                new_base_kernels = cell(1,total_new_covariances);
                new_base_kernels_names = cell(1, total_new_covariances);
                c = 0;
                for i = 1:n
                    for j = 1:dimension
                        c = c + 1;
                        new_base_kernels{c} = base_kernels{i}.mask(j);
                        new_base_kernels_names{c} = new_base_kernels{c}.name;
                    end
                end
                base_kernels = new_base_kernels;
                base_kernels_names = new_base_kernels_names;
            end
            
            obj.dimension = dimension;
            obj.hyperpriors = hyperprior;
            obj.base_kernels_names = base_kernels_names;
            obj.base_kernels = base_kernels;
            
        end
        
        function new_kernels = expand(obj, kernel)
            number_of_base_kernels = numel(obj.base_kernels);
            new_kernels = cell(1,2*number_of_base_kernels);
            
            for i=1:number_of_base_kernels
                new_kernels{2*i - 1} = ...
                    kernel + obj.base_kernels{i};
                new_kernels{2*i} = ...
                    kernel * obj.base_kernels{i};
            end
        end
        
    end
    
end

