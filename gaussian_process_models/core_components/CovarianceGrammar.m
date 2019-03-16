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

        function all_kernels = full_expand(obj, level, max_number_of_models)
        % this is a brute-force implementation, use it for level < 4
        
            current_kernels = obj.base_kernels(:);
            if level == 0
                all_kernels = current_kernels;
                return
            end
            
            function repeated = remove_duplicates(new_kernels, all_kernels)
                    repeated = false(1, numel(new_kernels));
                    for k=1:numel(new_kernels)
                        for j=1:numel(all_kernels)
                            if new_kernels{k} == all_kernels{j}
                                repeated(k) = true;
                                break;
                            end
                        end
                    end
            end
            all_kernels = {};
            
            while (level > 0)
                this_level = {};
                number_of_models = numel(all_kernels);
                for i=1:numel(current_kernels)
                    new_kernels = obj.expand(current_kernels{i});
                    repeated = remove_duplicates(new_kernels, this_level);
                    this_level = [this_level, new_kernels(~repeated)];
                    number_of_models = number_of_models + numel(new_kernels(~repeated));
                    if number_of_models > max_number_of_models
                        all_kernels = [all_kernels, this_level];
                        all_kernels = all_kernels(1:max_number_of_models);
                        return
                    end
                end
                current_kernels = this_level;
                all_kernels = [all_kernels, this_level];
                level = level - 1;
            end
        end
        
    end
    
end

