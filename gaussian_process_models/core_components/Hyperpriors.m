classdef Hyperpriors
    properties
        data_noise
        length_scale_mean
        length_scale_var
        output_scale_mean
        output_scale_var
        p_length_scale_mean
        p_length_scale_var
        p_mean
        p_var
        alpha_mean
        alpha_var
        lik_noise_std_mean
        lik_noise_std_var
        mean_offset_mean
        mean_offset_var
    end
    methods
        % Constructor
        function obj = Hyperpriors(data_noise)
            if nargin == 0
                data_noise = 0.01;
            end

            obj.length_scale_mean    = log(0.1);
            obj.length_scale_var     = 1;

            obj.output_scale_mean    = log(0.4);
            obj.output_scale_var     = 1;

            obj.p_length_scale_mean  = log(2);
            obj.p_length_scale_var   = 0.5;

            obj.p_mean               = log(0.1);
            obj.p_var                = 0.5;

            obj.alpha_mean           = log(0.05);
            obj.alpha_var            = 0.5;

            obj.lik_noise_std_mean   = log(data_noise);
            obj.lik_noise_std_var    = 1;

            obj.mean_offset_mean     = 0;
            obj.mean_offset_var      = 1;
        end
        function prior = gaussian_prior(obj, hyperparameter_name)
            mean_value = eval(['obj.', hyperparameter_name, '_mean']);
            variance_value = eval(['obj.', hyperparameter_name, '_var']);
            prior = get_prior(@gaussian_prior, mean_value, variance_value);
        end
    end
end