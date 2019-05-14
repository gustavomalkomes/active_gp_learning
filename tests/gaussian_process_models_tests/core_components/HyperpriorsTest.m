classdef HyperpriorsTest < matlab.unittest.TestCase
    properties
        length_scale_mean_value = log(0.1);
        length_scale_var_value = 1;
        output_scale_mean_value = log(0.4);
        output_scale_var_value = 1;
        p_length_scale_mean_value = log(2);
        p_length_scale_var_value = 0.5;
        p_mean_value = log(0.1);
        p_var_value = 0.5;
        alpha_mean_value = log(0.05);
        alpha_var_value = 0.5;
        lik_noise_std_mean_value = log(0.01);
        lik_noise_std_var_value = 0.1;
        mean_offset_value = 0;
        mean_offset_var_value = 1;
    end
    methods (Test)
        function testConstructorDefaultParameters(testCase)
            hyperpriors = Hyperpriors();
            
            testCase.verifyEqual(hyperpriors.length_scale_mean, ...
                testCase.length_scale_mean_value);
            testCase.verifyEqual(hyperpriors.length_scale_var, ...
                testCase.length_scale_var_value);
            
            testCase.verifyEqual(hyperpriors.output_scale_mean, ...
                testCase.output_scale_mean_value);
            testCase.verifyEqual(hyperpriors.output_scale_var, ...
                testCase.output_scale_var_value);
            
            testCase.verifyEqual(hyperpriors.p_length_scale_mean, ...
                testCase.p_length_scale_mean_value);
            testCase.verifyEqual(hyperpriors.p_length_scale_var, ...
                testCase.p_length_scale_var_value);
            
            testCase.verifyEqual(hyperpriors.p_mean, ...
                testCase.p_mean_value);
            testCase.verifyEqual(hyperpriors.p_var, ...
                testCase.p_var_value);
            
            testCase.verifyEqual(hyperpriors.alpha_mean, ...
                testCase.alpha_mean_value);
            testCase.verifyEqual(hyperpriors.alpha_var, ...
                testCase.alpha_var_value);
            
            testCase.verifyEqual(hyperpriors.lik_noise_std_mean, ...
                testCase.lik_noise_std_mean_value);
            testCase.verifyEqual(hyperpriors.lik_noise_std_var, ...
                testCase.lik_noise_std_var_value);
            
            testCase.verifyEqual(hyperpriors.mean_offset_mean, ...
                testCase.mean_offset_value);
            testCase.verifyEqual(hyperpriors.mean_offset_var, ...
                testCase.mean_offset_var_value);
            
        end
        
        function testConstructorDefaultParametersDataNoise(testCase)
            data_noise = 0.1951;
            hyperpriors = Hyperpriors(data_noise);
            
            testCase.verifyEqual(hyperpriors.lik_noise_std_mean, ...
                log(data_noise));
            testCase.verifyEqual(hyperpriors.lik_noise_std_var, ...
                testCase.lik_noise_std_var_value);
            
            testCase.verifyEqual(hyperpriors.length_scale_mean, ...
                testCase.length_scale_mean_value);
            testCase.verifyEqual(hyperpriors.length_scale_var, ...
                testCase.length_scale_var_value);
            
            testCase.verifyEqual(hyperpriors.output_scale_mean, ...
                testCase.output_scale_mean_value);
            testCase.verifyEqual(hyperpriors.output_scale_var, ...
                testCase.output_scale_var_value);
            
            testCase.verifyEqual(hyperpriors.p_length_scale_mean, ...
                testCase.p_length_scale_mean_value);
            testCase.verifyEqual(hyperpriors.p_length_scale_var, ...
                testCase.p_length_scale_var_value);
            
            testCase.verifyEqual(hyperpriors.p_mean, ...
                testCase.p_mean_value);
            testCase.verifyEqual(hyperpriors.p_var, ...
                testCase.p_var_value);
            
            testCase.verifyEqual(hyperpriors.alpha_mean, ...
                testCase.alpha_mean_value);
            testCase.verifyEqual(hyperpriors.alpha_var, ...
                testCase.alpha_var_value);
            
            testCase.verifyEqual(hyperpriors.mean_offset_mean, ...
                testCase.mean_offset_value);
            testCase.verifyEqual(hyperpriors.mean_offset_var, ...
                testCase.mean_offset_var_value);
            
        end
        
        function testGaussianPriorMethod(testCase)
            total_samples = 500000;
            tolerance = 1e-3;
            hyperpriors = Hyperpriors();
            
            hyperparameter_name = 'length_scale';
            mean_value = exp(testCase.length_scale_mean_value);
            var_value = testCase.length_scale_var_value;
            
            prior = hyperpriors.gaussian_prior(hyperparameter_name);
            samples = zeros(1,total_samples);
            for i=1:total_samples
                samples(i) = prior();
            end

            diagnostic = sprintf('Mean is %f, expected value is %f', ...
                exp(mean(samples)), mean_value);            
            mean_difference = abs(exp(mean(samples)) - mean_value);
            testCase.assertTrue(mean_difference < tolerance, diagnostic)
            
            tolerance = 1e-2;
            diagnostic = sprintf('Variance is %f, expected value is %f', ...
                var(samples), var_value);
            var_difference = abs(var(samples) - var_value);
            testCase.assertTrue(var_difference < tolerance, diagnostic)
        end
        
    end
    
end