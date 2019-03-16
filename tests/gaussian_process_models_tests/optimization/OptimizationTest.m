classdef OptimizationTest < matlab.unittest.TestCase
    
    properties
        x_data
        y_data
        x_test
    end
    
    methods(TestMethodSetup)
        function create_data(testCase)
            x = randn(20, 1);
            testCase.x_data = x;
            testCase.y_data = sin(3*x) + 0.1*randn(20, 1);
            testCase.x_test = linspace(-3, 3, 61)';
        end
    end
    
    methods (Test)
        function test_minFunc(testCase)
            x = testCase.x_data;
            y = testCase.y_data;
            z = testCase.x_test;
            
            % model
            hyperpriors = Hyperpriors();
            covariance = Covariance.str2covariance('SE', hyperpriors);
            model = GpModel(covariance, hyperpriors);
                        
            hyp = model.prior();
            [new_hyp, nlZ] = minimize_minFunc(model, x, y, ...
                'initial_hyperparameters', hyp, ...
                'num_restarts', 0);
            testCase.assertEqual(hyp, new_hyp)
            
            [new_hyp, new_nlZ] = minimize_minFunc(model, x, y, ...
                'initial_hyperparameters', hyp, ...
                'num_restarts', 3);
            testCase.assertNotEqual(new_hyp, hyp)
            testCase.assertLessThan(new_nlZ, nlZ)
        end
        
    end
    
end