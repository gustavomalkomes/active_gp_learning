classdef GpModelTest < matlab.unittest.TestCase    
    methods (Test)
        function testCreateGp(testCase)
            hyperpriors = Hyperpriors();
            covariance = Covariance.str2covariance('SE', hyperpriors);
            model = GpModel(covariance, hyperpriors);
            
            testCase.assertEqual(covariance,model.covariance)
            testCase.assertEqual(hyperpriors,model.hyperpriors)
            % TODO test everything else
        end
        
        function testPredictions(testCase)
            hyperpriors = Hyperpriors();
            covariance = Covariance.str2covariance('SE', hyperpriors);
            model = GpModel(covariance, hyperpriors);
            
            % be quiet
            model.optimization_options.display = 0;
            
            x = randn(20, 1);                 % 20 training inputs
            y = sin(3*x) + 0.1*randn(20, 1);  % 20 noisy training targets
            xs = linspace(-3, 3, 61)';        % 61 test inputs             

            testCase.verifyError(@model.predict, ...
                'AGPL:GpModel:prediction:thetaIsEmpty')
            
            model = model.train(x,y);
            ys = model.predict(x,y,xs);
            ys_post = model.predict(x,[],xs);
            
            testCase.assertTrue(sum(abs(ys - ys_post)) < 1e-6)
        end
    end
    
end