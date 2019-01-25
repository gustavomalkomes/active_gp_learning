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

            testCase.verifyError(@model.predict,'prediction_error:thetaIsEmpty')
            model = model.train(x,y);
            ys = model.predict(x,y,xs);
            
            % just make sure we have predictions
            % TODO expand this
            testCase.assertEqual( numel(xs), numel(ys) ) 
        end
    end
    
end