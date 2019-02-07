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
            
        end
        
        function test_minimize(testCase)
            
        end
        
    end
    
end