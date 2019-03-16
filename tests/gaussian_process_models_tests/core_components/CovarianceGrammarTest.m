classdef CovarianceGrammarTest < matlab.unittest.TestCase    
    methods (Test)
        function testCovarianceGrammarCreate(testCase)
            
            base_kernels_names = {'SE', 'RQ'};
            dimension = 1;
            
            % no arguments
            grammar = CovarianceGrammar(base_kernels_names, dimension, []);
            
            testCase.assertEqual(...
                base_kernels_names, grammar.base_kernels_names)
            
            testCase.assertTrue(...
                numel(base_kernels_names) == numel(grammar.base_kernels))

            testCase.assertTrue(dimension == grammar.dimension)
            
            testCase.assertClass(grammar.hyperpriors, 'Hyperpriors')
                        
            expected_covariance = Covariance.str2covariance('SE', []);
            testCase.assertEqual(expected_covariance.function_handle, ...
                grammar.base_kernels{1}.function_handle)

            expected_covariance = Covariance.str2covariance('RQ', []);
            testCase.assertEqual(expected_covariance.function_handle, ...
                grammar.base_kernels{2}.function_handle)

            expected_covariance = Covariance.str2covariance('SE', []);
            testCase.assertNotEqual(expected_covariance.function_handle, ...
                grammar.base_kernels{2}.function_handle)
            
        end 
        
        
        function testCovarianceGrammarExpand_one_dimension(testCase)

            root = {'SE', 'RQ', 'LIN'};
            dim = 1;
            grammar = CovarianceGrammar(root, dim, []);
            
            % expanding around SE
            se = grammar.base_kernels{1};
            expected_covariances = { ...
                '(SE+SE)', ...
                '(SE*SE)', ...
                '(SE+RQ)', ...
                '(SE*RQ)', ...
                '(SE+LIN)', ...                
                '(SE*LIN)', ...
                };
            new_covariances = grammar.expand(se);
            
            for i=1:numel(expected_covariances)
                testCase.assertEqual(expected_covariances{i}, ...
                    new_covariances{i}.name)
            end
            
            % expanding around (SE+RQ)
            se_plus_rq = new_covariances{3};
            expected_covariances = { ...
                '((SE+RQ)+SE)', ...
                '((SE+RQ)*SE)', ...
                '((SE+RQ)+RQ)', ...
                '((SE+RQ)*RQ)', ...
                '((SE+RQ)+LIN)', ...
                '((SE+RQ)*LIN)', ...
                };
            new_covariances = grammar.expand(se_plus_rq);
            
            for i=1:numel(expected_covariances)
                testCase.assertEqual(expected_covariances{i}, ...
                    new_covariances{i}.name)
            end
        end
        

        function testCovarianceGrammarMaskKernels(testCase)
            root = {'SE', 'RQ'};
            dim = 3;
            grammar = CovarianceGrammar(root, dim, []);
            expected_covariances = { ...
                'SE_1', ...
                'SE_2', ...
                'SE_3', ...
                'RQ_1', ...
                'RQ_2', ...
                'RQ_3', ...
                };
            
            for i=1:numel(expected_covariances)
                testCase.assertEqual(expected_covariances{i}, ...
                    grammar.base_kernels_names{i})
            end
            
        end
        
        function testCovarianceGrammarExpand_multi_dimensions(testCase)
            root = {'SE', 'RQ'};
            dim = 3;
            grammar = CovarianceGrammar(root, dim, []);
            
            % expanding around SE
            se_1 = grammar.base_kernels{1};
            expected_covariances = { ...
                '(SE_1+SE_1)', ...
                '(SE_1*SE_1)', ...
                '(SE_1+SE_2)', ...
                '(SE_1*SE_2)', ...
                '(SE_1+SE_3)', ...
                '(SE_1*SE_3)', ...
                '(SE_1+RQ_1)', ...
                '(SE_1*RQ_1)', ...
                '(SE_1+RQ_2)', ...
                '(SE_1*RQ_2)', ...
                '(SE_1+RQ_3)', ...
                '(SE_1*RQ_3)'                
                };
            new_covariances = grammar.expand(se_1);
            for i=1:numel(expected_covariances)
                testCase.assertEqual(expected_covariances{i}, ...
                    new_covariances{i}.name)
            end
                        
        end

        function testFull_expand(testCase)
           
            root = {'SE', 'RQ'};
            dim = 2;
            grammar = CovarianceGrammar(root, dim, []);
            
            max_number_of_models = 1000;
            level = 0;
            covariances = grammar.full_expand(level, max_number_of_models);
            expected = numel(root)*dim;
            testCase.assertEqual(expected, numel(covariances));
            
            level = 1;
            covariances = grammar.full_expand(level, max_number_of_models);
            n = numel(root);
            expected = (n*dim+1)*n*dim;
            testCase.assertEqual(expected, numel(covariances));
            
            level = 2;
            covariances = grammar.full_expand(level, max_number_of_models);
            expected = 134;
            testCase.assertEqual(expected, numel(covariances));

            level = 4;
            covariances = grammar.full_expand(level, max_number_of_models);
            expected = max_number_of_models;
            testCase.assertEqual(expected, numel(covariances));
        end
    end
    
end