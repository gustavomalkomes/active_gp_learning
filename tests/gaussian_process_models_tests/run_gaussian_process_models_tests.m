function test_report = run_gaussian_process_models_tests(run_all)

% should we run all tests? or just the quick tests?
% default: just quick tests
if ~exist('run_all','var')
    run_all = false;
end
 
% define tests
quick_test = true;
testCases_to_run = { ...
    {'CovarianceTest', ~quick_test}, ...
    {'CovarianceNumericalTest', ~quick_test}, ...
    {'CovarianceGrammarTest', quick_test}, ...
    {'GpModelTest', ~quick_test}, ...
    {'HyperpriorsTest', quick_test} ...
    };


if run_all
    fprintf('Running all tests\n');
else
    fprintf('Running quick tests\n');
end

test_report = cell(numel(testCases_to_run),1);

for i = 1:numel(testCases_to_run)
    test_to_run = testCases_to_run{i}{1};
    is_test_quick = testCases_to_run{i}{2};
    
    if run_all || is_test_quick
        eval(sprintf('test_report{i} = run(%s);', test_to_run))
    end
end

end