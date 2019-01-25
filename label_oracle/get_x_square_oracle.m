function label_oracle = get_x_square_oracle()

    label_oracle = @(problem, x_train, y_train, x_star) x_star.^2;

end