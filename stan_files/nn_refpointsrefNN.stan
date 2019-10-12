// Fixed NN model:

data 
{
    int<lower=0> Nobs;
    int<lower=0> in_dim;
    int<lower=0> out_dim;
    int<lower=0> hidden_dim_1;
    int<lower=0> n_ref;
    int<lower=0> n_ref_w1;
    int<lower=0> n_ref_b1;
    int<lower=0> n_ref_w2;
    
    matrix[in_dim, n_ref_w1] weights_1_ref;
    row_vector[n_ref_b1] bias_1_ref;
    matrix[n_ref_w2, out_dim] weights_2_ref;
    
    matrix[Nobs-n_ref,out_dim] y;
    matrix[n_ref,out_dim] y_ref;
    matrix[n_ref,in_dim] X_ref;
}

parameters 
{
    matrix[Nobs-n_ref,in_dim] X;
    real<lower=0> error_sigma2;
    real<lower=0> prior_sigma2;
    matrix[in_dim, hidden_dim_1-n_ref_w1] w_1;
    row_vector[hidden_dim_1-n_ref_b1] b_1;
    matrix[hidden_dim_1-n_ref_w2, out_dim] w_2;
    row_vector[out_dim] bias_2;
}

transformed parameters
{
    matrix[in_dim, hidden_dim_1] weights_1;
    row_vector[hidden_dim_1] bias_1;
    matrix[hidden_dim_1, out_dim] weights_2;
    
    weights_1[:,1:n_ref_w1] = weights_1_ref;
    weights_1[:,n_ref_w1+1:hidden_dim_1] = w_1;
    bias_1[1:n_ref_b1] = bias_1_ref;
    bias_1[n_ref_b1+1:hidden_dim_1] = b_1;
    weights_2[1:n_ref_w2,:] = weights_2_ref;
    weights_2[n_ref_w2+1:hidden_dim_1,:] = w_2;
}



model 
{
    // Priors:
    error_sigma2 ~ cauchy(0,5);
    prior_sigma2 ~ gamma(5,5);
    b_1 ~ normal(0,prior_sigma2);
    bias_2 ~ normal(0,prior_sigma2);
    
    for (i in 1:in_dim) 
    {
        w_1[i,:] ~ normal(0,prior_sigma2);
    }
    for (i in 1:out_dim)
    {
        w_2[:,i] ~ normal(0,prior_sigma2);
    }
    
    for (n in 1:(Nobs-n_ref))
    {
        X[n,:] ~ normal(0,1);
    }

    // Likelihood:
    for (n in 1:(Nobs-n_ref))
    {
       y[n] ~ normal(tanh(X[n,:]*weights_1 + bias_1)*weights_2 + bias_2, error_sigma2);
    }
    for (n in 1:n_ref)
    {
        y_ref[n] ~ normal(tanh(X_ref[n,:]*weights_1 + bias_1)*weights_2 + bias_2, error_sigma2);
    }
}

