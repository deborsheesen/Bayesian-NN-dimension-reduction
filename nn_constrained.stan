// NN wth some reference points tied down:

functions 
{
    vector leaky_relu(vector x) 
    {
        vector[dims(x)[1]] lrelu;
        for (i in 1:dims(x)[1])
        {
            if(x[i]>0) 
            {
                lrelu[i] = x[i];
            }
            else 
            {
                lrelu[i] = 0.01*x[i];
            }
        }
        return lrelu;
    }
}

data 
{
    int<lower=0> Nobs;
    int<lower=0> in_dim;
    int<lower=0> out_dim;
    int<lower=0> hidden_dim;
    matrix[out_dim,Nobs] y;
}

parameters 
{
    unit_vector[in_dim] X[Nobs];
    
    //vector[in_dim] X[Nobs];
    
    matrix[in_dim, hidden_dim] weights_1;
    
    //unit_vector[hidden_dim] weights_1[in_dim];
    unit_vector[out_dim] weights_2[hidden_dim];
    
    vector[hidden_dim] bias_1;
    vector[out_dim] bias_2;
    real<lower=0> error_sigma2;
    real<lower=0> prior_sigma2;
}
transformed parameters
{   
    matrix[in_dim, Nobs] X_tr; 
    matrix[out_dim, hidden_dim] weights_2_tr;
    for (i in 1:Nobs) 
    {
        X_tr[:,i] = X[i];
    }
    for (i in 1:hidden_dim)
    {
        weights_2_tr[:,i] = weights_2[i];
    }
}
model 
{  
    // Latent distribution:
    for (i in 1:Nobs)
    {
        X[i] ~ normal(0,1);
    }    
    
    // Priors:
    error_sigma2 ~ cauchy(0,5);
    prior_sigma2 ~ cauchy(0,5);
    for (i in 1:in_dim)
    {
        weights_1[i] ~ normal(0,prior_sigma2);;
    }
    for (i in 1:hidden_dim)
    {
        weights_2[i] ~ normal(0,prior_sigma2);
    }
    bias_1 ~ normal(0,prior_sigma2);
    bias_2 ~ normal(0,prior_sigma2);

    // Likelihood:
    for (n in 1:Nobs)
    {
       y[:,n] ~ normal(weights_2_tr*leaky_relu(weights_1'*X_tr[:,n] + bias_1) + bias_2, error_sigma2);
    }

}

