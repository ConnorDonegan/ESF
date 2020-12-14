library(rstan)
library(MASS)
library(spdep)
library(sf)
library(ggplot2)
library(bayesplot)
## you can load some helpful functions from the geostan R package;
## The package will be released to CRAN sometime soon.
## github.com/ConnorDonegan/geostan
source("https://raw.githubusercontent.com/ConnorDonegan/geostan/master/R/convenience-functions.R")

## compile the Stan model 
esf_model <- stan_model("ESF.stan")

## you can view the model using print
print(esf_model)

## load the Ohio election data from Donegan et al. 2020
## ohio is of class `sf` (simple features)
load("ohio.rda")

## make binary spatial connectivity matrix
 ## use C for creating eigenvectors
C <- shape2mat(ohio, style = "B")

## create eigenvectors
EV <- make_EV(C)

## make row-standardized spatial weights matrix
  ## use W for creating spatially lagged covariates, and measuring spatial autocorrelation using aple() or mc()
W <- shape2mat(ohio, style = "W")

## visualize/measure spatial autocorrelation
moran_plot(ohio$gop_growth, W)
aple(ohio$gop_growth, W)

## prepare data for Stan
# no intercept in the design matrix
form <- gop_growth ~ 0 + historic_gop + log(pop_density) + white_nonhispanic + college_educated + unemployment
X <- model.matrix(form, ohio)

## add spatially lagged covariates 
WX <- W %*% X
colnames(WX) <- paste0("W.", colnames(WX))
X <- cbind(WX, X)

## scale it (priors are always relative to scale of data)
X <- scale(X)

## dimensions of data, outcome variable
dx <- ncol(X)
n <- nrow(ohio)
y <- ohio$gop_growth

## make prior for all the coefficients
## they are getting the same Student t prior
## each needs: degrees of freedom, location, scale
beta_priors <- data.frame(
    df = rep(15, dx),
    location = rep(0, dx),
    scale = rep(5, dx)
    )

  ## the data
stan_data <- list(
    n = n,
    dev = ncol(EV),
    dx = ncol(X),
    y = y,
    x = X,
    EV = EV,
    ## horseshoe hyper-priors
    scale_global = .4,
    slab_df = 15,
    slab_scale = 15,
    ## Student t prior for the intercept: student_t(df, location, scale)
    alpha_prior = c(15, 0, 10),
    ## Student t priors for coefficients (beta)
    beta_prior = t(beta_priors),
    ## Studnt t prior for the scale of the outcome
    sigma_prior = c(10, 0, 5)
    )

## fit the model
fit <- sampling(esf_model,
                data = stan_data,
                chains = 4,
                cores = 4,
                iter = 2e3,
                ## you can limit which parameters are returned, optional
                pars = c("Intercept", "beta", "nu", "sigma", "esf", "fitted", "residual", "yrep")
                )

## check Stan diagnostics
rstan::stan_ess(fit)
rstan::stan_rhat(fit)

## posterior predictive check
yrep <- as.matrix(fit, pars = "yrep")
bayesplot::ppc_dens_overlay(y, yrep[1:75,])

## residual spatial autocorrelation
res <- as.matrix(fit, pars = "residual")
res_mean <- apply(res, 2, mean)
moran_plot(res_mean, W)

## map the spatial filter
sf <- as.matrix(fit, pars = "esf")
sf_mean <- apply(sf, 2, mean)
ggplot(ohio) +
    geom_sf(aes(fill = sf_mean)) +
    scale_fill_gradient2()

## print results
print(fit, pars = c("Intercept", "beta", "sigma", "nu"))
