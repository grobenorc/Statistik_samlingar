# -------------------------------------
# Script: 

# Author: Claes

# Ämne: 

# Beskrivning:
# Script som gör plottar av de vanliga fördelningar:
  # - Binomialfördelning
  # - Normalfördelninge
  # - F-fördelningen
  # - T-fördelningen
  # - Chi2-fördelninge


# License: MIT

# Encoding: UTF-8-SV


# Paket:

# Paket andra


# Working directory


# @param
#

# ====================================================================
# @return
#


# -------------------------------------

pdf_binom <- function(n, p, obs = NULL) {
  x <- seq(0, n)
  y <- dbinom(x, size = n, prob = p)
  cum_prob <- pbinom(x, size = n, prob = p)
  data <- data.frame(x, y, cum_prob)
  
  plot <- ggplot2::ggplot(data = data, aes(x,y)) +
    geom_point(color = "steelblue") +
    geom_line(color = "steelblue") + 
    labs(x = "Number of successes (x)", y = "Probability", 
         title = "Sannolikhetsfunktionen för Binomial-fördelningen") +
    theme_bw() + 
    annotate("text", x = n, y =max(y), 
             label = paste("E(X): ", round(n*p, 2)), 
             hjust = 1, size = 4)
  
  if (!is.null(obs)) {
    obs_prob <- dbinom(obs, size = n, prob = p)
    obs_cum_prob <- pbinom(obs, size = n, prob = p)
    plot <- plot +
      geom_point(aes(x = obs, y = obs_prob), color = "red", size = 5) +
      annotate("text", x = n, y = max(y), 
               label = paste("Prob: ", round(obs_prob, 4), "\nCum. Prob: ", round(obs_cum_prob, 4)), 
               hjust = 1, vjust = 1.5, size = 4, color = "red")
  }
  
  plot
}




pdf_norm <- function(mu, sigma, obs = NULL, xlim = c(mu - 4*sigma, mu + 4*sigma), n = 100) {
  x <- seq(xlim[1], xlim[2], length.out = n)
  y <- dnorm(x, mean = mu, sd = sigma)
  cum_prob <- pnorm(x, mean = mu, sd = sigma)
  data <- data.frame(x, y, cum_prob)
  
  plot <- ggplot2::ggplot(data = data, aes(x,y)) +
    geom_line(color = "steelblue") + 
    labs(x = "x", y = "Probability", 
         title = "Probability Density Function (PDF) av Normalfördelningen") +
    theme_bw() + 
    annotate("text", x = mu + 3.5, y =max(y), 
             label = paste("E(X): ", round(mu, 2)), 
             hjust = 1, size = 4)
  
  if (!is.null(obs)) {
    obs_prob <- dnorm(obs, mean = mu, sd = sigma)
    obs_cum_prob <- pnorm(obs, mean = mu, sd = sigma)
    plot <- plot +
      geom_point(aes(x = obs, y = obs_prob), color = "red", size = 5) +
      annotate("text", x = mu+3.5, y = max(y), 
               label = paste("Prob: ", round(obs_prob, 4), "\nCum. Prob: ", round(obs_cum_prob, 4)), 
               hjust = 1, vjust = 1.5, size = 4, color = "red")
  }
  
  plot
}




pdf_t <- function(df, obs = NULL, xlim = c(-4, 4), n = 100) {
  x <- seq(xlim[1], xlim[2], length.out = n)
  y <- dt(x, df = df)
  cum_prob <- pt(x, df = df)
  data <- data.frame(x, y, cum_prob)
  
  plot <- ggplot2::ggplot(data = data, aes(x, y)) +
    geom_line(color = "steelblue") + 
    labs(x = "x", y = "Probability", 
         title = "Probability Density Function (PDF) av t-fördelningen") +
    theme_bw() + 
    annotate("text", x = 3.5, y = max(y), 
             label = paste("E(X): 0"), 
             hjust = 1, size = 4)
  
  if (!is.null(obs)) {
    obs_prob <- dt(obs, df = df)
    obs_cum_prob <- pt(obs, df = df)
    plot <- plot +
      geom_point(aes(x = obs, y = obs_prob), color = "red", size = 5) +
      annotate("text", x = 3.5, y = max(y), 
               label = paste("Prob: ", round(obs_prob, 4), "\nCum. Prob: ", round(obs_cum_prob, 4)), 
               hjust = 1, vjust = 1.5, size = 4, color = "red")
  }
  
  plot
}





pdf_f <- function(df1, df2, obs = NULL) {
  xlim <- c(0, df1+df2)
  x <- seq(xlim[1], xlim[2], length.out = 500)
  y <- df(x, df1 = df1, df2 = df2)
  cum_prob <- pf(x, df1 = df1, df2 = df2)
  data <- data.frame(x, y, cum_prob)
  
  plot <- ggplot2::ggplot(data = data, aes(x, y)) +
    geom_line(color = "steelblue") + 
    labs(x = "x", y = "Probability", 
         title = "Probability Density Function (PDF) av F-Fördelningen") +
    theme_bw() + 
    annotate("text", x = (df1+df2)-0.5, y = max(y), 
             label = paste("E(X): ", round(df2/(df2 - 2), 2)), 
             hjust = 1, size = 4)
  
  if (!is.null(obs)) {
    obs_prob <- df(obs, df1 = df1, df2 = df2)
    obs_cum_prob <- pf(obs, df1 = df1, df2 = df2)
    plot <- plot +
      geom_point(aes(x = obs, y = obs_prob), color = "red", size = 5) +
      annotate("text", x = (df1+df2)-0.5, y = max(y), 
               label = paste("Prob: ", round(obs_prob, 4), "\nCum. Prob: ", round(obs_cum_prob, 4)), 
               hjust = 1, vjust = 1.5, size = 4, color = "red")
  }
  
  plot
}




pdf_chi2 <- function(df, obs = NULL) {
  
  prob_cutoff <- 0.0001
  
  xlim <- c(ifelse(df<30, 0, qchisq(prob_cutoff, df = df)), qchisq(1-prob_cutoff, df = df))
  x <- seq(xlim[1], xlim[2], length.out = 300)
  y <- dchisq(x, df = df)
  cum_prob <- pchisq(x, df = df)
  data <- data.frame(x, y, cum_prob)
  
  plot <- ggplot2::ggplot(data = data, aes(x, y)) +
    geom_line(color = "steelblue") + 
    labs(x = "x", y = "Probability", 
         title = "Probability Density Function (PDF) av Chi-square Fördelningen") +
    theme_bw() + 
    annotate("text", x = qchisq(1-prob_cutoff, df = df), y = max(y), 
             label = paste("E(X): ", round(df, 2)), 
             hjust = 1, size = 4)
  
  if (!is.null(obs)) {
    obs_prob <- dchisq(obs, df = df)
    obs_cum_prob <- pchisq(obs, df = df)
    plot <- plot +
      geom_point(aes(x = obs, y = obs_prob), color = "red", size = 5) +
      annotate("text", x = qchisq(1-prob_cutoff, df = df), y = max(y), 
               label = paste("Prob: ", round(obs_prob, 4), "\nCum. Prob: ", round(obs_cum_prob, 4)), 
               hjust = 1, vjust = 1.5, size = 4, color = "red")
  }
  
  plot
}


