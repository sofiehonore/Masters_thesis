#!/usr/bin/Rscript

# Load packages
library(tidyverse)
library(ggplot2)
library(dplyr)
library(zoo)

feature_type <- "continuous"
#feature_type <- "binary"

# Define test set to run
run_test_set = 4

## Read in values for plot (from partial dependence in python) ##
ICE_df <- read.csv('/home/projects/ssi_10004/people/sofhor/hard_mod_mat_results/predictions_PDP_mean_DBP_27_30.csv', header = TRUE)

# Create tibble and streamline feature values
ICE_tibble <- tibble(ICE_df)

# Find average for each feature value for each test set
ICE_tibble <- ICE_tibble %>%
    group_by(Test_set, Feature_value) %>%
    mutate(Average = mean(Prediction),
           Centered_average = mean(Centered_prediction)) %>%
    ungroup() %>%
    arrange(Test_set, Feature_value)

# Define rolling average
average_tibble <- ICE_tibble %>%
    distinct(Test_set, Feature_value, Centered_average) %>%
    mutate(Rolling_centered_average = rollapply(Centered_average, 2, mean, align = "right", fill = NA))

ICE_tibble_with_rolling_average <- ICE_tibble %>%
    left_join(average_tibble, join_by(Test_set, Feature_value)) %>%
    filter(Test_set == run_test_set) 

# Define total number of rows in the tibble with all observations - to be able to cut the first ones out where the rolling average is faulty
total_number_of_rows_ICE <- nrow(ICE_tibble_with_rolling_average)

# Create plot for continuous feature
if(feature_type == "continuous"){
    # Define percentiles to plot on x-axis
    percentile_2_5 <- as.double(unname(quantile(ICE_tibble %>% filter(Test_set == run_test_set) %>% select(Feature_value) %>% pull(), probs = c(0.025))))
    percentile_25 <- as.double(unname(quantile(ICE_tibble %>% filter(Test_set == run_test_set) %>% select(Feature_value) %>% pull(), probs = c(0.25))))
    percentile_50 <- as.double(unname(quantile(ICE_tibble %>% filter(Test_set == run_test_set) %>% select(Feature_value) %>% pull(), probs = c(0.50))))
    percentile_75 <- as.double(unname(quantile(ICE_tibble %>% filter(Test_set == run_test_set) %>% select(Feature_value) %>% pull(), probs = c(0.75))))
    percentile_97_5 <- as.double(unname(quantile(ICE_tibble %>% filter(Test_set == run_test_set) %>% select(Feature_value) %>% pull(), probs = c(0.0975))))

    
    # Plot values 
    ICE_plot <- ICE_tibble_with_rolling_average %>% #ICE_tibble %>%
        ggplot() +
        geom_line(aes(x = Feature_value, y = Centered_prediction, group = Observation), alpha = 0.1) +
        geom_line(data = ICE_tibble_with_rolling_average[500:total_number_of_rows_ICE,], mapping = aes(x = Feature_value, y = Rolling_centered_average), alpha = 1, colour = "red") +
        xlab("Mean DBP (mmHg)") +
        ylab("Partial Dependence, Test Set 4") +
        geom_segment(aes(x = percentile_2_5, y = -0.22, xend = percentile_2_5, yend = -0.215)) +
        geom_segment(aes(x = percentile_25, y = -0.22, xend = percentile_25, yend = -0.215)) +
        geom_segment(aes(x = percentile_50, y = -0.22, xend = percentile_50, yend = -0.215)) +
        geom_segment(aes(x = percentile_75, y = -0.22, xend = percentile_75, yend = -0.215)) +
        geom_segment(aes(x = percentile_97_5, y = -0.22, xend = percentile_97_5, yend = -0.215)) +
        xlim(50,105)
        
    # Save plot
    ggsave('ICE_plot_mean_DBP_27_30_test4.png',
        plot = ICE_plot,
        device = "png",
        path = '/home/projects/ssi_10004/people/sofhor/PDP_ICE_plots')}

# Create box plot for binary feature
if(feature_type == "binary"){
        ICE_plot <- ICE_tibble %>%
        filter(Test_set == run_test_set) %>%
        ggplot(aes(x = as.character(Feature_value), y = Prediction)) +
        geom_boxplot() +
        xlab("Vitamins") +
        ylab("Partial Dependence") #+
        #scale_x_discrete(labels = c("0", "1"))
    
    # Save plot
    ggsave('ICE_plot_binary.png',
        plot = ICE_plot,
        device = "png",
        path = '/home/projects/ssi_10004/people/sofhor/PDP_ICE_plots')
    }
    
    

    






