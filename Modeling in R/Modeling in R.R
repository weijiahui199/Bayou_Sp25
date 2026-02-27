#D2K | Bayou City Waterkeeper
#Initial MLR and Poisson Models in R


#clear workspace
dev.off()
cat("\014")
rm(list=ls())
set.seed(18552)

#set working directory
setwd("Bayou_Sp25/Modeling/Modeling in R")

#import required packages
library(tidyr)
library(dplyr)
library(lfe)
library(car)
library(ggplot2)
library(lme4)


#import data
rainfall <- read.csv("../Data/Rainfall Data by Zipcode/monthly_rainfall_by_zipcode.csv")
monthly_private_events <- read.csv("../Data/Event Count Data by Zipcode/monthly_private_by_zipcode.csv")
monthly_public_events <- read.csv("../Data/Event Count Data by Zipcode/monthly_public_by_zipcode.csv")


#pivot to long data
rainfall_long <- rainfall %>%
  pivot_longer(cols = -Zipcode, names_to = "month", values_to = "rainfall")

private_long <- monthly_private_events %>%
  pivot_longer(cols = -Zipcode, names_to = "month", values_to = "private_events")

public_long <- monthly_public_events %>%
  pivot_longer(cols = -Zipcode, names_to = "month", values_to = "public_events")

#merge events and rainfall
private_events_rainfall <- merge(private_long, rainfall_long, by = c("Zipcode", "month"))
public_events_rainfall <- merge(public_long, rainfall_long, by = c("Zipcode", "month"))


#clean zip code to a factor
private_events_rainfall$Zipcode <- factor(private_events_rainfall$Zipcode)
public_events_rainfall$Zipcode <- factor(public_events_rainfall$Zipcode)

#clean month into an integer
private_events_rainfall$year_part <- as.numeric(sub('X(\\d{4})\\.(\\d{2})', '\\1', private_events_rainfall$month))

private_events_rainfall$month_part <- as.numeric(sub('X\\d{4}\\.(\\d{2})', '\\1', private_events_rainfall$month))

private_events_rainfall$month_cleaned <- (private_events_rainfall$year_part - 2022) * 12 + private_events_rainfall$month_part

#same for public events
public_events_rainfall$year_part <- as.numeric(sub('X(\\d{4})\\.(\\d{2})', '\\1', public_events_rainfall$month))

public_events_rainfall$month_part <- as.numeric(sub('X\\d{4}\\.(\\d{2})', '\\1', public_events_rainfall$month))

public_events_rainfall$month_cleaned <- (public_events_rainfall$year_part - 2022) * 12 + public_events_rainfall$month_part







#EDA ggplots
ggplot(private_events_rainfall, aes(x = rainfall, y = private_events)) +
  geom_point(alpha = 0.5) +
  labs(title = "Rainfall and Private Overflows", x = "Rainfall", y = "Private Overflow Count")

ggplot(public_events_rainfall, aes(x = rainfall, y = public_events)) +
  geom_point(alpha = 0.5) +
  labs(title = "Rainfall and Public Overflows", x = "Rainfall", y = "Public Overflow Count")

ggplot(private_events_rainfall, aes(x = private_events)) +
  geom_histogram() +
  labs(title = "Private Overflows", x = "Overflowcount", y = "Nuber of Zip Codes")

ggplot(public_events_rainfall, aes(x = public_events)) +
  geom_histogram() +
  labs(title = "Public Overflows", x = "Overflow count", y = "Number of Zip Codes")










#MLR modeling

#model equation: count of events by zip code by month = rainfall + time trend + zip code fixed effect + error

#using lfe package for fixed effects modeling
mlr <- felm(private_events ~ rainfall + (month_cleaned * factor(Zipcode)) + factor(Zipcode), data = private_events_rainfall)
summary(mlr)

#trend effect is month * factor Zipcode, zip code fixed effect is factor Zipcode using the lm package
#try this for weekly too


#all month effects significant, rainfall not

#adding quadratic term to gauge if extreme rainfall impacts events 
mlr2 <- felm(private_events ~ rainfall + I(rainfall^2) + month_cleaned | Zipcode | 0 | Zipcode, data = private_events_rainfall)
summary(mlr2)

#now both rainfall and the quadratic term are significant at the 0.1 level
#strangely rainfall has a negative effect on events while the quadratic term has a positive effect

#testing for a time lag
lagged_private_events <- private_events_rainfall %>%
  arrange(Zipcode, month_cleaned) %>%
  group_by(Zipcode) %>%
  mutate(rainfall_lag = lag(rainfall, 1))

mlr3 <- felm(private_events ~ rainfall + rainfall_lag + month_cleaned | Zipcode | 0 | Zipcode, data = lagged_private_events)
summary(mlr3)

#rainfall lagged 1 month significant at the 0.1 level with a negative effect 

#removing time trend
mlr_simple <- felm(private_events ~ rainfall | Zipcode | 0 | Zipcode, data = private_events_rainfall)
summary(mlr_simple)

#rainfall alone has a slightly significant negative effect-- every monthly inch of rainfall reduces private overflows by 0.02677 on average

#trying the original model on the public events
mlr_public <- lm(public_events ~ rainfall + month_cleaned + factor(Zipcode), data = public_events_rainfall)
summary(mlr_public)

#rainfall has a highly significant positive effect on public overflows









#Poisson modeling

poisson <- glm(private_events ~ rainfall + factor(Zipcode), data = private_events_rainfall, family = poisson)
summary(poisson)

#certain zip codes are significant, and rainfall with negative effect

#revised with zip code random effects
poisson2 <- glmer(private_events ~ rainfall + (1|Zipcode), data = private_events_rainfall, family = poisson(link = "log"))
summary(poisson2)

#similar, rainfall significant and negative

poisson3 <- glm(public_events ~ rainfall + factor(Zipcode), data = public_events_rainfall, family = poisson(link = "log"))
summary(poisson3)

#again, in the public data, the rainfall term is significant and positive impact on overflow events









#individual zip code models

coefs <- data.frame(Zipcode = character(), rainfall_coef = numeric(), p_value = numeric(), stringsAsFactors = FALSE)

#loop through each zip code
private_events_rainfall %>%
  distinct(Zipcode) %>%
  pull(Zipcode) %>%
  #apply function to fit simple model and extract coef to coefs df
  lapply(function(zip) {
    private_zip <- private_events_rainfall %>% filter(Zipcode == zip)
    
    model <- lm(private_events ~ rainfall, data = private_zip)
    
    coef_est <- coef(model)["rainfall"]
    p_value <- summary(model)$coefficients["rainfall", "Pr(>|t|)"]
    
    coefs <<- rbind(coefs, data.frame(Zipcode = zip, rainfall_coef = coef_est, p_value = p_value))
  })

coefs$p_value_significant <- ifelse(coefs$p_value < 0.05, "Significant", "Not Significant")

#plot coefs
ggplot(coefs, aes(x = Zipcode, y = rainfall_coef, fill = p_value_significant)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Significant" = "red", "Not Significant" = "gray"))+
  theme_minimal() +
  labs(title = "Rainfall Coefficient Estimates for Private Overflow Events by Zip Code",
       x = "Zip Code",
       y = "Coefficient Estimate (Rainfall)",
       fill = "Rainfall P-value") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  geom_text(data = subset(coefs, p_value_significant == "Significant"), 
            aes(label = Zipcode), 
            vjust = -0.5, size = 3, color = "black")




#same individual analysis for the public data
public_coefs <- data.frame(Zipcode = character(), rainfall_coef = numeric(), p_value = numeric(), stringsAsFactors = FALSE)

#loop through each zip code
public_events_rainfall %>%
  distinct(Zipcode) %>%
  pull(Zipcode) %>%
  #apply function to fit simple model and extract coef to coefs df
  lapply(function(zip) {
    public_zip <- public_events_rainfall %>% filter(Zipcode == zip)
    
    model <- lm(public_events ~ rainfall, data = public_zip)
    
    coef_est <- coef(model)["rainfall"]
    p_value <- summary(model)$coefficients["rainfall", "Pr(>|t|)"]
    
    public_coefs <<- rbind(public_coefs, data.frame(Zipcode = zip, rainfall_coef = coef_est, p_value = p_value))
  })

public_coefs$p_value_significant <- ifelse(public_coefs$p_value < 0.05, "Significant", "Not Significant")

#plot coefs
ggplot(public_coefs, aes(x = Zipcode, y = rainfall_coef, fill = p_value_significant)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Significant" = "red", "Not Significant" = "gray"))+
  theme_minimal() +
  labs(title = "Rainfall Coefficient Estimates for Public Overflow Events by Zip Code",
       x = "Zip Code",
       y = "Coefficient Estimate (Rainfall)",
       fill = "Rainfall P-value") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  geom_text(data = subset(public_coefs, p_value_significant == "Significant"), 
            aes(label = Zipcode), 
            vjust = -0.5, size = 3, color = "black")

#plot only zip codes with high coefficients
coefs_filtered <- coefs %>%
  filter(abs(rainfall_coef) > 0.125)

ggplot(coefs_filtered, aes(x = Zipcode, y = rainfall_coef, fill = p_value_significant)) +
  geom_bar(stat = "identity") +
  scale_fill_manual(values = c("Significant" = "red", "Not Significant" = "gray"))+
  theme_minimal() +
  labs(title = "Rainfall Coefficient Estimates for Private Overflow Events by Zip Code",
       x = "Zip Code",
       y = "Coefficient Estimate (Rainfall)",
       fill = "Rainfall P-value") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  geom_text(data = subset(coefs, p_value_significant == "Significant"), 
            aes(label = Zipcode), 
            vjust = -0.5, size = 4.5, color = "black")

#need to plot this with income and race but 77026 (strongest negative effect) is in Kashmere Gardens, a very low income neighborhood