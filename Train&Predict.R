### -----------------------------------------------------------
### Load packages
library(tidyverse)

### -----------------------------------------------------------
### Load data sets
cp1_transformed <- read.csv("./data/cp1_transformed.csv")
cp2_transformed <- read.csv("./data/cp2_transformed.csv")
cp3_transformed <- read.csv("./data/cp3_transformed.csv")

### -----------------------------------------------------------
### Prepare data for each question

## Checkpoint 1

# question 1
q1 <- cp1_transformed %>% 
  select(-c(q2, q3, correctness))
q1$q1 <- as.factor(q1$q1)
table(q1$q1)
# oversampling imbalanced data
q1_0s <- q1[q1$q1 == 0, ]
set.seed(579)
new_dat <- q1_0s[sample(nrow(q1_0s), 10000, replace = TRUE), ]
q1_balanced <- rbind(q1, new_dat)
table(q1_balanced$q1)
# remove used objects
rm(q1, q1_0s, new_dat)

# question 2
q2 <- cp1_transformed %>% 
  mutate(correctness  = q1) %>% 
  select(-c(q1, q3))
q2$q2 <- as.factor(q2$q2)
table(q2$q2)
# oversampling imbalanced data
q2_0s <- q2[q2$q2 == 0, ]
set.seed(579)
new_dat <- q2_0s[sample(nrow(q2_0s), 20000, replace = TRUE), ]
q2_balanced <- rbind(q2, new_dat)
table(q2_balanced$q2)
# remove used objects
rm(q2, q2_0s, new_dat)

# question 3
q3 <- cp1_transformed %>% 
  mutate(correctness  = (q1 + q2)/2) %>% 
  select(-c(q1, q2))
q3$q3 <- as.factor(q3$q3)
table(q3$q3)
# oversampling imbalanced data
q3_0s <- q3[q3$q3 == 0, ]
set.seed(579)
new_dat <- q3_0s[sample(nrow(q3_0s), 20000, replace = TRUE), ]
q3_balanced <- rbind(q3, new_dat)
table(q3_balanced$q3)
# remove used objects
rm(q3, q3_0s, new_dat)

## Checkpoint 2

# question 4
q4 <- cp2_transformed %>% 
  mutate(correctness  = cp1) %>% 
  select(-c(q1, q2, q3, q5, q6, q7, q8, q9, q10, q11, q12, q13, cp1))
q4$q4 <- as.factor(q4$q4)
table(q4$q4)
# oversampling imbalanced data
q4_0s <- q4[q4$q4 == 0, ]
set.seed(579)
new_dat <- q4_0s[sample(nrow(q4_0s), 13000, replace = TRUE), ]
q4_balanced <- rbind(q4, new_dat)
table(q4_balanced$q4)
# remove used objects
rm(q4, q4_0s, new_dat)

# question 5
q5 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + q4)/4) %>% 
  select(-c(q1, q2, q3, q4, q6, q7, q8, q9, q10, q11, q12, q13, cp1))
q5$q5 <- as.factor(q5$q5)
table(q5$q5) # original data is balanced
q5_balanced <- q5
# remove used objects
rm(q5)

# question 6
q6 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + q4 + q5)/5) %>% 
  select(-c(q1, q2, q3, q4, q5, q7, q8, q9, q10, q11, q12, q13, cp1))
q6$q6 <- as.factor(q6$q6)
table(q6$q6)
# oversampling imbalanced data
q6_0s <- q6[q6$q6 == 0, ]
set.seed(579)
new_dat <- q6_0s[sample(nrow(q6_0s), 13000, replace = TRUE), ]
q6_balanced <- rbind(q6, new_dat)
table(q6_balanced$q6)
# remove used objects
rm(q6, q6_0s, new_dat)

# question 7
q7 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + q4 + q5 + q6)/6) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q8, q9, q10, q11, q12, q13, cp1))
q7$q7 <- as.factor(q7$q7)
table(q7$q7)
# oversampling imbalanced data
q7_0s <- q7[q7$q7 == 0, ]
set.seed(579)
new_dat <- q7_0s[sample(nrow(q7_0s), 10000, replace = TRUE), ]
q7_balanced <- rbind(q7, new_dat)
table(q7_balanced$q7)
# remove used objects
rm(q7, q7_0s, new_dat)

# question 8
q8 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + q4 + q5 + q6 + q7)/7) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q9, q10, q11, q12, q13, cp1))
q8$q8 <- as.factor(q8$q8)
table(q8$q8)
# oversampling imbalanced data
q8_0s <- q8[q8$q8 == 0, ]
set.seed(579)
new_dat <- q8_0s[sample(nrow(q8_0s), 6000, replace = TRUE), ]
q8_balanced <- rbind(q8, new_dat)
table(q8_balanced$q8)
# remove used objects
rm(q8, q8_0s, new_dat)

# question 9
q9 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12 - q11 - q10 - q9)/8) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q10, q11, q12, q13, cp1))
q9$q9 <- as.factor(q9$q9)
table(q9$q9)
# oversampling imbalanced data
q9_0s <- q9[q9$q9 == 0, ]
set.seed(579)
new_dat <- q9_0s[sample(nrow(q9_0s), 10000, replace = TRUE), ]
q9_balanced <- rbind(q9, new_dat)
table(q9_balanced$q9)
# remove used objects
rm(q9, q9_0s, new_dat)

# question 10
q10 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12 - q11 - q10)/9) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q11, q12, q13, cp1))
q10$q10 <- as.factor(q10$q10)
table(q10$q10)  # already balanced
q10_balanced <- q10
# remove used objects
rm(q10)

# question 11
q11 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12 - q11)/10) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q12, q13, cp1))
q11$q11 <- as.factor(q11$q11)
table(q11$q11)
# oversampling imbalanced data
q11_0s <- q11[q11$q11 == 0, ]
set.seed(579)
new_dat <- q11_0s[sample(nrow(q11_0s), 8000, replace = TRUE), ]
q11_balanced <- rbind(q11, new_dat)
table(q11_balanced$q11)
# remove used objects
rm(q11, q11_0s, new_dat)

# question 12
q12 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13 - q12)/11) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q13, cp1))
q12$q12 <- as.factor(q12$q12)
table(q12$q12)
# oversampling imbalanced data
q12_0s <- q12[q12$q12 == 0, ]
set.seed(579)
new_dat <- q12_0s[sample(nrow(q12_0s), 15000, replace = TRUE), ]
q12_balanced <- rbind(q12, new_dat)
table(q12_balanced$q12)
# remove used objects
rm(q12, q12_0s, new_dat)

# question 13
q13 <- cp2_transformed %>% 
  mutate(correctness  = (cp1 * 3 + correctness * 10 - q13)/12) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, cp1))
q13$q13 <- as.factor(q13$q13)
table(q13$q13)
# oversampling imbalanced data
q13_1s <- q13[q13$q13 == 1, ]
set.seed(579)
new_dat <- q13_1s[sample(nrow(q13_1s), 10000, replace = TRUE), ]
q13_balanced <- rbind(q13, new_dat)
table(q13_balanced$q13)
# remove used objects
rm(q13, q13_1s, new_dat)

## Checkpoint 3

# question 14
q14 <- cp3_transformed %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10)/13) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q15, 
            q16, q17, q18, cp1, cp2))
q14$q14 <- as.factor(q14$q14)
table(q14$q14)
# oversampling imbalanced data
q14_0s <- q14[q14$q14 == 0, ]
set.seed(579)
new_dat <- q14_0s[sample(nrow(q14_0s), 10000, replace = TRUE), ]
q14_balanced <- rbind(q14, new_dat)
table(q14_balanced$q14)
# remove used objects
rm(q14, q14_0s, new_dat)

# question 15
q15 <- cp3_transformed %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + q14)/14) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q16, q17, q18, cp1, cp2))
q15$q15 <- as.factor(q15$q15)
table(q15$q15) # already balanced
q15_balanced <- q15
# remove used objects
rm(q15)

# question 16
q16 <- cp3_transformed %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + correctness * 5 - q18 - q17 - q16)/15) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q15, q17, q18, cp1, cp2))
q16$q16 <- as.factor(q16$q16)
table(q16$q16)
# oversampling imbalanced data
q16_0s <- q16[q16$q16 == 0, ]
set.seed(579)
new_dat <- q16_0s[sample(nrow(q16_0s), 10000, replace = TRUE), ]
q16_balanced <- rbind(q16, new_dat)
table(q16_balanced$q16)
# remove used objects
rm(q16, q16_0s, new_dat)

# question 17
q17 <- cp3_transformed %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + correctness * 5 - q18 - q17)/16) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q15, q16, q18, cp1, cp2))
q17$q17 <- as.factor(q17$q17)
table(q17$q17)
# oversampling imbalanced data
q17_0s <- q17[q17$q17 == 0, ]
set.seed(579)
new_dat <- q17_0s[sample(nrow(q17_0s), 10000, replace = TRUE), ]
q17_balanced <- rbind(q17, new_dat)
table(q17_balanced$q17)
# remove used objects
rm(q17, q17_0s, new_dat)

# question 18
q18 <- cp3_transformed %>% 
  mutate(correctness  = (cp1 * 3 + cp2 * 10 + correctness * 5 - q18)/17) %>% 
  select(-c(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, 
            q15, q16, q17, cp1, cp2))
q18$q18 <- as.factor(q18$q18)
table(q18$q18)
# oversampling imbalanced data
q18_0s <- q18[q18$q18 == 0, ]
set.seed(579)
new_dat <- q18_0s[sample(nrow(q18_0s), 20000, replace = TRUE), ]
q18_balanced <- rbind(q18, new_dat)
table(q18_balanced$q18)
# remove used objects
rm(q18, q18_0s, new_dat)






















