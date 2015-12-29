## generatezscores_ann.R

## annotated version of generatezscores.R, which does precisely what it sounds like it should do
## it reads in ANTHA.csv, combines length and height, removes subjects who do not have geniq measurements,
## removes subjects who are missing weight measurements, fixes the apgar score, bins categorical variables
## impute lenehi at agedays = 123 using stochastic regression imputation, calculate z-scores using who scripts

## packages:
##  dummies
##  plyr
##  stringr

## filepath: C:\Users\Alexander\Documents\rpi\research\2015_2fa\gates_r
## filepath: C:\Program Files (x86)\Graphviz2.38\bin

# @BEGIN generatezscores
# @OUT oldyung.csv @URI file:oldyung.csv

# @BEGIN load_data
# @IN ANTHA_data @URI file:ANTHA.csv
# @OUT ANTHA @as num_0
## our data
ANTHA <- read.csv("~/rpi/research/2015summer/datathingy/gates/ANTHA.csv", stringsAsFactors = FALSE, header = TRUE)
# @END load_data


# @BEGIN prune_features
# @IN ANTHA @as num_0
# @OUT ANTHA @as num_1
ANTHA$X <- NULL
ANTHA$bmi <- NULL
ANTHA$waz <- NULL
ANTHA$haz <- NULL
ANTHA$whz <- NULL
ANTHA$baz <- NULL
ANTHA$sex <- NULL
ANTHA$sexn <- ANTHA$sexn - 1
ANTHA$feeding <- NULL
ANTHA$mrace <- NULL
ANTHA$mracen <- (ANTHA$mracen-3)/2
ANTHA$mmarit <- NULL
ANTHA$ses <- NULL
ANTHA$diabp <- NULL
ANTHA$sysbp <- NULL
# @END prune_features

# @BEGIN combine_lencm_and_htcm_into_lenhei
# @IN ANTHA @as num_1
# @OUT ANTHA @as num_2
## combine lencm and htcm into lenhei
ANTHA$lencm <- pmax(ANTHA$htcm, ANTHA$lencm, na.rm = TRUE)
ANTHA <- rename(ANTHA, c("lencm" = "lenhei"))
ANTHA$htcm <- NULL
# @END combine_lencm_and_htcm_into_lenhei

# @BEGIN remove_subjects_who_do_not_have_geniq_measures_then_get_rid_of_geniq
# @IN ANTHA @as num_2
# @OUT ANTHA @as num_3
## remove subjects who do not have geniq measures then get rid of geniq
nullpats <- ANTHA[is.na(ANTHA$geniq) & ANTHA$agedays == 2558,]$subjid
ANTHA <- ANTHA[!(ANTHA$subjid %in% nullpats),]
rm(nullpats)
ANTHA$geniq <- NULL
# @END remove_subjects_who_do_not_have_geniq_measures_then_get_rid_of_geniq

# @BEGIN remove_subjects_who_are_missing_wtkg_measurements_at_any_time
# @IN ANTHA @as num_3
# @OUT ANTHA @as num_4
## remove subjects who are missing wtkg measurements at any time
nullwts <- ANTHA[is.na(ANTHA$wtkg),]$subjid
ANTHA <- ANTHA[!(ANTHA$subjid %in% nullwts),]
rm(nullwts)
# @END remove_subjects_who_are_missing_wtkg_measurements_at_any_time

# @BEGIN remove_subjects_who_are_missing_any_categorical_measurements
# @IN ANTHA @as num_4
# @OUT ANTHA @as num_5
## remove subjects who are missing any categorical measurements
nullcats <- ANTHA[is.na(ANTHA$apgar1) | is.na(ANTHA$apgar5) | is.na(ANTHA$mmaritn) | is.na(ANTHA$mcignum) | is.na(ANTHA$parity) | is.na(ANTHA$gravida) | is.na(ANTHA$meducyrs) | is.na(ANTHA$sesn),]$subjid
ANTHA <- ANTHA[!(ANTHA$subjid %in% nullcats),]
rm(nullcats)
# @END remove_subjects_who_are_missing_any_categorical_measurements

# @BEGIN subtract_20_from_apgar_scores_that_are_greater_than_10
# @IN ANTHA @as num_5
# @OUT ANTHA @as num_6
## subtract 20 from apgar scores that are greater than 10
apgid <- ANTHA[ANTHA$apgar1 > 10,]$subjid
ANTHA[ANTHA$subjid %in% apgid,]$apgar1 <- ANTHA[ANTHA$subjid %in% apgid,]$apgar1 - 20
apgid <- ANTHA[ANTHA$apgar5 > 10,]$subjid
ANTHA[ANTHA$subjid %in% apgid,]$apgar5 <- ANTHA[ANTHA$subjid %in% apgid,]$apgar5 - 20
rm(apgid)
# @END subtract_20_from_apgar_scores_that_are_greater_than_10

# @BEGIN remove_subjects_who_do_not_have_measurements_at_all_five_values_of_agedays
# @IN ANTHA @as num_6
# @OUT ANTHA @as num_7
## remove subjects who do not have measurements at all five values of agedays
len_uni <- function(x){
  return(length(unique(x)))
}
unid <- unique(ANTHA$subjid)
find_ind <- function(i) {
  return(unid[i])
}
age_class <- split(ANTHA$agedays, ANTHA$subjid)
age_class_length <- sapply(age_class, len_uni)
inds_ord <- which((age_class_length == 5))# | (age_class_length == 4))
inds_id <- sapply(inds_ord, find_ind)
ANTHA <- ANTHA[(ANTHA$subjid %in% inds_id),]
rm(age_class, age_class_length, inds_id, inds_ord, unid, len_uni, find_ind)
# @END remove_subjects_who_do_not_have_measurements_at_all_five_values_of_agedays

# @BEGIN code_categorical_variables_as_indicator_variables_binning_some_of_them_first
# @IN ANTHA @as num_7
# @OUT ANTHA @as num_8
## code categorical variables as indicator variables, binning some of them first
## gagebrth (from wikipedia)
## 37-38 : early;   39-40 : full;   41-42 : late;   43 : post
ANTHA$gage_full <- rep(0, dim(ANTHA)[1])
ANTHA$gage_late <- rep(0, dim(ANTHA)[1])
ANTHA$gage_post <- rep(0, dim(ANTHA)[1])
ANTHA[ANTHA$gagebrth == 39 | ANTHA$gagebrth == 40,]$gage_full <- 1
ANTHA[ANTHA$gagebrth == 41 | ANTHA$gagebrth == 42,]$gage_late <- 1
ANTHA[ANTHA$gagebrth == 43,]$gage_post <- 1
ANTHA$gagebrth <- NULL

## apgar (from wikipedia)
## < 4 : low;  4-6 : medium; > 6 : high
ANTHA$ap1_me <- rep(0, dim(ANTHA)[1])
ANTHA$ap1_hi <- rep(0, dim(ANTHA)[1])
ANTHA$ap5_me <- rep(0, dim(ANTHA)[1])
ANTHA$ap5_hi <- rep(0, dim(ANTHA)[1])
ANTHA[ANTHA$apgar1 > 3 & ANTHA$apgar1 < 7,]$ap1_me <- 1
ANTHA[ANTHA$apgar1 > 6,]$ap1_hi <- 1
ANTHA[ANTHA$apgar5 > 3 & ANTHA$apgar5 < 7,]$ap5_me <- 1
ANTHA[ANTHA$apgar5 > 6,]$ap5_hi <- 1
ANTHA$apgar1 <- NULL
ANTHA$apgar5 <- NULL

## mcignum (from lei)
## 0 : none;  1-19 : medium; 20+ : high
ANTHA$cig_me <- rep(0, dim(ANTHA)[1])
ANTHA$cig_hi <- rep(0, dim(ANTHA)[1])
ANTHA[ANTHA$mcignum > 0 & ANTHA$mcignum < 20,]$cig_me <- 1
ANTHA[ANTHA$mcignum > 19,]$cig_hi <- 1
ANTHA$mcignum <- NULL

## parity and gravida (from lei)
## 0 : none; 1 : low; 2 : medium; > 2 : high
ANTHA$par_lo <- rep(0, dim(ANTHA)[1])
ANTHA$par_me <- rep(0, dim(ANTHA)[1])
ANTHA$par_hi <- rep(0, dim(ANTHA)[1])
ANTHA$gra_lo <- rep(0, dim(ANTHA)[1])
ANTHA$gra_me <- rep(0, dim(ANTHA)[1])
ANTHA$gra_hi <- rep(0, dim(ANTHA)[1])
ANTHA[ANTHA$parity == 1,]$par_lo <- 1
ANTHA[ANTHA$parity == 2,]$par_me <- 1
ANTHA[ANTHA$parity > 2,]$par_hi <- 1
ANTHA[ANTHA$gravida == 1,]$gra_lo <- 1
ANTHA[ANTHA$gravida == 2,]$gra_me <- 1
ANTHA[ANTHA$gravida > 2,]$gra_hi <- 1
ANTHA$parity <- NULL
ANTHA$gravida <- NULL

## meducyrs (from lei)
## 0-9 : low; 10-12 : medium; 12+ : high
ANTHA$ed_me <- rep(0, dim(ANTHA)[1])
ANTHA$ed_hi <- rep(0, dim(ANTHA)[1])
ANTHA[ANTHA$meducyrs > 9 & ANTHA$meducyrs < 13,]$ed_me <- 1
ANTHA[ANTHA$meducyrs > 12,]$ed_hi <- 1
ANTHA$meducyrs <- NULL

## mage (from lei)
## < 20 : low; 20-35 : medium; > 35 : high
ANTHA$ag_me <- rep(0, dim(ANTHA)[1])
ANTHA$ag_hi <- rep(0, dim(ANTHA)[1])
ANTHA[ANTHA$mage > 19 &  ANTHA$mage < 36,]$ag_me <- 1
ANTHA[ANTHA$mage > 35,]$ag_hi <- 1
ANTHA$mage <- NULL

catinds <- c(5,7,11,12)
temp <- dummy.data.frame(ANTHA[,catinds], dummy.class = "ALL")
ANTHA <- cbind(ANTHA, temp)
rm(temp, catinds)
ANTHA$siteid <- NULL
ANTHA$feedingn <- NULL
ANTHA$mmaritn <- NULL
ANTHA$sesn <- NULL
ANTHA$siteid5 <- NULL
ANTHA$feedingn1 <- NULL
ANTHA$mmaritn1 <- NULL
ANTHA$sesn25 <- NULL
# @END code_categorical_variables_as_indicator_variables_binning_some_of_them_first

# @BEGIN kludgily_impute_lenhei_at_agedays_==_123
# @IN ANTHA @as num_8
# @OUT ANTHA @as num_9
## kludgily impute lenhei at agedays == 123
set.seed(65537)
## make linear model to predict missing lenhei values
ANTHA_temp <- ANTHA[ANTHA$agedays != 123 & ANTHA$agedays != 2558,c(2:5)]
ANTHA_temp <- ANTHA_temp[complete.cases(ANTHA_temp),]
mod <- lm(lenhei ~., data = ANTHA_temp)
## make predictions
coef <- mod$coef
pred_lenhei <- function(x){
  return(coef[1] + coef[2]*x[2] + coef[3]*x[3] + coef[4]*x[5])
}
ANTHA[ANTHA$agedays == 123,]$lenhei <- apply(ANTHA[ANTHA$agedays == 123,], 1, function(x) pred_lenhei(x))
ANTHA[ANTHA$agedays == 123,]$lenhei <- ANTHA[ANTHA$agedays == 123,]$lenhei + rnorm(length(ANTHA[ANTHA$agedays == 123,]$lenhei), mean = 0, sd = summary(mod)$sigma)
rm(ANTHA_temp, coef, mod, pred_lenhei)
# @END kludgily_impute_lenhei_at_agedays_==_123

# @BEGIN remove_remaining_patients_with_missing_measurements
# @IN ANTHA @as num_9
# @OUT ANTHA @as ANTHA_clean
## remove remaining patients with missing measurements
nullleft <- ANTHA[(is.na(ANTHA$lenhei) | is.na(ANTHA$birthlen)) & ANTHA$agedays != 123,]$subjid
ANTHA <- ANTHA[!(ANTHA$subjid %in% nullleft),]
rm(nullleft)
# @END remove_remaining_patients_with_missing_measurements

# @BEGIN get_who_zscores_using_who_methods
# @IN ANTHA @as ANTHA_clean
# @OUT oldyung
ANTHA_old <- ANTHA
ANTHA$sexn <- ANTHA$sexn + 1

## @begin split_by_age
## @in ANTHA @as ANTHA_clean
## @out yung
## @out oldd
## @out row.names
yung <- ANTHA[ANTHA$agedays != 2558,1:5]
oldd <- ANTHA[ANTHA$agedays == 2558,1:5]
row.names(yung) <- 1:dim(yung)[1]
row.names(oldd) <- 1:dim(oldd)[1]
## @end split_by_age

## @begin add_measure_column_to_yung
## @in yung
## @out yung @as yung_1
add_mea <- function(x){
  if(x != 1462) {
    return('l')
  } else {
    return('h')
  }
}
yung$mea <- sapply(yung$agedays, function(x) add_mea(x))
rm(add_mea)
## @end add_measure_column_to_yung

## igrowup: ages 0 - 5
weianthro <- read.table("~/R/win-library/3.1/igrowup\\weianthro.txt", header = T, sep = "", skip = 0)
lenanthro <- read.table("~/R/win-library/3.1/igrowup\\lenanthro.txt", header = T, sep = "", skip = 0)
bmianthro <- read.table("~/R/win-library/3.1/igrowup\\bmianthro.txt", header = T, sep = "", skip = 0)
hcanthro <- read.table("~/R/win-library/3.1/igrowup\\hcanthro.txt", header = T, sep = "", skip = 0)
acanthro <- read.table("~/R/win-library/3.1/igrowup\\acanthro.txt", header = T, sep = "", skip = 0)
ssanthro <- read.table("~/R/win-library/3.1/igrowup\\ssanthro.txt", header = T, sep = "", skip = 0)
tsanthro <- read.table("~/R/win-library/3.1/igrowup\\tsanthro.txt", header = T, sep = "", skip = 0)
wflanthro <- read.table("~/R/win-library/3.1/igrowup\\wflanthro.txt", header = T, sep = "", skip = 0)
wfhanthro <- read.table("~/R/win-library/3.1/igrowup\\wfhanthro.txt", header = T, sep = "", skip = 0)
source("~/R/win-library/3.1/igrowup\\igrowup_standard.r")
source("~/R/win-library/3.1/igrowup\\igrowup_restricted.r")
igrowup.restricted(FilePath = "~\\rpi\\research\\2015summer\\datathingy\\gates", FileLab = "yung", mydf = yung, sex = sexn, age = agedays, weight = wtkg, lenhei = lenhei, measure = mea)

## who2007: ages 5+
wfawho2007 <- read.table("~/R/win-library/3.1/who2007/wfawho2007.txt", header = T, sep = "", skip = 0)
hfawho2007 <- read.table("~/R/win-library/3.1/who2007/hfawho2007.txt", header = T, sep = "", skip = 0)
bfawho2007 <- read.table("~/R/win-library/3.1/who2007/bfawho2007.txt", header = T, sep = "", skip = 0)
source("~/R/win-library/3.1/who2007/who2007.r")

oldd$agemons <- oldd$agedays / 31
who2007(FilePath = "~\\rpi\\research\\2015summer\\datathingy\\gates", FileLab = "oldd", mydf = oldd, sex = sexn, age = agemons, weight = wtkg, height = lenhei)

## import newly created csvs
oldd_z <- read.csv("~/rpi/research/2015summer/datathingy/gates/oldd_z.csv")
yung_z <- read.csv("~/rpi/research/2015summer/datathingy/gates/yung_z_rc.csv")

## rename variables because the who was not consistent in their names for some raisin :|
yung_z <- rename(yung_z, c("zlen" = "zhfa"))
yung_z <- rename(yung_z, c("zwei" = "zwfa"))
yung_z$zwfl <- NULL
yung_z <- rename(yung_z, c("zbmi" = "zbfa"))

## select portions of data to be merged together
yung_z <- yung_z[,c(1,2,5,10,11,12)]
oldd_z <- oldd_z[,c(1,2,5,9,10,11)]

## merge data
oy <- rbind(yung_z, oldd_z)
oy <- oy[with(oy, order(subjid, agedays)), ]
row.names(oy) <- 1:dim(oy)[1]

oyy <- cbind(oy, ANTHA_old[,-c(1:5)])
write.csv(oyy, file = "oldyung2.csv")
# @END get_who_zscores_using_who_methods

# @END generatezscores
