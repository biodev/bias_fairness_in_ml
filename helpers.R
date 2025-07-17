library(tidyverse)
library(tidymodels)
library(yardstick)
library(probably)
library(assertthat)

#`process_meps_csv` and `meps_base_recipe` based on pre-processing steps from the aif360 pythong implementation:
#https=//github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/meps_dataset_panel19_fy2015.py

categorical_vars <- 
c('REGION','SEX','MARRY','FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
'PHQ242','EMPST','POVCAT','INSCOV')

vars_to_keep <- 
c('high_utilization', 'REGION','AGE','SEX','RACE','MARRY',
'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42','PCS42',
'MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV')


process_meps_csv <- function(meps_csv, panel){

    meps_data <- readr::read_delim(meps_csv,show_col_types = FALSE) 

    meps_data <- mutate(
    meps_data,
    RACE = ifelse(
        HISPANX == 2 & RACEV2X == 1,
        "White",
        "Non-White"
        )
    )

    meps_subset <- filter(
        meps_data,
        PANEL == panel
    )

    names(meps_subset) <- sub("53X*$", "", names(meps_subset))

    for(i in c("POVCAT", "INSCOV","OBTOTV", "OPTOTV", "ERTOT", "IPNGTD", "HHTOTD")){
      names(meps_subset) <- sub(paste0(i, "\\d+$"), i, names(meps_subset))
    }

    #this will keep the first one, probably not necessarily what we want if we actually keep these columns
    meps_subset <- meps_subset[,!duplicated(names(meps_subset))]

    #meps_subset <- rename(
    #    meps_subset,
    #    POVCAT = POVCAT15,
    #    INSCOV = INSCOV15
    #)


    meps_subset <- filter(
        meps_subset,
        REGION >= 0 & AGE >= 0 & MARRY >= 0 & ASTHDX >= 0
    )

    meps_subset <- filter(
    meps_subset,
        if_all(c(FTSTU,ACTDTY,HONRDC,RTHLTH,MNHLTH,HIBPDX,CHDDX,ANGIDX,EDUCYR,HIDEG,
                                    MIDX,OHRTDX,STRKDX,EMPHDX,CHBRON,CHOLDX,CANCERDX,DIABDX,
                                    JTPAIN,ARTHDX,ARTHTYPE,ASTHDX,ADHDADDX,PREGNT,WLKLIM,
                                    ACTLIM,SOCLIM,COGLIM,DFHEAR42,DFSEE42,ADSMOK42,
                                    PHQ242,EMPST,POVCAT,INSCOV), ~ .x >= -1)
    )

    #instead of UTILIZATION, define a similar factor for ease of operability in R
    meps_subset <- mutate(
        meps_subset,
        high_utilization=factor(ifelse(OBTOTV + OPTOTV + ERTOT + IPNGTD + HHTOTD < 10, "low", "high"), 
                                   levels=c("low", "high"))
    )

    meps_subset
}

meps_base_recipe <- function(meps_tbl, addtl_vars=NULL){

    if ((missing(addtl_vars) || all(is.na(addtl_vars)) || is.null(addtl_vars)) == F){
        vars_to_keep <- c(vars_to_keep, addtl_vars)
    }

    level_range <- range(unique(unlist(meps_tbl[, categorical_vars])))
    level_univ <- as.character(seq(from = level_range[1], to = level_range[2]))
    level_transform <- function(x) { as.integer(x + 2)}

    meps_rec <- recipe(
        meps_tbl,
        vars = vars_to_keep,
        roles = ifelse(vars_to_keep == "high_utilization", "outcome", "predictor")
    ) %>%
    #update_role(UTILIZATION, new_role = "outcome") %>%
    step_num2factor(
        all_of(categorical_vars),
        transform = level_transform,
        levels = level_univ
    ) %>%
    #step_reweighting(RACE) %>%
    step_dummy(all_of(categorical_vars), one_hot = TRUE) %>%
    step_dummy(RACE) %>%
    step_zv(all_predictors())

    meps_rec

}

meps_base_recipe2 <- function(meps_tbl, addtl_vars=NULL){
  
  if ((missing(addtl_vars) || all(is.na(addtl_vars)) || is.null(addtl_vars)) == F){
    vars_to_keep <- c(vars_to_keep, addtl_vars)
  }
  
  numeric.vars <- setdiff(vars_to_keep, c(categorical_vars, "high_utilization", "RACE", addtl_vars))
  
  meps_rec <- recipe(
    meps_tbl,
    vars = vars_to_keep,
    roles = ifelse(vars_to_keep == "high_utilization", "outcome", "predictor")
  ) %>% 
    step_dummy(all_of(categorical_vars)) %>%
    step_dummy(RACE) %>%
    step_zv(all_predictors()) %>%
    #below deviates from original code, but makes this more useful in other modeling contexts
    step_lincomb(all_predictors()) %>%
    #need to remove this manually since step_lincomb keeps it, but is not estimable
    step_rm(REGION_X4) %>%
    #for the numeric variables, median impute and rescaled to 0-1 to be somewhat comparable to categorical
    step_impute_mean(all_of(numeric.vars)) %>%
    step_range(all_of(numeric.vars),min=0, max=1)
  
  
  meps_rec
  
}

#R implementation of bias-reduction reweighting:
#Kamiran, F., Calders, T. Data preprocessing techniques for classification without discrimination. Knowl Inf Syst 33, 1–33 (2012). https://doi.org/10.1007/s10115-011-0463-8
#Given a (training) dataset tibble, column name of categorical outcome and column name of protected feature
#return the tibble with a new column 'instance_weights' containing the bias-reduction weights
reweight <- function(tbl, outcome, protected_feature){

  group_by(
    tbl, 
    across(all_of(c(outcome, protected_feature)))
  ) %>% 
  mutate(
    oPlusP=n()
  ) %>%
  group_by(
    across(all_of(outcome))
  ) %>%
  mutate(
    O = n()
  ) %>%
  group_by(
    across(all_of(protected_feature))  
  ) %>%
  mutate(
    P = n()
  ) %>%
  ungroup() %>%
  mutate(
    N = n()
  ) %>%
  mutate(
    instance_weights = (O * P) / (N * oPlusP)
  ) %>%
  select(
    -O, -P, -oPlusP, -N
  )

}

#R implementation of disparate_impact: 
#https://github.com/Trusted-AI/AIF360/blob/master/aif360/sklearn/metrics/metrics.py#L537C15-L537C15
#All factor levels are non-positive, positive respectively
#and unprotected, protected respectively
disparate_impact <- function(tbl, outcome, predicted_outcome, protected_feature){

    cmat <- conf_mat(
        group_by(
                tbl, 
                across(all_of(protected_feature))
            ), 
        truth=outcome, 
        estimate=predicted_outcome)

    tcmat <- mutate(
        cmat,
        tidied = lapply(conf_mat, tidy)
        ) %>%
    unnest(tidied)

    pos <- 
        filter(
            tcmat,
            name %in% c("cell_2_1", "cell_2_2")
        ) %>%
        group_by(
            across(all_of(protected_feature))
        ) %>%
        summarize(
            positive = sum(value)
        )

    tot <- 
        tcmat %>%
        group_by(
            across(all_of(protected_feature))
        ) %>%
        summarize(
            total=sum(value)
        )

    rates <- 
        inner_join(
            pos, 
            tot, 
            by=protected_feature
        ) %>%
        mutate(
            pos_rate = positive / total
        )

    unprot_level <- levels(tbl[[protected_feature]])[1]
    prot_level <- levels(tbl[[protected_feature]])[2]

    wide_rates <- 
        pivot_wider(
            rates, 
            id_cols=-c("positive", "total"), 
            names_from=all_of(protected_feature), 
            values_from="pos_rate"
        ) 
    
    wide_rates$disparate_impact <- 
        wide_rates[[prot_level]] / wide_rates[[unprot_level]]

    wide_rates

}

#Compute Balanced Accuracy for a given probability cutoff
#Here factor levels are non-positive, positive respectively
#and unprotected, protected respectively
cutoff_stats <- function(val_tbl, cutoff, outcome, protected_feature){

  #replace .pred_class with newly calculated version
  #however, as the levels are low, high and high is typically the less frequent class, need to adjust cutoff
  called_val <- 
    val_tbl %>%
    mutate(
      .pred_class = as.factor(make_two_class_pred(
        estimate = .pred_low, 
        levels = levels(.data[[outcome]]), 
        threshold = 1-cutoff
      ))
    )

  #recompute balanced accuracy, could also add in sensitivity/specificity
    #for more traditional assessment of alternate cutoffs

  called_bal_acc <- 
    bal_accuracy_vec(
      truth=called_val[[outcome]], 
      estimate=called_val$.pred_class)
    

  #compute disparate impact (DI) and base rates used in the DI computation
  
  di <- disparate_impact(
        called_val, 
        outcome, 
        ".pred_class", 
        protected_feature
    )

  bind_cols(
    tibble(
      balanced_accuracy=called_bal_acc, 
      p_cutoff=cutoff
    ), 
    di
  )

}

#Compute Balanced Accuracy for a range of probability cutoffs
alternate_cutoff_stats <- function(val_tbl, outcome, protected_feature){
  
  #iterate over a series of candidate thresholds and collate results into a tibble
  bind_rows(lapply(seq(0, .5, length.out=50), function(x){
    
    cutoff_stats(val_tbl, x, outcome, protected_feature)
    
  }))
    
}
