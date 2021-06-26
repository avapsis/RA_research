#using Pkg
#Pkg.add(["StatsBase", "Random", "LinearAlgebra", "Plots", "DataFramesMeta", "DecisionTree", "DataFrames", 
#"MLDataUtils", "CSV", "CategoricalArrays", "ScikitLearn"])

using  StatsBase, Random, LinearAlgebra, Plots, DataFramesMeta, DecisionTree, DataFrames, 
MLDataUtils, CSV, CategoricalArrays, ScikitLearn

# DATA LOAD
cd("/home/gridsan/avapsi/data/reprocessed_data/Elective/")
#cd("/Users/annitavapsi/Dropbox (MIT)/MGH-All-Data/NSQIP_Emergency/Data/data_processed/Elective/")

train_X_mortal =  CSV.read("train_sample_X_mort.csv")
train_y_mortal =  CSV.read("train_sample_y_mort.csv").MORT
test_X_mortal = CSV.read("test_sample_X_mort.csv")
test_y_mortal =  CSV.read("test_sample_y_mort.csv").MORT

train_X_morbid =  CSV.read("train_sample_X_morb.csv")
train_y_morbid =  CSV.read("train_sample_y_morb.csv").MORB_ANY
test_X_morbid =  CSV.read("test_sample_X_morb.csv")
test_y_morbid =  CSV.read("test_sample_y_morb.csv").MORB_ANY;

column_to_keep = [
    :SEX,:RACE_NEW,:ETHNICITY_HISPANIC,:INOUT,:Age,:SURGSPEC,:DIABETES,:SMOKE,:DYSPNEA,
    :FNSTATUS2,:VENTILAT,:HXCOPD,:ASCITES,:HXCHF,:HYPERMED,:RENAFAIL,:DIALYSIS,
    :DISCANCR,:WNDINF,:STEROID,:WTLOSS,:BLEEDDIS,:TRANSFUS,:PRSEPIS,:PRSODM,:PRBUN,:PRCREAT,:PRALBUM,:PRBILI,
    :PRSGOT,:PRALKPH,:PRWBC,:PRHCT,:PRPLATE,:PRPTT,:PRINR,:BMI
]

train_X_mortal = train_X_mortal[:,column_to_keep]
test_X_mortal = test_X_mortal[:,column_to_keep]
train_X_morbid = train_X_morbid[:,column_to_keep]
test_X_morbid = test_X_morbid[:,column_to_keep];

# printing dataframe sizes
println(size(train_X_mortal))
println(size(train_y_mortal))
println(size(test_X_mortal))
println(size(test_y_mortal))

println(size(test_X_morbid))
println(size(train_y_morbid))
println(size(test_X_morbid))
println(size(test_y_morbid))


function categorical_specs(X_train, X_test)
    ordered_categorical_variables = [
        :DYSPNEA, :PRSEPIS, :FNSTATUS2
        ]
    for col_name in ordered_categorical_variables
        X_train[!, col_name] = CategoricalArray(X_train[!, col_name], ordered=true)
        X_test[!, col_name] = CategoricalArray(X_test[!, col_name], ordered=true)
        println("Transformed column '", col_name, "' to Ordered Categorical")
    end
    
    non_ordered_categorical_variables = [
        :SEX,:RACE_NEW,:ETHNICITY_HISPANIC,:INOUT,:SURGSPEC,:DIABETES,:SMOKE,
        :VENTILAT,:HXCOPD,:ASCITES,:HXCHF,:HYPERMED,:RENAFAIL,:DIALYSIS,:DISCANCR,:WNDINF,:STEROID,:WTLOSS,
        :BLEEDDIS,:TRANSFUS
        ] 
    
    for col_name in non_ordered_categorical_variables
        X_train[!, col_name] = CategoricalArray(X_train[!, col_name], ordered=false)
        X_test[!, col_name] = CategoricalArray(X_test[!, col_name], ordered=false)
        println("Transformed column '", col_name, "' to Non-Ordered Categorical")
    end
    
    println("##########################################################################################")
    println("##########################################################################################")

    return X_train, X_test
end


train_X_mortal, test_X_mortal = categorical_specs(train_X_mortal, test_X_mortal)
train_X_morbid, test_X_morbid = categorical_specs(train_X_morbid, test_X_morbid);

colnames = ["Sex","Race","Ethnicity","Inpatient","Age","Surgical Specialty", 
    "Diabetes", "Smoker", "Dyspnea", 
    "Functional Status", "Ventilator Dependent","COPD","Ascites", "CHF", "Hypertension", "Renal Failure", "Dialysis", 
    "Disseminated Cancer",
    "Wound Infection", "Steroid Use", "Weight Loss", "Bleeding Disorder", "Pre-op Transfusion", "Pre-op Sepsis",
    "Pre-op Na", "Pre-op BUN", "Pre-op Creatinine", "Pre-op Albumin", "Pre-op Bilirubin", "Pre-op SGOT", 
    "Pre-op Alk Phos", "Pre-op WBC", "Pre-op Hematocrit", "Pre-op Platelet", "Pre-op PTT", "Pre-op INR", 
    "BMI"]

names!(train_X_mortal, Symbol.(colnames))
names!(test_X_mortal, Symbol.(colnames));
names!(train_X_morbid, Symbol.(colnames))
names!(test_X_morbid, Symbol.(colnames));


# Model results output
df_final_results_random_split = DataFrame(id = Any[], seed = Any[],
          depth = Any[], minbucket = Any[], criterion = Any[], cp = Any[],
          auc_train = Float64[], auc_valid = Float64[], auc_test = Float64[])

function oct(train_X, test_X, train_y, test_y, seed, outcome, minbucket)
    id = "seed=$(seed)___outcome=$(outcome)___minbucket=$(minbucket)___nsqip"
    outcome = "$(outcome)"
    # Process Data
    Random.seed!(seed)
    @show size(train_X)
    @show size(test_X)

    println("Running OCT for: $(id)")
    # Run Optimal Classification Trees
    # Set of OCT learner and training grid
    oct_lnr = IAI.OptimalTreeClassifier(
        ls_num_tree_restarts=60,
        random_seed=seed,
        treat_unknown_level_missing=true,
        minbucket=minbucket,
        missingdatamode=:separate_class
    )
    grid = IAI.GridSearch(
        oct_lnr,
        max_depth = 9:10,  ###CHANGED DEPTH
        criterion = :gini
    )
    println("Started Fitting the OCT grid search for $(outcome)...")
    IAI.fit!(grid, train_X, train_y, validation_criterion=:auc, sample_weight=:autobalance)
    println("Finished Fitting the OCT grid search for $(outcome)")
    lnr = IAI.get_learner(grid)
    y_pred_proba = IAI.predict_proba(lnr, test_X)
    CSV.write(
        joinpath(@__DIR__, "max_depth15_outputs/y_pred_proba_$(id).csv"),
        y_pred_proba
    )
      

    println("Chosen Parameters:")
    for (param, val) in grid.best_params
            println("$(param): $(val)")
    end

    train_auc = IAI.score(grid, train_X, train_y, criterion=:auc)
    test_auc = IAI.score(grid, test_X, test_y, criterion=:auc)

    println("OCT-1 Results of tree $(id) ")
    println("--------------------------------")
    println("Max Depth $(id) = ", grid.best_params[:max_depth])
    println("Training AUC $(id) = ", round(100 * train_auc, digits=3), "%")
    println("Testing AUC $(id)  = ", round(100 * test_auc, digits=3), "%")
    println("##########################################################################################")
    println("##########################################################################################")

    lnr = IAI.get_learner(grid)
    IAI.show_in_browser(IAI.ROCCurve(lnr, test_X, test_y))

    # Save the tree
    IAI.write_html(joinpath(@__DIR__, "$(id)_auc$(round(Int, test_auc * 1000)).html"),
        lnr)
    IAI.write_json(joinpath(@__DIR__, "$(id)_auc$(round(Int, test_auc * 1000)).json"),
        lnr)
    println("Saved the trees as HTML and JSON")

    return [
        id, seed, grid.best_params[:max_depth], minbucket,
        grid.best_params[:criterion], grid.best_params[:cp],
        train_auc, train_auc, test_auc
    ]
end


minbucket = 50
seed = 1
## Fit OCTs for mortality
oct_results_mort = oct(
   train_X_mortal,
   test_X_mortal,
   train_y_mortal,
   test_y_mortal,
   seed,
   "hosp_mortality",
   minbucket
)
push!(df_final_results_random_split, vcat(oct_results_mort))

# Fit OCTs for morbidity
oct_results_morb = oct(
    train_X_morbid,
    test_X_morbid,
    train_y_morbid,
    test_y_morbid,
    seed,
    "hosp_morbidity",
    minbucket
)
push!(df_final_results_random_split, vcat(oct_results_morb))
println(df_final_results_random_split)
CSV.write(
    joinpath(@__DIR__, "results_random_split_nsqip.csv"),
    df_final_results_random_split
)


