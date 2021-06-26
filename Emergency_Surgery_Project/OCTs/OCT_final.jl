#using Pkg
#Pkg.add(["StatsBase", "Random", "LinearAlgebra", "Plots", "DataFramesMeta", "DecisionTree", "DataFrames", 
#"MLDataUtils", "CSV", "CategoricalArrays", "ScikitLearn"])

using  StatsBase, Random, LinearAlgebra, Plots, DataFramesMeta, DecisionTree, DataFrames, 
MLDataUtils, CSV, CategoricalArrays, ScikitLearn

# DATA LOAD
cd("/home/gridsan/avapsi/data/reprocessed_data/Emergency")
#cd("/Users/annitavapsi/Dropbox (MIT)/MGH-All-Data/NSQIP_Emergency/Data/data_processed/Emergency")

train_X =  CSV.read("train1_X_emerg.csv")
train_y =  CSV.read("train1_y_emerg.csv").need_for_ICU
test_X = CSV.read("test1_X_emerg.csv")
test_y =  CSV.read("test1_y_emerg.csv").need_for_ICU;

X_columns = [
    :SEX,:RACE_NEW,:ETHNICITY_HISPANIC,:Age,:SURGSPEC,:DIABETES,:SMOKE,:DYSPNEA,:FNSTATUS2,:HXCOPD,
    :ASCITES,:HXCHF,:HYPERMED,:RENAFAIL,:DIALYSIS,:DISCANCR,:WNDINF,:STEROID,:WTLOSS,:BLEEDDIS,
    :TRANSFUS,:PRSEPIS,:PRSODM,:PRBUN,:PRCREAT,:PRWBC,:PRHCT,:PRPLATE,:BMI 
]
#,:PRALBUM,:PRBILI,:PRSGOT,:PRALKPH,:PRPTT,:PRINR

train_X = train_X[!,X_columns]
test_X = test_X[!,X_columns]

function categorical_specs(X_train, X_test)
    ordered_categorical_variables = [
        :DYSPNEA, :PRSEPIS, :FNSTATUS2 #:WNDCLAS, 
        ]
    for col_name in ordered_categorical_variables
        X_train[!,col_name] = CategoricalArray(X_train[!,col_name], ordered=true)
        X_test[!,col_name] = CategoricalArray(X_test[!,col_name], ordered=true)
        println("Transformed column '", col_name, "' to Ordered Categorical")
    end
    
    non_ordered_categorical_variables = [
        :SEX,:RACE_NEW,:ETHNICITY_HISPANIC,:SURGSPEC,:DIABETES,:SMOKE,:DYSPNEA,:HXCOPD,:ASCITES,
        :HXCHF,:HYPERMED,:RENAFAIL,:DIALYSIS,:DISCANCR,:WNDINF,:STEROID,:WTLOSS,
        :BLEEDDIS,:TRANSFUS
        ] 
    
    for col_name in non_ordered_categorical_variables
        X_train[!,col_name] = CategoricalArray(X_train[!,col_name], ordered=false)
        X_test[!,col_name] = CategoricalArray(X_test[!,col_name], ordered=false)
        println("Transformed column '", col_name, "' to Non-Ordered Categorical")
    end
    
    println("##########################################################################################")
    println("##########################################################################################")

    return X_train, X_test
end


train_X, test_X = categorical_specs(train_X, test_X);

colnames = ["Sex","Race","Ethnicity","Age","Surgical Specialty", "Diabetes", "Smoker", "Dyspnea", 
    "Functional Status","COPD","Ascites", "CHF", "Hypertension", "Renal Failure", "Dialysis", "Disseminated Cancer",
    "Wound Infection", "Steroid Use", "Weight Loss", "Bleeding Disorder", "Pre-op Transfusion", "Pre-op Sepsis",
    "Pre-op Na", "Pre-op BUN", "Pre-op Creatinine","Pre-op WBC", "Pre-op Hematocrit", "Pre-op Platelet","BMI"]
    #"Pre-op Albumin","Pre-op Bilirubin", "Pre-op SGOT","Pre-op Alk Phos","Pre-op PTT", "Pre-op INR",
names!(train_X, Symbol.(colnames))
names!(test_X, Symbol.(colnames));

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
        max_depth = 8:11,  ###CHANGED DEPTH
        criterion = :gini,
    )
    println("Started Fitting the OCT grid search for $(outcome)...")
    IAI.fit!(grid, train_X, train_y, validation_criterion=:auc, sample_weight=:autobalance)
    println("Finished Fitting the OCT grid search for $(outcome)")
    lnr = IAI.get_learner(grid)
    y_pred_proba = IAI.predict_proba(lnr, test_X)
    CSV.write(
        joinpath(@__DIR__, "outputs/y_pred_proba_$(id).csv"),
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
    # Save roc curve
    curve = IAI.ROCCurve(lnr, test_X, test_y)
    IAI.write_html(joinpath(@__DIR__, "outputs/$(id)_auc$(round(Int, test_auc * 1000))_roccurve.html"),
         curve)
    # Save the tree
    IAI.write_html(joinpath(@__DIR__, "outputs/$(id)_auc$(round(Int, test_auc * 1000)).html"),
        lnr)
    IAI.write_json(joinpath(@__DIR__, "outputs/$(id)_auc$(round(Int, test_auc * 1000)).json"),
        lnr)
    println("Saved the trees and roc curve  as HTML and JSON")

    return [
        id, seed, grid.best_params[:max_depth], minbucket,
        grid.best_params[:criterion], grid.best_params[:cp],
        train_auc, train_auc, test_auc
    ]
end


minbucket = 100
seed = 1
# Fit OCTs for need_for_ICU
oct_results = oct(
   train_X,
   test_X,
   train_y,
   test_y,
   seed,
   "need_for_ICU",
   minbucket
)
push!(df_final_results_random_split, vcat(oct_results_mort))
println(df_final_results_random_split)

CSV.write(
    joinpath(@__DIR__, "results_random_split_nsqip_need_for_ICU.csv"),
    df_final_results_random_split
)

# changed categorical specs fundtion 
# depth = 5
