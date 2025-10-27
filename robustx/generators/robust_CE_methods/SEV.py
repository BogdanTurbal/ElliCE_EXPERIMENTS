from robustx.lib.tasks.Task import Task
from ..CEGenerator import CEGenerator
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from itertools import combinations, product
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from warnings import filterwarnings
filterwarnings("ignore")

class OneHotDecoder:
    """
    This class is a Decoder to transform the One-hot Encoded Features to numerical values
    The idea of this class is to use two mappings between one-hot encoding and numerical values
    and it is used mostly for the transform between population mean and its one-hot encoded version
    """
    def __init__(self):
        self.one_hot_mapping = None # from one-hot tuple to encoded value
        self.one_hot_mapping_reverse = None # from encoded value to the one-hot tuple
        self.column_name = None # save the columns that needs to be encoded
    def fit(self, X):
        index_X = X.groupby(list(X.columns)).count().index # get the one-hot encoded columns
        # create the mapping
        self.one_hot_mapping = dict(zip(index_X,np.arange(len(index_X))))
        self.one_hot_mapping_reverse = dict(zip(np.arange(len(index_X)),index_X))
        # save the column name for future transform
        self.column_name = X.columns
    def transform(self, X):
        # check if the one-hot encoding is fitted
        if self.one_hot_mapping is None:
            raise ValueError("Haven't fitted yet!")
        # save the processed X in a list
        output = []
        for Xi in np.array(X[self.column_name]):
            try:
                output.append(self.one_hot_mapping[tuple(Xi)]) # map the one-hot value to numerical
            except:
                output.append(-1)
        return output
    def reverse_transform(self,value):
        return self.one_hot_mapping_reverse[value] # map the numerical value to one-hot

class DataEncoder:
    """
    This class enables us to input a dataset transform the features into correct version and
    save its mean value
    - Numerical Features: No extra pre-processing needed, save Mean value as the population mean
    - Binary Features: The Standard of Binary Features is the number of unique value in this feature
    equals to 2, it would encode the feature into one column using OnehotEncoder(drop="ifbinary"), \
    and save the mode value as the population mean
    - Categorical Features: The Standard of Binary Features is the number of unique value n in this features
    greater than 2 and the column type is object (if you want some ordinal value be processed in one-hot
    version, you can convert the column type of that feature into string), it would encode the feature
    into n columns for n unique values and the mode value as the population mean.
    - One-hot encoded Features: In some case, the input dataset is already under one-hot encoded version,
    thus, it is hard to get the population mean in this case, the encoder can input a dictionary with the
    merged feature name as the key and the categories in the features as a list. It would use a One-hot
    decoder to transform the one-hot value to a numerical one and then encode it just as the categorical
    features.
    """
    def __init__(self, standard=False):
        # initialization
        self.original_columns = [] # column names after the one-hot features are merged
        self.columns_types = {} # column types for each featurs in the original columns
        self.columns_labelencoder = {} # map for all OnehotEncoders of categorical features
        self.columns_mean = {} # map for all features and its population mean
        self.merge_dict = {} # map for one-encoded features that need to be merged
        self.columns_onehotdecoder = {} # the one-hot decoder of one-hot encoded features in raw data

        self.standard = standard
        if self.standard:
            self.numerical_standard_encoder = {} # dictionary of standard encoder

    def fit(self,df,categorical_dict ={}):
        """
        The fitting function in the DataEncoder can be divided into three main steps:
        1. Merge the one-hot encoded features into features in corresponding to based on OnehotDecoder
        2. Label each features based on their unique values and column types and use the OneHotEncoder
        to encode all the features
        3.
        :param df: The input dataset
        :param categorical_dict: The map between the merged features names and the one-hot
        encoded feature name
        :return: The dataset with merged features based on the One-hot Decoder
        """
        # print("Start to merge the features...")
        self.merge_dict = categorical_dict
        for key,values in self.merge_dict.items():
            # check if the select features are one-hot encoded
            # if df.groupby(values).count().shape[0] != len(values):
            #     raise ValueError("This is not a one-hot vector!")
            self.columns_onehotdecoder[key] = OneHotDecoder()
            self.columns_onehotdecoder[key].fit(df[values]) # fit the one-hot decoder fot data

        # merge the columns representing the same feature
        df_merged = df.copy()
        for key, values in self.merge_dict.items():
            df_merged[key] = self.columns_onehotdecoder[key].transform(df[values])
            # delete the original dataset
            for value in values:
                del df_merged[value]
        # label each columns based on its value details are mentioned in description
        for data_col in df_merged.columns:
            if df_merged[data_col].nunique() == 2:
                self.columns_types[data_col] = "binary"
            elif (df_merged[data_col].dtype == object) and (df_merged[data_col].nunique() > 2):
                self.columns_types[data_col] = "category"
            elif df_merged[data_col].nunique() == 1:
                self.columns_types[data_col] = "constant"
            elif data_col in self.merge_dict.keys():
                self.columns_types[data_col] = "category"
            else:
                self.columns_types[data_col] = "numerical"
        # do the label encoding for the different types of features
        for i, data_col in enumerate(df_merged.columns):
            if (self.columns_types[data_col] == "binary"):
                # fit the columns by One-Hot Encoder for binary features
                self.columns_labelencoder[data_col] = OneHotEncoder(drop="if_binary")
                self.columns_labelencoder[data_col].fit(np.array(df_merged[data_col]).reshape(-1, 1))
                df_merged[data_col] = self.columns_labelencoder[data_col].transform(np.array(df_merged[data_col]).reshape(-1, 1)).toarray()
            elif (self.columns_types[data_col] == "category"):
                # fit the columns by One-Hot Encoder for categorical features
                self.columns_labelencoder[data_col] = OneHotEncoder()
                self.columns_labelencoder[data_col].fit(np.array(df_merged[data_col]).reshape(-1, 1))
            elif (self.columns_types[data_col] == "numerical"):
                if self.standard:
                    self.numerical_standard_encoder[data_col] = StandardScaler()
                    self.numerical_standard_encoder[data_col].fit(np.array(df_merged[data_col]).reshape(-1, 1))
                    df_merged[data_col] = self.numerical_standard_encoder[data_col].transform(np.array(df_merged[data_col]).reshape(-1, 1))
            # elif self.columns_types[data_col] == "constant":
            #     print("The feature",data_col,"is a constant feature, thus it is ignored.")
            #     # ignore the constant columns
            #     continue
            self.original_columns.append(data_col)

        # print("The features are merged! The remaining features are",self.original_columns)

        # print("Start calculating the mean of the features...")

        # calculate the population mean for each features
        for feature,type in self.columns_types.items():
            if type == "category":
                self.columns_mean[feature] = np.array(df_merged[feature].mode())[0]
            elif type == "binary":
                self.columns_mean[feature] = np.array(df_merged[feature].mode())[0]
            elif type == "numerical":
                self.columns_mean[feature] = 0
            elif type == "constant":
                self.columns_mean[feature] = np.array(df_merged[feature].mode())[0]
        # print("Successfuly get the mean of the features.")
        return df_merged

    def transform(self, df):
        # print(df.columns)
        # print(df)
        # first step: transform the dataset to the merged form
        df_output = df.copy() # copy the dataset to be a newer version for processing
        # the difference between merge version and the original version is the merge of one-hot
        # encoded features

        # second step: do the label encoding for categorical and binary features
        for feature, type in self.columns_types.items():
            if type == "numerical":
                if self.standard:
                    df_output[feature] = self.numerical_standard_encoder[feature].transform(np.array(df_output[feature]).reshape(-1,1))
                else:
                    df_output[feature] = df_output[feature]
            elif type == "category":
                if feature in self.merge_dict.keys():
                    continue
                for cate_type in self.columns_labelencoder[feature].categories_[0]:
                    df_output[str(feature)+"="+str(cate_type)] = (df_output[feature] == cate_type)+0
                del df_output[feature]
            elif type == "binary":
                df_output[feature] = self.columns_labelencoder[feature].transform(np.array(df[feature]).reshape(-1, 1)).toarray().astype(int)

        return df_output

class SEV(CEGenerator):
    def __init__(self,
                 task: Task,
                 eps: float = 0.2,
                 k: int = 5,
                 strict: list = None,
                 custom_distance_func=None):

        super().__init__(task, custom_distance_func)
        
        self.model = task.model
        self.data_encoder = DataEncoder(standard=False)

        X = task.training_data.X
        y = task.training_data.y.values.flatten()
        mask = pd.Series(y == 0, index=X.index) 
        X_neg = X.loc[mask]

        self.data_encoder.fit(X_neg)
        self.X = self.data_encoder.transform(X)

        self.X_neg =  self.data_encoder.transform(X_neg)

        self.tol = eps
        self.k = k
        self.strict = strict or []
        self._fitted = False
        self.data_map = pd.DataFrame(np.zeros((len(self.data_encoder.original_columns),len(X.columns))),columns = X.columns, index = self.data_encoder.original_columns)

        self.data_mean = {}
        self.overall_mean = []
        self.choices = list(np.arange(len(self.data_encoder.original_columns)))

        self.features = []

        for index,feature in enumerate(self.data_encoder.original_columns):
            # for numerical feature, the mean is just the mean
            if self.data_encoder.columns_types[feature] == "numerical":
                self.data_mean[feature] = self.data_encoder.columns_mean[feature]
                self.overall_mean.append(self.data_encoder.columns_mean[feature])
                self.data_map.loc[feature,feature] = 1
            # for binary feature, the mean is the mode of the feature
            elif self.data_encoder.columns_types[feature] in ["binary","constant"]:
                try:
                    self.data_mean[feature] = self.data_encoder.columns_labelencoder[feature].transform(np.array([self.data_encoder.columns_mean[feature]]).reshape(-1,1)).toarray()[0,0]
                    self.overall_mean.append(self.data_encoder.columns_labelencoder[feature].transform(np.array([self.data_encoder.columns_mean[feature]]).reshape(-1,1)).toarray()[0,0])
                except:
                    self.data_mean[feature] = self.data_encoder.columns_mean[feature]
                    self.overall_mean.append(self.data_encoder.columns_mean[feature])
                self.data_map.loc[feature, feature] = 1
            # for the categorical features, suppose it is getting from the one-hot feature,
            # then reverse transform back to the one-hot encoded version, suppose it is getting
            # from the categorical features directly, use the OneEncoder to transform the value
            # into one-hot version
            elif self.data_encoder.columns_types[feature] in ["category"]: 
                mode_value = self.data_encoder.columns_mean[feature]
                if feature in self.data_encoder.merge_dict.keys():
                    result = self.data_encoder.columns_onehotdecoder[feature].reverse_transform(mode_value)
                    self.data_mean[feature] = result
                    self.data_map.loc[feature, self.data_encoder.merge_dict[feature]] = 1
                    self.overall_mean += list(result)
                else:
                    result = self.data_encoder.columns_labelencoder[feature].transform([[mode_value]]).toarray()[0]
                    self.data_mean[feature] = result
                    self.overall_mean += list(result)
                    cats = [str(feature) + "=" + str(cat) for cat in self.data_encoder.columns_labelencoder[feature].categories_[0]]
                    self.data_map.loc[feature, cats] = 1
            
            if (strict is not None) and (feature in strict):
                self.choices.remove(index)
        # save the data map
        self.data_map = np.array(self.data_map)

        self.overall_mean = np.array(self.overall_mean)
        
        # self.flexible_mean = pd.DataFrame(self.overall_mean.reshape(1,-1),columns = X.columns)

        # get the final flexible mean
        # self.final_flexible_mean = pd.DataFrame(self.flexible_mean.copy())

        # do a quantile transformation for the numerical features
        # collect the quantiles for the numerical features' means
        # quantile_dict = {}
        # for index,feature in enumerate(self.data_encoder.original_columns):
        #     if self.data_encoder.columns_types[feature] == "numerical":
        #         quantile_loc = (X_neg[feature]<X_neg[feature].mean()).mean()
        #         # get the upper bound of the feature
        #         quantile_upper = np.quantile(X_neg[feature],min(quantile_loc+self.tol,1))
        #         # get the lower bound of the feature
        #         quantile_lower = np.quantile(X_neg[feature],max(quantile_loc-self.tol,0))
        #         # do a linespace to sample 5 points from the lower bound to the upper bound(include)
        #         quantile_dict[feature] = np.linspace(quantile_lower,quantile_upper,k)
        #         self.adjusted_mean = []
        #         # for each value in the quantile_dict[feature], replace the value in the overall_mean with it
        #         # and calculate the score, then choose the furthest point as the adjusted mean
        #         for value in quantile_dict[feature]:
        #             # copy a new flexible mean
        #             # flexible_mean_temp = self.flexible_mean.copy()
        #             # replace the value in the flexible mean
        #             # flexible_mean_temp[feature] = value
        #             # calculate the score
        #             try:
        #                 score = self.model.predict_proba(pd.Series(flexible_mean_temp.ravel()))[0,1]
        #             except:
        #                 score = 1
        #             # if the score is the smallest, then replace the adjusted mean
        #             if (len(self.adjusted_mean) == 0) or (score < self.adjusted_mean[1]):
        #                 self.adjusted_mean = [value,score]
        #         # replace the value in the flexible mean
        #         self.final_flexible_mean[feature] = self.adjusted_mean[0]

        # print("The original overall mean is",self.overall_mean)
        # print("The original flexible mean is",self.final_flexible_mean.values)
        

                
        # save the results
        self.result = {}
        from sklearn.mixture import GaussianMixture
        self.kde = GaussianMixture(n_components=4).fit(np.array(X_neg.dropna()))

        # density = self.kde.score_samples(np.array(X_neg))
        
        self.thresholds = -5


    def _generation_method(self,
                           instance: pd.Series,
                           neg_value: int = 0,
                           column_name: str = "target",
                           **kwargs) -> pd.DataFrame:
        """
        Generate a counterfactual for a single input instance.
        """
        if isinstance(instance, pd.DataFrame):
            instance = instance.iloc[0]
        elif isinstance(instance, pd.Series):
            instance = instance#.drop(labels=column_name)
        else:
            raise ValueError("x must be Series or single‐row DataFrame")

        max_depth = kwargs.get("max_depth")
        if max_depth is None:
            max_depth = len(self.data_encoder.original_columns)
        try:
            cf_df = self.sev_explain(
                Xi=instance.values,
                depth=max_depth,
                mode="minus" if neg_value == 0 else "plus",
            )[0]
            cf_df = pd.DataFrame(cf_df)
            return cf_df
        except IndexError:
            return pd.DataFrame([instance])
    
    def transform(self, Xi, conditions):
        """
        This function aims to transfer Xi based on its boolean vector
        :param Xi: a DataFrame row for training dataset
        :param conditions: a boolean vector represents which feature should take the mean
        :return: Xi_temp: the transferred Xi
        """
        
        remain_columns = conditions.dot(self.data_map)
        Xi_temp = Xi*remain_columns + self.overall_mean *(1-remain_columns)
        return Xi_temp.reshape(1, -1)

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def calculate_all(self,Xi):
        """
        Calculate all the possible combinations for the result
        :param Xi: a numpy array row in the dataset
        :return: The least value Node in the boolean lattice
        """
        # list out all the possible combinations
        combinations = list(product([0,1],repeat = len(self.data_encoder.original_columns)))
        # initialize the least score
        least_score = None
        for combination in combinations:
            # calculate the score of Xi
            score = self.model.predict_proba(self.transform(Xi,np.array(combination)))[0,1]
            # get the least score and the least score population
            if (least_score is None) or (score < least_score):
                self.least_unit = combination
                least_score = score
            # save the combination result in the dictionary
            self.result[combination]= score
        # return the least unit
        return self.least_unit

    def sev_cal(self,Xi, mode = "plus",max_depth=None):
        """
        Calculate the SEV value for Xi
        :param Xi: a DataFrame row for training dataset
        :param mode {'Mean','NegativeMost', 'Counterfactual'} default: 'Mean': the parameter to control
        what kind of SEV that we would like to calculate, 'Mean' represents to search from (0,0,0) and find
        the shortest path from it to first postive term, 'NegativeMost' represents to search the shortest path
        from the least negative outcomes value node to the first postive term and the 'Counterfactual' means to
        search from (1,1,1) to the first negative term value.
        :return: The selected SEV
        """
        if max_depth is None:
            max_depth = len(self.data_encoder.original_columns)
        choices = self.choices
        
        # BFS process
        for choice in  range(1,len(self.data_encoder.original_columns)+1):
            if choice > max_depth:
                return choice
            combs = combinations(choices,choice)
            for comb in combs:
                if mode == "plus":
                    pointer = np.zeros(len(self.data_encoder.original_columns))
                elif mode == "minus":
                    pointer = np.ones(len(self.data_encoder.original_columns))
                pointer[np.array(comb)] = 1-pointer[np.array(comb)]
                # print(pointer)
                # try to collect the score from the result dictionary if it is already calculated
                try:
                    score = self.result[tuple(pointer)]
                except:
                    score = self.model.predict_proba(self.transform(Xi, pointer))[0, 1]
                # for counterfactual the score should be negative
                if mode == "minus":
                    # print("im here")
                    if score < 0.5:
                        # print out the kde of the sample
                        probability = self.kde.score_samples(self.transform(Xi, pointer))
                        # if the probability is not high continue the loop
                        if probability <= self.thresholds:
                            continue
                        return len(comb),probability, self.transform(Xi, pointer)
                else:
                    if score >= 0.5:
                        probability = self.kde.score_samples(self.transform(Xi, pointer))
                        if probability <= self.thresholds:
                            continue
                        return len(comb),probability, self.transform(Xi, pointer)
        return len(comb),[0], self.transform(Xi, pointer)

    # def sev_cal(self,Xi, mode = "plus",max_depth=None):
    #     """
    #     Calculate the SEV value for Xi
    #     :param Xi: a DataFrame row for training dataset
    #     :param mode {'Mean','NegativeMost', 'Counterfactual'} default: 'Mean': the parameter to control
    #     what kind of SEV that we would like to calculate, 'Mean' represents to search from (0,0,0) and find
    #     the shortest path from it to first postive term, 'NegativeMost' represents to search the shortest path
    #     from the least negative outcomes value node to the first postive term and the 'Counterfactual' means to
    #     search from (1,1,1) to the first negative term value.
    #     :return: The selected SEV
    #     """
    #     if max_depth is None:
    #         max_depth = len(self.data_encoder.original_columns)
    #     choices = self.choices

    #     prev_choice = 1
    #     # BFS process
    #     for choice in  range(1,len(self.data_encoder.original_columns)+1):
    #         if choice > max_depth:
    #             return choice,Xi-self.final_flexible_mean.values,True
    #         combs = combinations(choices,choice)
    #         for comb in combs:
    #             if mode == "plus":
    #                 pointer = np.zeros(len(self.data_encoder.original_columns))
    #             elif mode == "minus":
    #                 pointer = np.ones(len(self.data_encoder.original_columns))
    #             pointer[np.array(comb)] = 1-pointer[np.array(comb)]
    #             # print(pointer)
    #             # try to collect the score from the result dictionary if it is already calculated
    #             try:
    #                 score = self.result[tuple(pointer)]
    #             except:
    #                 score = self.model.predict(pd.Series(self.transform(Xi, pointer).ravel()))
    #                 score = float(score.iloc[0])
    #             # for counterfactual the score should be negative
    #             if mode == "minus":
    #                 if score < 0.5:
    #                     return len(comb),self.transform(Xi, pointer) - Xi,False
    #             else:
    #                 if score >= 0.5:
    #                     return len(comb), self.transform(Xi, pointer) - Xi,False
    #         if prev_choice == choice-1 and self.tol != 0:
    #             prev_choice = choice
    #             combs = combinations(choices,choice)
    #             for comb in combs:
    #                 if mode == "plus":
    #                     pointer = np.zeros(len(self.data_encoder.original_columns))
    #                 elif mode == "minus":
    #                     pointer = np.ones(len(self.data_encoder.original_columns))
    #                 pointer[np.array(comb)] = 1-pointer[np.array(comb)]
    #                 # print(pointer)
    #                 # try to collect the score from the result dictionary if it is already calculated
    #                 try:
    #                     score = self.result[tuple(pointer)]
    #                 except:
    #                     score = self.model.predict(pd.Series(self.flexible_transform(Xi, pointer).ravel()))[0]
    #                     score = float(score.iloc[0])
    #                 # for counterfactual the score should be negative
    #                 if mode == "minus":
    #                     if score < 0.5:
    #                         return len(comb),self.flexible_transform(Xi, pointer) - Xi,True
    #                 else:
    #                     if score >= 0.5:
    #                         return len(comb),self.flexible_transform(Xi, pointer) - Xi,True
        
    # def transform(self, Xi, conditions):
    #     """
    #     This function aims to transfer Xi based on its boolean vector
    #     :param Xi: a DataFrame row for training dataset
    #     :param conditions: a boolean vector represents which feature should take the mean
    #     :return: Xi_temp: the transferred Xi
    #     """
    #     remain_columns = conditions.dot(self.data_map)
    #     Xi_temp = Xi*remain_columns + self.overall_mean *(1-remain_columns)
    #     return Xi_temp.reshape(1, -1)
    
    # def flexible_transform(self, Xi, conditions):
    #     """
    #     This function aims to transfer Xi based on its boolean vector
    #     :param Xi: a DataFrame row for training dataset
    #     :param conditions: a boolean vector represents which feature should take the mean
    #     :return: Xi_temp: the transferred Xi
    #     """
    #     remain_columns = conditions.dot(self.data_map)
    #     Xi_temp = Xi*remain_columns + self.final_flexible_mean.values *(1-remain_columns)
    #     return Xi_temp.reshape(1, -1)

    # def sigmoid(self,x):
    #     return 1 / (1 + np.exp(-x))

    # def calculate_all(self,Xi):
    #     """
    #     Calculate all the possible combinations for the result
    #     :param Xi: a numpy array row in the dataset
    #     :return: The least value Node in the boolean lattice
    #     """
    #     # list out all the possible combinations
    #     combinations = list(product([0,1],repeat = len(self.data_encoder.original_columns)))
    #     # initialize the least score
    #     least_score = None
    #     for combination in combinations:
    #         # calculate the score of Xi
    #         score = self.model.predict_proba(pd.Series(self.transform(Xi,np.array(combination)).ravel()))[0,1]
    #         # get the least score and the least score population
    #         if (least_score is None) or (score < least_score):
    #             self.least_unit = combination
    #             least_score = score
    #         # save the combination result in the dictionary
    #         self.result[combination]= score
    #     # return the least unit
    #     return self.least_unit

    # def sev_cal(self,Xi, mode = "plus",max_depth=None):
    #     """
    #     Calculate the SEV value for Xi
    #     :param Xi: a DataFrame row for training dataset
    #     :param mode {'Mean','NegativeMost', 'Counterfactual'} default: 'Mean': the parameter to control
    #     what kind of SEV that we would like to calculate, 'Mean' represents to search from (0,0,0) and find
    #     the shortest path from it to first postive term, 'NegativeMost' represents to search the shortest path
    #     from the least negative outcomes value node to the first postive term and the 'Counterfactual' means to
    #     search from (1,1,1) to the first negative term value.
    #     :return: The selected SEV
    #     """
    #     if max_depth is None:
    #         max_depth = len(self.data_encoder.original_columns)
    #     choices = self.choices

    #     prev_choice = 1
    #     # BFS process
    #     for choice in  range(1,len(self.data_encoder.original_columns)+1):
    #         if choice > max_depth:
    #             return choice,Xi-self.final_flexible_mean.values,True
    #         combs = combinations(choices,choice)
    #         for comb in combs:
    #             if mode == "plus":
    #                 pointer = np.zeros(len(self.data_encoder.original_columns))
    #             elif mode == "minus":
    #                 pointer = np.ones(len(self.data_encoder.original_columns))
    #             pointer[np.array(comb)] = 1-pointer[np.array(comb)]
    #             # print(pointer)
    #             # try to collect the score from the result dictionary if it is already calculated
    #             try:
    #                 score = self.result[tuple(pointer)]
    #             except:
    #                 score = self.model.predict(pd.Series(self.transform(Xi, pointer).ravel()))
    #                 score = float(score.iloc[0])
    #             # for counterfactual the score should be negative
    #             if mode == "minus":
    #                 if score < 0.5:
    #                     return len(comb),self.transform(Xi, pointer) - Xi,False
    #             else:
    #                 if score >= 0.5:
    #                     return len(comb), self.transform(Xi, pointer) - Xi,False
    #         if prev_choice == choice-1 and self.tol != 0:
    #             prev_choice = choice
    #             combs = combinations(choices,choice)
    #             for comb in combs:
    #                 if mode == "plus":
    #                     pointer = np.zeros(len(self.data_encoder.original_columns))
    #                 elif mode == "minus":
    #                     pointer = np.ones(len(self.data_encoder.original_columns))
    #                 pointer[np.array(comb)] = 1-pointer[np.array(comb)]
    #                 # print(pointer)
    #                 # try to collect the score from the result dictionary if it is already calculated
    #                 try:
    #                     score = self.result[tuple(pointer)]
    #                 except:
    #                     score = self.model.predict(pd.Series(self.flexible_transform(Xi, pointer).ravel()))[0]
    #                     score = float(score.iloc[0])
    #                 # for counterfactual the score should be negative
    #                 if mode == "minus":
    #                     if score < 0.5:
    #                         return len(comb),self.flexible_transform(Xi, pointer) - Xi,True
    #                 else:
    #                     if score >= 0.5:
    #                         return len(comb),self.flexible_transform(Xi, pointer) - Xi,True
        
    def sev_explain(self,Xi, depth, mode="plus"):
        choice = depth
        explanations = []

        combs = combinations(self.choices,choice)
        for comb in combs:
            flag = False
            if mode == "plus":
                pointer = np.zeros(len(self.data_encoder.original_columns))
            elif mode == "minus":
                pointer = np.ones(len(self.data_encoder.original_columns))
            # print(pointer)
            pointer[np.array(comb)] = 1-pointer[np.array(comb)]
            # print(pointer)
            # try to collect the score from the result dictionary if it is already calculated
            try:
                score = self.result[tuple(pointer)]
            except:
                score = self.model.predict(pd.Series(self.transform(Xi, pointer).ravel()))
                score = float(score.iloc[0])
            # for counterfactual the score should be negative
            if mode == "minus":
                
                if score < 0.5:
                    explanations.append(self.transform(Xi, pointer) - Xi)
                    flag = True
            else:
                if score >= 0.5:
                    explanations.append(self.transform(Xi, pointer) - Xi)
                    flag = True
            if flag == False and self.tol != 0:
                if mode == "plus":
                    pointer = np.zeros(len(self.data_encoder.original_columns))
                elif mode == "minus":
                    pointer = np.ones(len(self.data_encoder.original_columns))
                pointer[np.array(comb)] = 1-pointer[np.array(comb)]
                # print(pointer)
                # try to collect the score from the result dictionary if it is already calculated
                try:
                    score = self.result[tuple(pointer)]
                except:
                    score = self.model.predict(pd.Series(self.flexible_transform(Xi, pointer).ravel()))[0]
                    score = float(score.iloc[0])
                # for counterfactual the score should be negative
                if mode == "minus":
                    if score < 0.5:
                        explanations.append(self.flexible_transform(Xi, pointer) - Xi)
                else:
                    if score >= 0.5:
                        explanations.append(self.flexible_transform(Xi, pointer) - Xi)
        return explanations