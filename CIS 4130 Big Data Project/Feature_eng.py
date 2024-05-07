#!/usr/bin/env python
# coding: utf-8

# In[1]:


spark


# In[2]:


import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import size, split
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# In[3]:


sdf = spark.read.parquet('gs://my-bigdata-project-dk/cleaned/data_cleaning.parquet', header=True)


# In[4]:


sdf.printSchema()


# In[5]:


sdf.count()


# In[6]:


sdf.select("attack", "dataset", "label").summary("count", "min", "max", "mean").show()


# In[7]:


# Create an assembler and scaler
# input columns used for feature
input_columns = [
    'L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 
    'OUT_PKTS', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL', 
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 
    'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES', 
    'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT', 
    'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 
    'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'TCP_WIN_MAX_IN', 
    'TCP_WIN_MAX_OUT'
]

# Creating the VectorAssembler
assembler = VectorAssembler(inputCols=input_columns, outputCol="features")

# Configure MinMaxScaler
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")


# In[8]:


sdf_pipe = Pipeline(stages=[assembler,scaler])

# Call .fit to transform the data
transformed_sdf = sdf_pipe.fit(sdf).transform(sdf)


# In[9]:


# Split the data into 70% training and 30% test sets  
trainingData, testData = transformed_sdf.randomSplit([0.7, 0.3], seed=42)

# Create a LogisticRegression Estimator
lr = LogisticRegression(featuresCol="features", labelCol="LABEL")

# Fit the model to the training data
lr_model = lr.fit(trainingData)

# Show model coefficients and intercept
print("Coefficients: ", lr_model.coefficients)
print("Intercept: ", lr_model.intercept)

# Test the model on the testData
test_results = lr_model.transform(testData)


# In[10]:


# Show the confusion matrix
lr_predictions = lr_model.transform(testData)
test_results.groupby('label').pivot('prediction').count().sort('label').show()

confusion_matrix = test_results.groupby('label').pivot('prediction').count().fillna(0).collect()

def calculate_recall_precision(confusion_matrix):
    tn = confusion_matrix[0][1]  # True Negative
    fp = confusion_matrix[0][2]  # False Positive
    fn = confusion_matrix[1][1]  # False Negative
    tp = confusion_matrix[1][2]  # True Positive
    precision = tp / ( tp + fp )            
    recall = tp / ( tp + fn )
    accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
    f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )
    return accuracy, precision, recall, f1_score

print("Accuracy, Precision, Recall, F1 Score")
print( calculate_recall_precision(confusion_matrix) )


# In[11]:


#try random forest 
from pyspark.ml.classification import RandomForestClassifier

# Create a RandomForest model.
rf = RandomForestClassifier(featuresCol="features", labelCol="LABEL")

#pipeline
sdf_rf_pipe = Pipeline(stages=[assembler,scaler,rf])
# Define the parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 50]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

# Create the evaluator
evaluator = BinaryClassificationEvaluator(labelCol="LABEL", metricName="areaUnderROC")

# Create and configure the CrossValidator
cv = CrossValidator(estimator=sdf_rf_pipe,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=3)

# Split data and fit the model
(trainingData, testData) = sdf.randomSplit([0.7, 0.3])
cvModel = cv.fit(trainingData)

# Train the model
#rf_model = rf.fit(trainingData)

# Make predictions.
#predictions = rf_model.transform(testData)

# Select example rows to display.
#predictions.select("prediction", "features").show(5)


# In[ ]:


#finding ROC and best parameters

bestModel = cvModel.bestModel
predictions = cvModel.transform(testData)
roc_auc = evaluator.evaluate(predictions)
print("Test Area Under ROC: {}".format(roc_auc))

rfModel = bestModel.stages[-1]

# Print the parameters of the best RandomForest model
print("Best model's numTrees: ", rfModel.getNumTrees)
print("Best model's maxDepth: ", rfModel.getMaxDepth())


# In[12]:


#tuned model eval

sdf_pipe = Pipeline(stages=[assembler,scaler])

# Split the data into 70% training and 30% test sets  
trainingData, testData = transformed_sdf.randomSplit([0.7, 0.3], seed=42)

# Call .fit to transform the data
transformed_sdf = sdf_pipe.fit(sdf).transform(sdf)

# Create a RandomForest model to confirm these are best parameters
rf = RandomForestClassifier(featuresCol="features", labelCol="LABEL",numTrees=50,maxDepth=15)
model = rf.fit(trainingData)
# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "LABEL", "features").show(5)

# Test the model on the testData
test_results = model.transform(testData)

# Show the confusion matrix
test_results.groupby('label').pivot('prediction').count().sort('label').show()

confusion_matrix = test_results.groupby('label').pivot('prediction').count().fillna(0).collect()

def calculate_recall_precision(confusion_matrix):
    tn = confusion_matrix[0][1]  # True Negative
    fp = confusion_matrix[0][2]  # False Positive
    fn = confusion_matrix[1][1]  # False Negative
    tp = confusion_matrix[1][2]  # True Positive
    precision = tp / ( tp + fp )            
    recall = tp / ( tp + fn )
    accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
    f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )
    return accuracy, precision, recall, f1_score

print("Accuracy, Precision, Recall, F1 Score")
print( calculate_recall_precision(confusion_matrix) )


# In[ ]:


#ORIGINAL RF MODEL EVAL

# Test the model on the testData
#test_results = rf_model.transform(testData)

# Show the confusion matrix
#test_results.groupby('label').pivot('prediction').count().sort('label').show()

#confusion_matrix = test_results.groupby('label').pivot('prediction').count().fillna(0).collect()

#def calculate_recall_precision(confusion_matrix):
 #   tn = confusion_matrix[0][1]  # True Negative
 #   fp = confusion_matrix[0][2]  # False Positive
  #  fn = confusion_matrix[1][1]  # False Negative
  #  tp = confusion_matrix[1][2]  # True Positive
  #  precision = tp / ( tp + fp )            
  #  recall = tp / ( tp + fn )
 #   accuracy = ( tp + tn ) / ( tp + tn + fp + fn )
 #   f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )
#    return accuracy, precision, recall, f1_score
#
#print("Accuracy, Precision, Recall, F1 Score")
#print( calculate_recall_precision(confusion_matrix) )


# In[ ]:


# saving model to files
rf_model.write().overwrite().save('gs://my-bigdata-project-dk/models/rf_model')


# In[ ]:


#saving transformed data
output_path = "gs://my-bigdata-project-dk/trusted/trusted_data.parquet"

# Save the DataFrame to Parquet
transformed_sdf.write.mode('overwrite').parquet(output_path)


# In[13]:


#random forest roc viz
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
import matplotlib.pyplot as plt

# Selecting the probability of the positive class
probabilities = predictions.select("probability").rdd.map(lambda x: x[0][1]).collect()
labels = predictions.select("LABEL").rdd.map(lambda x: x[0]).collect()

# Computing the ROC points
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(labels, probabilities)
roc_auc = auc(fpr, tpr)

# Ploting ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[14]:


#logistic regression roc viz
# Selecting the probability of the positive class
probabilities = lr_predictions.select("probability").rdd.map(lambda x: x[0][1]).collect()
labels = predictions.select("LABEL").rdd.map(lambda x: x[0]).collect()

# Computing the ROC points
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(labels, probabilities)
roc_auc = auc(fpr, tpr)

# Ploting ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[61]:


coeff = lr_model.coefficients.toArray().tolist()

# Loop through the features to extract the original column names. Store in the var_index dictionary
var_index = dict()
for variable_type in ['numeric']:
    for variable in predictions.schema["features"].metadata["ml_attr"]["attrs"][variable_type]:
         print(f"Found variable: {variable}" )
         idx = variable['idx']
         name = variable['name']
         var_index[idx] = name      # Add the name to the dictionary

# Loop through all of the variables found and print out the associated coefficients
for i in range(len(var_index)):
    print(f"Coefficient {i} {var_index[i]}  {coeff[i]}")


# In[28]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Assuming `predictions` DataFrame has the necessary probability and label columns
probabilities = predictions.select("probability").rdd.map(lambda x: x[0][1]).collect()
labels = predictions.select("LABEL").rdd.map(lambda x: x[0]).collect()

precision, recall, thresholds = precision_recall_curve(labels, probabilities)

# Plot Precision-Recall curve
plt.figure(figsize=(14, 10)) 
plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()


# In[30]:


# Correlation matrix using Seaborn
import seaborn as sns
# Convert the numeric values to vector columns
vector_column = "correlation_features"
# Choose the numeric (Double) columns 
numeric_columns = [
    'PROTOCOL','IN_PKTS','OUT_PKTS', 'TCP_FLAGS', 'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'DURATION_IN', 'DURATION_OUT', 'MIN_TTL', 'MAX_TTL', 
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN','RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_BYTES', 
    'RETRANSMITTED_OUT_PKTS','NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'NUM_PKTS_256_TO_512_BYTES', 
    'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES'
]
assembler = VectorAssembler(inputCols=numeric_columns, outputCol=vector_column)
sdf_vector = assembler.transform(sdf).select(vector_column)

# Create the correlation matrix, then get just the values and convert to a list
matrix = Correlation.corr(sdf_vector, vector_column).collect()[0][0]
correlation_matrix = matrix.toArray().tolist() 
# Convert the correlation to a Pandas dataframe
correlation_matrix_df = pd.DataFrame(data=correlation_matrix, columns=numeric_columns, index=numeric_columns) 

heatmap_plot = plt.figure(figsize=(20,10))  
# Set the style for Seaborn plots
sns.set_style("white")

sns.heatmap(correlation_matrix_df, 
            xticklabels=correlation_matrix_df.columns.values,
            yticklabels=correlation_matrix_df.columns.values,  cmap="Greens", annot=True)
plt.savefig("correlation_matrix.png")


# In[ ]:





# In[ ]:




