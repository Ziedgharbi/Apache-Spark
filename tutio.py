import pyspark
import pandas as pd
from  pyspark.sql import SparkSession 

spark=SparkSession.builder.appName('Tutio').getOrCreate()


direct= "C:/Users/Zied/Nextcloud/Formation/Python/GITHUB/Apache Spark/"
data=pd.read_csv(direct+"Cancer_Data.csv")
print(data)

data.columns

# create spark session : obligatoire

spark
#read dataset

df=spark.read.csv(direct+'Cancer_Data.csv')

#structure de df 
df
# print df
df.show()

#configure option 
df=spark.read.option('header','true').csv(direct+'Cancer_Data.csv',inferSchema=True) # inferSchema for columns types

df.show()

type(df)

df.head()

#print schema for sql dataframe
df.printSchema()

df.dtypes

df.schema


# other way to read csv 

df=spark.read.csv(direct+'Cancer_Data.csv',header=True, inferSchema=True)

df.head(5)

df.printSchema()
df.dtypes
df.schema.fields
df.show()

type(df)

#df columns
df.columns

#select column by name
df.select('area_se')
df.select('area_se').show()

df.select(['area_se',"area_worst"]).show()

#df types
df.dtypes

df.select(['area_se',"area_worst"]).describe().show()

### adding collumns

df=df.withColumn('area_se more 11', df['area_se']+1000)

df.select(['area_se',"area_se more 11"]).describe().show()

#drop columns

df=df.drop("area_se more 11")
df.columns

# rename column
df=df.withColumnRenamed('area_se', "new area_se")
df.show()


"""  new tutorial       """

import pyspark
from pyspark.sql import SparkSession

session=SparkSession.builder.appName("Test").getOrCreate()

data=session.read.csv(direct+"test.csv", header=True,inferSchema=True,sep=";")

data.columns

data.printSchema()
data.head(3)
data.show()

#drop null row 

data.na.drop().show()

## how=any where row has at least 1 null /// how=all drop row when all clomnus are null
data.na.drop(how='any').show()

## thresh=2 at least row has 2 no null to be kept
data.na.drop(how='any', thresh=2).show()


#subset: remove only from specific columns, subset
data.na.drop(how="any", subset=["Salary"]).show()


#fill missing value

data.na.fill(0.000,subset=["Name"]).show()

data.na.fill({"Name":"Unknown", "Experience": 00}).show()


# fill na with imputer : strategy, mean, median,...

from pyspark.ml.feature import Imputer

imputer= Imputer(inputCols=['Age','Experience','Salary'],
                 outputCols=['{}_imputed'.format(c) for c in ["Age","Experience",'Salary']])
imputer.setStrategy('mean')

imputer.fit(data).transform(data).show()


#♦ apply filter
data.filter('Salary<4000').show()
data.filter('Salary>=3500').select(["Name", "Salary"]).show()

data.filter((data['Salary']>3500) & (data["Experience"]>5)).show()
data.filter((data['Salary']>3500) | (data["Experience"]>5)).show()

data.filter(~(data['Salary']>3500) ).show()  # not condition

#GroupBy and aggregate function
data.groupBy("Name").count().show()
data.groupBy("Name").sum().show()
data.groupBy("Name").sum("Salary").show()

data.groupBy("Name").max("Salary").show()


#total sum salary

data.agg({'Salary':'sum'}).show()


###### Exemple spark ML: regression ######

import pyspark
from  pyspark.sql import SparkSession 

spark=SparkSession.builder.appName('Tutio').getOrCreate()

data=session.read.csv(direct+"test.csv", header=True,inferSchema=True,sep=";")

data.show()

data.printSchema()
data.columns


#groupe variable or independant variable [Age,Experience]-------> new feature : independant

from pyspark.ml.feature import VectorAssembler

vector=VectorAssembler(inputCols=["Age", "Experience"], outputCol="Independant_feature")

data=data.na.drop(how="any")

output=vector.transform(data)
output.show()


dataset=output.select(["Independant_feature","Salary"])
dataset.show()

# model definition 
from pyspark.ml.regression import LinearRegression 

#train and test set 

train_data, test_data=dataset.randomSplit([0.75,0.25])

train_data.show()

test_data.show()

model=LinearRegression(featuresCol="Independant_feature", labelCol="Salary")

res=model.fit(train_data)

res.coefficients
res.intercept

#evaluation with test_data
res.evaluate(test_data).predictions.show()

rslt=res.evaluate(test_data)

rslt.r2 , rslt.meanSquaredError
