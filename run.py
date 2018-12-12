from PIL import Image
import numpy as np
from sklearn.feature_selection import SelectFromModel   

#var1="./Dataset/training dataset/train (";
#var2="./Dataset/resized malignant/malignant (";
var3="./Dataset/training dataset/"
var4="train ("      #benign images named as train symbol as 0
var5="malignant ("  #malignant images named as malignant symbol as 1
#for i in range(1,151):
#    temp1=var1+str(i)+").bmp";
#    temp2=var2+str(i)+")"
#    img=Image.open(temp1)
#    img.resize((95,75)).save(temp2+".png")
#    img.close()

feature_vector=np.empty((175,7126), int)
for i in range(1,176):          #reading the test data images, convert them to vector and adding labels
    if(i<=105):
        temp3=var3+var5+str(i)+").png"
        imgtype=1
    else:
        temp3=var3+var4+str(i-105)+").png"
        imgtype=0
    newimg=Image.open(temp3).resize((95,75)).convert('L')
    vector=np.array(newimg).ravel()
    vector=np.append(vector, imgtype)       # adding the image type to the array
    feature_vector[i-1]=vector
for i in range(1,11):                   #shuffle array 10 times
        np.random.shuffle(feature_vector)
#print(feature_vector)


# preparing training metrices
X_train=feature_vector[:,0:7125]
Y_train=feature_vector[:,7125]
np.set_printoptions(precision=3)
X_test=np.empty((70,7125),int)
Y_test=np.empty((70),int)
        
#variables corresponding to the features weights of each feature selection algorithm
univar=np.zeros(7125)
mutinfo=np.zeros(7125)
ridreg=np.zeros(7125)
lasso=np.zeros(7125)
trees=np.zeros(7125)


# univariate feature selection algorithm--Filter methods--feature selection
def univariate_feature_selection(X_train,Y_train):
    from sklearn.feature_selection import SelectKBest, chi2
    test = SelectKBest(score_func=chi2, k=2200)
    fit = test.fit(X_train, Y_train)
    features = fit.transform(X_train)
    global univar
    univar=fit.scores_
    print("Features Scores :",fit.scores_)
    print("Features Selected: 2200")
    TrainlogisticRegression(features, Y_train)
    features = fit.transform(X_test)
    predict(logisticRegr, features, Y_test)


#mutual_info feature selection algorithm--Filter methods--feature selection
def mutual_info_(X_train,Y_train):
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    test = SelectKBest(score_func=mutual_info_classif, k=2700)
    fit = test.fit(X_train, Y_train)
    features = fit.transform(X_train)
    global mutinfo
    mutinfo=fit.scores_
    print("Features Scores :",fit.scores_)
    print("Features Selected: 2700")
    TrainlogisticRegression(features, Y_train)
    features = fit.transform(X_test)
    predict(logisticRegr, features, Y_test)


#Ridge Regression algorithm--Embedded methods--Feature selection
def RidgeRegression(X_train, Y_train):
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train,Y_train)
    global ridreg
    ridreg=ridge.coef_
    print("Features Scores :",ridge.coef_)    # gives the importance of the features
# Set a minimum threshold of 0.00005 to select the features above this threshold
    sfm = SelectFromModel(ridge, threshold=0.00005, prefit=True)  
    features = sfm.transform(X_train)
    print("Features Selected: ",features.shape[1])
    TrainlogisticRegression(features, Y_train)
    features = sfm.transform(X_test)
    predict(logisticRegr, features, Y_test)
    

def Lassocv(X_train, Y_train):              #give very less number of relevant features
    from sklearn.linear_model import Lasso
    clf = Lasso(alpha=0.2).fit(X_train,Y_train)
    global lasso
    lasso=clf.coef_
    print("Features Scores :",clf.coef_)
# Set a minimum threshold of 0.001 to select the features above this threshold
    sfm = SelectFromModel(clf, threshold=0.001, prefit=True)  
    features = sfm.transform(X_train)
    print("Features Selected: ",features.shape[1])
    TrainlogisticRegression(features, Y_train)
    features = sfm.transform(X_test)
    predict(logisticRegr, features, Y_test)


#Extra treesclassifier algorithm--Embedded methods--Feature selection
def Extratreesclassifier(X_train,Y_train):
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier()
    model.fit(X_train, Y_train)
    global trees
    trees=model.feature_importances_
    print("Features Scores :",model.feature_importances_)
# Set a minimum threshold of 0.001 to select the features above this threshold
    sfm = SelectFromModel(model, threshold=0.0001, prefit=True)  
    features = sfm.transform(X_train)
    print("Features Selected: ",features.shape[1])
    TrainlogisticRegression(features, Y_train)
    features = sfm.transform(X_test)
    predict(logisticRegr, features, Y_test)

#Preparing test data vector
def PrepareTestData():
    
    global X_test
    global Y_test 
    test_feature_vector=np.empty((70,7126), int)
    for i in range(1,71):          #reading the test data images, convert them to vector and adding labels
        if(i>25):
            temp3="./Dataset/test dataset/malignant ("+str(i-25)+").png"     # malignant image path
            imgtype=1
        else:
            temp3="./Dataset/test dataset/benign ("+str(i)+").png"        #benign image path
            imgtype=0
        newimg=Image.open(temp3).convert('L')
        vector=np.array(newimg).ravel()
        vector=np.append(vector, imgtype)       # adding the image type to the array
        test_feature_vector[i-1]=vector
#    for i in range(1,11):                   #shuffle array 10 times
#        np.random.shuffle(test_feature_vector)
       
    X_test=test_feature_vector[:,0:7125]
    Y_test=test_feature_vector[:,7125]
    print("\nTest Data:\n ",X_test)
    print("\nTest Expected Output:",Y_test)

    
#Training logistic regression mpdel
def TrainlogisticRegression(X_train, Y_train):   
    from sklearn.linear_model import LogisticRegression
    global logisticRegr
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train,Y_train)

#predict using logistic regression
def predict(logisticRegr, X_test, Y_test):              #predicting the new image data
#    print("\nPredicting new images :")
    predictions = logisticRegr.predict(X_test)
#    print("\t\t",predictions)
    score = logisticRegr.score(X_test, Y_test)
    print("Accuracy: ",(score*100),"%")
    

PrepareTestData()  
print("\n\nTraining logisticRegression Model without Feature Selection Algorithms.....")
TrainlogisticRegression(X_train, Y_train)
predict(logisticRegr, X_test, Y_test)
print("\nExecuting mutual info...........................................\n")
mutual_info_(X_train,Y_train)
print("\n\nExecuting univariate_feature_selection........................\n")
univariate_feature_selection(X_train,Y_train)
print("\n\nExecuting RidgeRegression.....................................\n")
RidgeRegression(X_train, Y_train)
print("\n\nExecuting Extratreesclassifier................................\n")
Extratreesclassifier(X_train,Y_train)
print("\n\nExecuting Lassocv.............................................\n")
Lassocv(X_train, Y_train)

#print("printing the variables")
#print(univar)
#print(mutinfo)
#print(ridreg)
#print(lasso)
#print(trees)

temp1=np.add(univar,mutinfo)
#print(temp1)
temp2=np.add(ridreg,lasso)
#print(temp2)
temp3=np.add(temp1,temp2)
#print(temp3)
temp4=np.add(temp3, trees)

print("\n\nCombined Feature importance :\n\t",temp4)

def topfeatures(arr):
    ind = np.argpartition(arr, -2800)[-2800:]
    ind=np.sort(ind)
    print("Features Selected : ",ind.shape[0])
    global X_train, X_test
    X_train=X_train.T
    X_test=X_test.T
    selected_train=X_train[ind].T
    selected_test=X_test[ind].T
    X_train=X_train.T
    X_test=X_test.T
    
#    print(selected)
    return selected_train, selected_test
    
    
    
print("\n\nTraining model after implementing all feature selection alogorithms............ \n")    
selected_train, selected_test=topfeatures(temp4) 
TrainlogisticRegression(selected_train, Y_train)
predict(logisticRegr, selected_test, Y_test)

from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVC(kernel='poly') 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(X_train, Y_train)
abc=model.score(X_train, Y_train)
#Predict Output
predicted= model.predict(X_test)  
#print(predicted)  

