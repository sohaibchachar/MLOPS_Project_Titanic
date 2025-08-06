from src.logger import get_logger
from src.custom_exception import CustomException
from src.feature_store import RedisFeatureStore
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd 
import os
import pickle

logger = get_logger(__name__)

class TrainingModel:
    def __init__ (self,feature_store:RedisFeatureStore,model_save_path = "artifacts/model"):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model   = None
        os.makedirs(self.model_save_path,exist_ok=True)
        logger.info("model training initilized")


    def load_data_from_redis(self,entity_ids):
        try:
            logger.info("Extracting data from redis")

            data = []

            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features:
                    data.append(features)
                else:
                    logger.warning("feature not found")
            return data
        except Exception as e:
            logger.error(f"Error while loading data from redis {e}")
            raise CustomException(str(e))             

    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()
            train_entity_ids , test_entity_ids = train_test_split(entity_ids,test_size=0.2)
            train_data = self.load_data_from_redis(train_entity_ids)
            test_data = self.load_data_from_redis(test_entity_ids)

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            X_train = train_df.drop("Survived",axis = 1)
            logger.info(X_train.columns)
            X_test = test_df.drop("Survived",axis= 1)
            y_train = train_df["Survived"]
            y_test = test_df["Survived"]
            return X_train,X_test,y_train,y_test
        
        except Exception as e:
            logger.error(f"Error while preparing data for training{e}")
            raise CustomException(str(e)) 


    def hyperparameter_tuning(self,X_train,y_train):
        try:
            param_distributions = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
            }
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(rf, param_distributions, n_iter=10, cv=3, scoring='accuracy', random_state=42)
            random_search.fit(X_train, y_train)
            logger.info(f"Best parameters are {random_search.best_params_}")
            return random_search.best_estimator_
        except Exception as e:
            logger.error(f"Error while hyperparameter tuning{e}")
            raise CustomException(str(e)) 


    def train_and_evaluate(self,X_train , y_train , X_test, y_test):
        try:
            best_rf_model = self.hyperparameter_tuning(X_train,y_train)
            y_pred = best_rf_model.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            logger.info(f"Accuracy is {accuracy}")
            self.save_model(best_rf_model)
        except Exception as e:
            logger.error(f"Error while model training {e}")
            raise CustomException(str(e)) 

    def save_model(self,model):
        try:
            model_filename = os.path.join(self.model_save_path, "random_forest_model.pkl") 
            with open(model_filename,'wb') as file_model:
                pickle.dump(model,file_model)   
            logger.info(f"Model has been saved at {model_filename}")
        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException(str(e))            

    def run(self):
        try:
            logger.info("Starting model training  pipeline")
            X_train,X_test,y_train,y_test = self.prepare_data()
            self.train_and_evaluate(X_train,y_train,X_test,y_test)
            logger.info("End of model trainiung pipeline")
        except Exception as e:
            logger.error(f"Error while model training pipeline {e}")
            raise CustomException(str(e))     

if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_trainer = TrainingModel(feature_store)
    model_trainer.run()




