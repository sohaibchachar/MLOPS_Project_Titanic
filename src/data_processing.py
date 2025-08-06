from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
from src.custom_exception import CustomException
from src.feature_store import RedisFeatureStore
from config.paths_config import *
import pandas as pd  

logger  = get_logger(__name__)


class DataProcessing:
    def __init__(self, train_data_path, test_data_path, features_store : RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_resampled = None
        self.y_resampled = None
        self.feature_store = features_store

        logger.info("Data Processing is initialized")

    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logger.info("Data reading is successfully completed")
        except Exception as e:
            logger.error(f"Error while reading data {e}")
            raise CustomException(str(e))        


    def preprocess_data(self):
        try:
            self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())

            self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])

            self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())

            self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1})

            self.data['Embarked'] = self.data['Embarked'].astype('category').cat.codes

            self.data['Familysize'] = self.data['SibSp'] + self.data['Parch'] + 1

            self.data['Isalone'] = (self.data['Familysize'] == 1).astype(int)

            self.data['HasCabin'] = self.data['Cabin'].notnull().astype(int)

            self.data['Title'] = self.data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).map(
                {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
            ).fillna(4)

            self.data['Pclass_Fare'] = self.data['Pclass'] * self.data['Fare']

            self.data['Age_Fare'] = self.data['Age'] * self.data['Fare']

            logger.info("Data Preprocessing is done")



        except Exception as e:
            logger.error(f"Error while preprocessing data {e}")
            raise CustomException(str(e))


    def handele_imbalanced_data(self):
        try:
            X =self.data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Familysize', 'Isalone', 'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']]
            y = self.data['Survived']
            smote = SMOTE(random_state=42)
            self.X_resampled, self.y_resampled = smote.fit_resample(X, y)
            logger.info("Handled Imbalanced Data Successfully")



        except Exception as e:
            logger.error(f"Error while handling imbalanced data {e}")
            raise CustomException(str(e))
        

    def store_features_in_redis(self):
        try:
            batch_data = {}
            for idx, row in self.data.iterrows():
                entity_id = row["PassengerId"]
                features = {
                    "Age" : row['Age'],
                    "Fare" : row["Fare"],
                    "Pclass" : row["Pclass"],
                    "Sex" : row["Sex"],
                    "Embarked" : row["Embarked"],
                    "Familysize": row["Familysize"],
                    "Isalone" : row["Isalone"],
                    "HasCabin" : row["HasCabin"],
                    "Title" : row["Title"],
                    "Pclass_Fare" : row["Pclass_Fare"],
                    "Age_Fare" : row["Age_Fare"],
                    "Survived" : row["Survived"]
                }
                batch_data[entity_id] = features
            self.feature_store.store_batch_features(batch_data)
            logger.info("Data has been sent to feature store")
        except Exception as e:
            logger.error(f"Error while feature storing {e}")
            raise CustomException(str(e))    


    def retrive_features_redis_store(self,entity_id):
        features = self.feature_store.get_features(entity_id)
        if features:
            return features
        return None
    

    def run(self):
        try:
            logger.info("Starting Data Preprocessing Pipeline")
            self.load_data()
            self.preprocess_data()
            self.handele_imbalanced_data()
            self.store_features_in_redis()
            logger.info("Ending Data Preprocessing Pipeline")
        except Exception as e:
            logger.error(f"Error while running data preprocessing pipeline {e}")
            raise CustomException(str(e))   
            
if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    data_processor = DataProcessing(TRAIN_PATH,TEST_PATH,feature_store)
    data_processor.run()
    print(data_processor.retrive_features_redis_store(entity_id = 705))

