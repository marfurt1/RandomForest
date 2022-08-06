url='https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df = pd.read_csv(url, index_col=[0,3])

df=df.drop(columns='Cabin')

df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df[['Sex','Embarked','Survived']]=df[['Sex','Embarked','Survived']].astype('category')

y=df['Survived']
X=df.drop(columns=['Ticket','Survived']).copy()

X['Sex']=X['Sex'].cat.codes
X['Embarked']=X['Embarked'].cat.codes

df[['Sex','Embarked','Survived']]=df[['Sex','Embarked','Survived']].astype('category')

y=df['Survived']
X=df.drop(columns=['Ticket','Survived']).copy()

X['Sex']=X['Sex'].cat.codes

X['Embarked']=X['Embarked'].cat.codes


filename = '../models/finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

