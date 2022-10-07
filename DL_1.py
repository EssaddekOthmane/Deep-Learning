import pandas as pd












df=pd.read_excel("DB aout - Copie.xlsx")
df=df.dropna()
df = df.sample(frac = 1)

columns=[ 'S0_F0', 'S1_F0', 'S2_F0', 'S3_F0',
       'S4_F0', 'S5_F0', 'S6_F0', 'S7_F0', 'S8_F0', 'S9_F0', 'S10_F0',
       'S11_F0', 'S12_F0', 'S13_F0', 'S14_F0', 'S0_F1', 'S0_F2', 'S0_F3',
       'S0_F4', 'S1_F1', 'S1_F2', 'S1_F3', 'S1_F4', 'S2_F1', 'S2_F2', 'S2_F3',
       'S2_F4', 'S3_F1', 'S3_F2', 'S3_F3', 'S3_F4', 'S4_F1', 'S4_F2', 'S4_F3',
       'S4_F4', 'S5_F1', 'S5_F2', 'S5_F3', 'S5_F4', 'S6_F1', 'S6_F2', 'S6_F3',
       'S6_F4', 'S7_F1', 'S7_F2', 'S7_F3', 'S7_F4', 'S8_F1', 'S8_F2', 'S8_F3',
       'S8_F4', 'S9_F1', 'S9_F2', 'S9_F3', 'S9_F4', 'S10_F1', 'S10_F2',
       'S10_F3', 'S10_F4', 'S11_F1', 'S11_F2', 'S11_F3', 'S11_F4', 'S12_F1',
       'S12_F2', 'S12_F3', 'S12_F4', 'S13_F1', 'S13_F2', 'S13_F3', 'S13_F4',
       'S14_F1', 'S14_F2', 'S14_F3', 'S14_F4', 'S0_F5', 'S1_F5', 'S2_F5',
       'S3_F5', 'S4_F5', 'S5_F5', 'S6_F5', 'S7_F5', 'S8_F5', 'S9_F5', 'S10_F5',
       'S11_F5', 'S12_F5', 'S13_F5', 'S14_F5']


X=df[columns].values
Y1=df['ref1'].values.reshape(-1, 1)
Y2=df['ref2'].values.reshape(-1, 1)
st.dataframe(df)



PredictorScaler=StandardScaler()
TargetVarScaler1=StandardScaler()
TargetVarScaler2=StandardScaler()
 

PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit1=TargetVarScaler1.fit(Y1)
TargetVarScalerFit2=TargetVarScaler2.fit(Y2)
 

X=PredictorScalerFit.transform(X)
Y1=TargetVarScalerFit1.transform(Y1)
Y2=TargetVarScalerFit2.transform(Y2)
 


X_train1, X_test1, y1_train, y1_test = train_test_split(X, Y1, test_size=0.3, random_state=42)
X_train2, X_test2, y2_train, y2_test = train_test_split(X, Y2, test_size=0.3, random_state=42)
 
