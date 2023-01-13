import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def main():
    data = pd.read_csv('NFLdraftdata.csv')
    qb_features = ['pos','pick','player', 'year', 'height_inches', 'weight', 'passing.attempts', 'passing.comp.pct',
       'passing.completions', 'passing.games', 'passing.pass.ints',
       'passing.pass.tds', 'passing.pass.yards', 'passing.seasons']
    df = data[qb_features]
    df = df[df['pos'] == 'QB']
    df = df.dropna()
    
    df['PowRank'] = None
    
    years = pd.unique(df['year'])
    
    for year in years:
      percentsArr = []
      newdf = df[df['year'] == year]
      newdf['Rank'] = newdf['pick'].rank(method='min',ascending=True)
      for i in newdf.index:
         if newdf['pick'][i] != 300:
            perc = ((len(newdf)-(newdf['Rank'][i]))/len(newdf)) * 100
         else:
            perc = 0
         percentsArr.append(perc)
      df.loc[newdf.index,'PowRank'] = percentsArr
      
    df = df.replace(',','', regex=True)
    prediction_target = df.PowRank.astype(float)

    qb_features_clean = ['year','height_inches', 'weight', 'passing.attempts', 'passing.comp.pct',
       'passing.completions', 'passing.games', 'passing.pass.ints',
       'passing.pass.tds', 'passing.pass.yards', 'passing.seasons']
    df = df[qb_features_clean]
    df = df.astype(float)

    train_DF, val_DF, train_picks, val_picks = train_test_split(df, prediction_target, random_state=1)

    qb_model = RandomForestRegressor(random_state=1)
    qb_model.fit(train_DF,train_picks)
    qb_preds = qb_model.predict(val_DF)

    qb_val_mae = mean_absolute_error(val_picks, qb_preds)
    print("Validation MAE for Random Forest Model: {}".format(qb_val_mae))

if __name__ == "__main__":
    main()