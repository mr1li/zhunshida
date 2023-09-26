from gluonts.model.deepar import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
from pathlib import Path
from gluonts.mx.trainer import Trainer
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
def demand(pre_length,date,customer_name,customer_part_no,supplier_name,supplier_part_no,manufacture_name,site_db,file_name,k,delta):
    name1 = customer_name + '_' + customer_part_no + '_error'
    name2 = customer_name + '_' + customer_part_no + '_model'
    name3= customer_name + '_' + customer_part_no + '_forecast'+'.csv'
    path = 'E://mxnet//tmp//' + name2
    name1 = 'E://mxnet//tmp//' + name2 + '//' + name1
    name3='E://mxnet//tmp//' + name2+'//'+name3
    con=pre_length
    if k==True:
        df = pd.read_csv(file_name)
        # sorted_df = df.sort_values(by=["customer_name", "customer_part_no"])
        df['eta'] = pd.to_datetime(df['eta']).dt.strftime('%Y-%m-%d')
        result_df = df.groupby(['customer_name', 'customer_part_no', 'eta'], as_index=False)['quantity'].sum()
        result_df = result_df.groupby(['customer_name', 'customer_part_no'], as_index=False).apply(
            lambda x: x.sort_values(by='eta')).reset_index(drop=True)
        filtered_data = result_df[(result_df['customer_name'] == '2e9d0015563e8f6c3614c219c759934c') & (result_df['customer_part_no'] == 'fa63b13e7f6ba2d7596dd292659498bc')]
        df = pd.DataFrame(filtered_data)
        # 将 'eta' 列设置为日期类型
        df['eta'] = pd.to_datetime(df['eta'])
        # 创建一个完整的日期范围
        date_range = pd.date_range(df['eta'].min(), df['eta'].max(), freq='D')
        # 使用 merge 将原始数据与完整的日期范围合并，并使用 fillna 填充缺失的值
        df = date_range.to_frame(index=False, name='eta').merge(df, on='eta', how='left').fillna({'quantity': 0})
        df['quantity'] = df['quantity'].rolling(window=30, min_periods=1).mean()
        train_dataset = ListDataset([{"start": df['eta'].min(), "target": df['quantity']}], freq="D")
        estimator = DeepAREstimator(
            prediction_length=pre_length,
            context_length=pre_length,  # 预测未来30个时间点
            freq="D",  # 时间序列频率
            trainer=Trainer(epochs=100),  # 训练的时期数
        )
        predictor = estimator.train(train_dataset)
        num_windows = (len(train_dataset[0]['target']) - pre_length) //pre_length # 计算可以预测的窗口数量
        error = []
        for i in range(num_windows):
            start_idx = i * pre_length
            end_idx = pre_length + (i + 1) * pre_length
            # 构造输入数据，使用历史数据作为 context
            input_data = [{'target': train_dataset[0]['target'][start_idx:end_idx], 'start': pd.Timestamp('2023-09-01')}]

            input_ds = ListDataset(input_data, freq='D')

            # 使用模型进行预测
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=input_ds,
                predictor=predictor,
                num_samples=100,
            )

            forecasts = list(forecast_it)
            tss = list(ts_it)

            # 获取预测分布和真实值
            distribution = [forecast.quantile(0.5) for forecast in forecasts]

            true_values = [ts[-pre_length:] for ts in tss]
            true_values = true_values[0].values.flatten()

            errors = [abs(d - t) / (t+1) for d, t in zip(distribution[0], true_values)]
            error = error + errors
        error = np.array(error)
        sort_error = np.sort(error)

        if not os.path.exists(path):
            # 如果不存在则创建文件夹
            os.makedirs(path)
        predictor.serialize(Path(path))
        np.save(name1,sort_error)
    if k==False:
        error=np.load(name1+'.npy')
        predictor=Predictor.deserialize(Path(path))
        #构造预测集：
        date=datetime.strptime(date, "%Y-%m-%d")
        df = pd.read_csv(file_name)
        # sorted_df = df.sort_values(by=["customer_name", "customer_part_no"])
        df['eta'] = pd.to_datetime(df['eta']).dt.strftime('%Y-%m-%d')
        result_df = df.groupby(['customer_name', 'customer_part_no', 'eta'], as_index=False)['quantity'].sum()
        result_df = result_df.groupby(['customer_name', 'customer_part_no'], as_index=False).apply(
            lambda x: x.sort_values(by='eta')).reset_index(drop=True)
        filtered_data = result_df[(result_df['customer_name'] == '2e9d0015563e8f6c3614c219c759934c') & (
                    result_df['customer_part_no'] == 'fa63b13e7f6ba2d7596dd292659498bc')]
        df = pd.DataFrame(filtered_data)
        # 将 'eta' 列设置为日期类型
        df['eta'] = pd.to_datetime(df['eta'])
        # 创建一个完整的日期范围
        date_range = pd.date_range(df['eta'].min(), df['eta'].max(), freq='D')
        # 使用 merge 将原始数据与完整的日期范围合并，并使用 fillna 填充缺失的值
        df = date_range.to_frame(index=False, name='eta').merge(df, on='eta', how='left').fillna({'quantity': 0})
        out_name = name3
        df.to_csv(out_name,index=False,line_terminator='\n')
        df['quantity'] = df['quantity'].rolling(window=7, min_periods=1).mean()

        test_dataset = ListDataset([{"start": date- timedelta(days=con), "target":np.concatenate((df['quantity'][-pre_length:],np.zeros(pre_length)))}], freq="D")
        print(test_dataset)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_dataset,  # 使用训练数据集的信息进行预测
            predictor=predictor,
            num_samples=100  # 可以根据需要调整采样次数
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        prediction=forecasts[0].mean
        alpha=error[int(delta*len(error))]
        prediction2=prediction*alpha




        print(prediction)

        print(alpha)
        #结果保存
        date_list = [date + timedelta(days=i) for i in range(pre_length)]
        date_strings = [date.strftime('%Y-%m-%d') for date in date_list]
        out= {'date':date_strings,'mean': prediction, str(delta): prediction2}
        print(out)
        out=pd.DataFrame(out)
        out_name=name3
        print(out_name)
        out.to_csv(out_name,index=False,line_terminator='\n')

















if __name__ == "__main__":
    file_name=r'F:\cpan\桌面\data2\out.csv'
    demand(15,'2023-09-01','a','b',0,0,0,0,file_name,True,0.8)
    demand(15, '2023-09-01', 'a', 'b', 0, 0, 0, 0, file_name, False, 0.8)
