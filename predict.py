import csv
import numpy as np
from gatgm.tool import set_predict_argument, get_scaler, load_args, load_data, load_model
from gatgm.train import predict
from gatgm.data import MoleDataSet


def predicting(args):
    print('加载参数。')
    scaler = get_scaler(args.model_path)
    print('scaler', scaler)
    train_args = load_args(args.model_path)

    # 从训练参数中获取变量并添加到预测参数中
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('加载数据。')
    test_data = load_data(args.predict_path, args)
    print('加载模型。')
    model = load_model(args.model_path, args.cuda)
    test_pred = predict(model, test_data, args.batch_size, scaler)
    assert len(test_data) == len(test_pred)
    test_pred = np.array(test_pred)
    test_pred = test_pred.tolist()

    print('写入结果。')
    write_smile = test_data.smile()
    with open(args.result_path, 'w', newline='') as file:
        writer = csv.writer(file)

        line = ['Smiles']
        line.extend(args.task_names)
        writer.writerow(line)

        for i in range(len(test_data)):
            line = []
            line.append(write_smile[i])
            line.extend(test_pred[i])
            writer.writerow(line)


if __name__ == '__main__':
    args = set_predict_argument()
    predicting(args)
