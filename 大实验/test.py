import torch
from config import parse_config
from data_loader import DataBatchIterator
from numpy import *
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
import warnings


def test_textcnn_model(model, test_data, config):
    model.eval()
    test_data_iter = iter(test_data)
    pred_l = []  # 预测标签
    true_l = []  # 实际标签
    for idx, batch in enumerate(test_data_iter):
        model.zero_grad()
        outputs = model(batch.sent)
        pred_l += torch.max(outputs, 1)[1].tolist()
        true_l += batch.label.tolist()
    # score2 = f1_score(true_l,pred_l,average='macro')
    # print("score2:", score2)
    return true_l , pred_l


def main():
# 读配置文件
    config = parse_config()
# 载入测试集合
    test_data = DataBatchIterator(
        config=config,
        is_train=False,
        dataset="test",
        batch_size=config.batch_size)
    test_data.load()
 # 加载模型
    model = torch.load('./results/model.pt')
# 打印模型信息
    print("model: ",model)
# 测试
    true_l, pred_l = test_textcnn_model(model, test_data, config)
# 打印结果
    warnings.filterwarnings("ignore")
    target_names=['news_edu','news_finance','news_house',
                      'news_travel','news_tech','news_sports',
                      'news_game','news_culture','news_car',
                      'news_story','news_entertainment','news_tech',
                      'news_agriculture','news_world','news_stock']
    print(classification_report(true_l, pred_l, target_names=target_names))


if __name__ == "__main__":
    main()
