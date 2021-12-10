import numpy as np

'''
    验证时采用该组指标衡量模型（直接采用交叉熵损失，无法真实反应出模型的分割效果）
'''
class eval_metrics(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, label, preds):
        for lbl, pred in zip(label, preds):
            self.confusion_matrix += self.fast_hist(lbl.flatten(), pred.flatten())

    def fast_hist(self, label, pred):
        mask = (label >= 0) & (label < self.num_classes)
        hist = np.bincount(self.num_classes * label[mask].astype(int) + pred[mask],
                           minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)    # 统计数组中每项出现的词频，并做成n*n矩阵
        return hist

    # 重置混淆矩阵
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    """
        Returns accuracy score evaluation result.
            - overall accuracy，总体准确率
            - mean accuracy，平均准确率
            - mean IOU，所有类别的平均iou
            - fwavacc 
    """
    def get_results(self):
        hist = self.confusion_matrix    # 每一行表示：某一类别，有多少被分为第一列（类别1），有多少被分为第二列（类别2），，，总共num_classes列
        # 1.计算对于所有类别，分类的准确率：对角线上的总和/混淆矩阵的总和
        acc = np.diag(hist).sum() / hist.sum()    # np.diag(hist)取出对角线上的值；对角线的值即为所有分类对的

        # 2.分别计算每一类别的准确率
        acc_cls = np.diag(hist) / hist.sum(axis=1)    # [1, num_classes]
        acc_cls = np.nanmean(acc_cls)    # 平均每一类别的准确率

        # 3.每个类别的IOU：交集/(A+B-交集)
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))    # [1, num_classes]
        mean_iou = np.nanmean(iou)    # 所有类别的平均iou

        # 4.标签中每种类别像素点占比
        freq = hist.sum(axis=1) / hist.sum()    # [1, num_classes]，标签中每个类别的像素点占总体的占比
        # 加权综合IOU指标
        weighted_integration_IOU = (freq[freq > 0] * iou[freq > 0]).sum()    # 计算一个综合的IOU，权重就是每种类别像素点在总体的占比。设计思想类似于F1，在P和R之间的调和

        # 5.各类别分别的iou
        cls_iou = dict(zip(range(self.num_classes), iou))

        return {
                "overall_ACC": acc,
                "mean_ACC": acc_cls,
                "weighted_integration_IOU": weighted_integration_IOU,
                "mean_IOU": mean_iou,
                "class_IOU": cls_iou,
            }
