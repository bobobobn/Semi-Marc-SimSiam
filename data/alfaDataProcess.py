import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from collections import Counter
from scipy.fft import fft
import glob
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from .data_preprocess import CWRUdata
from DCAe1d_alfa import create_dcae_label


def load_and_preprocess_data(end_path):
    files = glob.glob(os.path.join(end_path, '*.xlsx'))
    print(f"Found {len(files)} files.")
    dfs = []
    label_count = 2
    end_dict = {}

    for file in files:
        if '~$' in file:
            continue
        file_name = os.path.basename(file)
        df = pd.read_excel(file)
        if '1reduced_no_failure_combined' in file_name:
            df['label'] = 1
            label_key = 1
        else:
            df['label'] = label_count
            label_key = label_count
            label_count += 1
        dfs.append(df)
        end_dict[label_key] = (label_key, df)

    if not dfs:
        raise ValueError("No valid files to process.")

    result_df = pd.concat(dfs, ignore_index=True)
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(result_df.drop(columns=['label', '%time'], errors='ignore'))
    scaled_df = pd.DataFrame(scaled_features, columns=result_df.columns.drop(['label', '%time'], errors='ignore'))
    scaled_df['label'] = result_df['label'].values

    return scaled_df

def check_data(data):
    if np.any(np.isnan(data)):
        print("Data contains NaN values")
    if np.any(np.isinf(data)):
        print("Data contains infinite values")
    print("Data check passed")

def create_balanced_dataset(scaled_df, sample_size=420):
    sampled_data_list = []
    sampled_labels_list = []

    for label in scaled_df['label'].unique():
        label_data = scaled_df[scaled_df['label'] == label]
        label_sample = label_data.sample(n=min(sample_size, len(label_data)), random_state=42)
        sampled_data_list.append(label_sample)
        sampled_labels_list.extend([label] * len(label_sample))

    sampled_data = pd.concat(sampled_data_list, ignore_index=True)
    sampled_labels = np.array(sampled_labels_list)

    return sampled_data, sampled_labels

def create_imbalanced_dataset(Train_X, Train_Y, imbalance_ratio=1):
    print('构造不平衡数据集')
    imbalanced_data_list = []
    imbalanced_labels_list = []

    # 获取标签为1的数据
    class_1_data = Train_X[Train_Y == 1]
    class_1_count = len(class_1_data)

    if class_1_count == 0:
        return None, None  # 无标签为1的数据，返回None

    # 添加标签为1的数据到不平衡数据列表
    imbalanced_data_list.append(class_1_data)
    imbalanced_labels_list.append(np.ones(len(class_1_data)))

    # 对其他标签的数据进行采样
    for i in range(2, int(Train_Y.max()) + 1):
        class_data = Train_X[Train_Y == i]
        if class_data.empty:
            print(f"No data for class {i}")
            continue
        sample_size_i = int(class_1_count / imbalance_ratio)
        sampled_data = class_data.sample(n=min(sample_size_i, len(class_data)), random_state=42)
        if not sampled_data.empty:
            imbalanced_data_list.append(sampled_data)
            imbalanced_labels_list.append(np.ones(len(sampled_data)) * i)

    # 合并所有数据
    if not imbalanced_data_list:
        return None, None

    imbalanced_train_data = pd.concat(imbalanced_data_list, ignore_index=True)
    imbalanced_train_labels = np.concatenate(imbalanced_labels_list)

    return imbalanced_train_data, imbalanced_train_labels

def one_hot(Train_Y, Test_Y):
    Train_Y = np.array(Train_Y).reshape([-1, 1])
    Test_Y = np.array(Test_Y).reshape([-1, 1])
    Encoder = OneHotEncoder()
    Encoder.fit(Train_Y)
    Train_Y = Encoder.transform(Train_Y).toarray()
    Test_Y = Encoder.transform(Test_Y).toarray()
    Test_Y = np.asarray(Test_Y, dtype=np.int32)
    Train_Y = np.asarray(Train_Y, dtype=np.int32)
    return Train_Y, Test_Y

def valid_test_slice(Test_X, Test_Y):
    test_size = 0.25 / (0.25 + 0.25)
    ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
    for train_index, test_index in ss.split(Test_X, Test_Y):
        X_valid, X_test = Test_X[train_index], Test_X[test_index]
        Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
        return X_valid, Y_valid, X_test, Y_test


def plot_signals(original_signals, generated_signals, label):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for signal in original_signals:
        plt.plot(signal, color='green')
    plt.title(f'Original Signals for Label {label}')

    plt.subplot(1, 2, 2)
    for signal in generated_signals:
        plt.plot(signal, color='red')
    plt.title(f'Generated Signals for Label {label}')

    plt.show()




def data_augmentation(Train_X, Train_Y, over_sampling):
    if over_sampling not in ['GAN', 'SMOTE', 'ADASYN', 'RANDOM', 'none']:
        raise ValueError("Invalid over_sampling method. Choose from ['GAN', 'SMOTE', 'ADASYN', 'RANDOM', 'none']")

    # Fill NaN values
    imputer = SimpleImputer(strategy='mean')
    if Train_X.shape[0] == 0:
        raise ValueError("No samples to impute")

    Train_X = imputer.fit_transform(Train_X)

    # 获取标签为1的数据量
    class_1_count = len(Train_X[Train_Y == 1])
    all_generated_samples = []
    all_generated_labels = []

    if over_sampling == 'GAN':
        classified_data = {}
        classified_label = {}
        aa = []
        bb = []
        for index, label in enumerate(Train_Y):
            if label not in classified_data:
                classified_data[label] = []
                classified_label[label] = []
            classified_data[label].append(Train_X[index])
            classified_label[label].append(Train_Y[index])
        for label in classified_data:
            print('label', label)
            classified_data[label] = np.array(classified_data[label])
            classaa = classified_data[label]
            classbb = np.concatenate(classaa)

            print(f'classbb shape: {classbb.shape}')

            #result = generate_augmented_data(classaa)
            # result = enhanced_generate_augmented_data(classaa)

            resultt = np.concatenate(result)
            resultplt = resultt.flatten()
            print(f'resultt shape: {resultplt.shape}')

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            for signal in classaa:
                plt.plot(signal, color='blue')
            plt.subplot(1, 2, 2)
            for signal in result:
                plt.plot(signal, color='red', alpha=0.5)
            plt.title(f'Original and Generated Signals for Label {label}')
            plt.legend(['Original', 'Generated'])
            plt.show()

            relabel = np.ones(len(result))
            relabel[:] = label
            all_generated_samples.append(result)
            all_generated_labels.extend(relabel)

    elif over_sampling == 'SMOTE':

        classified_data = {}
        classified_label = {}
        for index, label in enumerate(Train_Y):
            if label not in classified_data:
                classified_data[label] = []
                classified_label[label] = []
            classified_data[label].append(Train_X[index])
            classified_label[label].append(Train_Y[index])

        all_generated_samples = []
        all_generated_labels = []

        for label in classified_data:
            print('Processing label', label)
            classified_data[label] = np.array(classified_data[label])
            original_signals = np.vstack(classified_data[label])  # 确保 original_signals 是二维数组

            # 使用SMOTE进行数据增强
            oversample = SMOTE(k_neighbors=3)
            Train_X_resampled, Train_Y_resampled = oversample.fit_resample(Train_X, Train_Y)

            # 添加生成的样本和标签到列表中
            all_generated_samples.append(Train_X_resampled)
            all_generated_labels.extend(Train_Y_resampled)

            # 重新分类数据
            new_classified_data = {}
            new_classified_label = {}
            for index, new_label in enumerate(Train_Y_resampled):
                if new_label not in new_classified_data:
                    new_classified_data[new_label] = []
                    new_classified_label[new_label] = []
                new_classified_data[new_label].append(Train_X_resampled[index])
                new_classified_label[new_label].append(Train_Y_resampled[index])

            for new_label in new_classified_data:
                print('New label', new_label)
                new_classified_data[new_label] = np.array(new_classified_data[new_label])
                generated_signals = np.vstack(new_classified_data[new_label])  # 确保 generated_signals 是二维数组

                plot_signals(original_signals, generated_signals, new_label)

        Train_X_combined = np.vstack(all_generated_samples)
        Train_Y_combined = np.array(all_generated_labels)

        resu = Counter(Train_Y_combined)
        print('Enhanced Train_Y counts', resu)








    elif over_sampling == 'ADASYN':
        classified_data = {}
        classified_label = {}

        for index, label in enumerate(Train_Y):
            if label not in classified_data:
                classified_data[label] = []
                classified_label[label] = []
            classified_data[label].append(Train_X[index])
            classified_label[label].append(Train_Y[index])

        all_generated_samples = []
        all_generated_labels = []

        for label in classified_data:
            print('label', label)
            classified_data[label] = np.array(classified_data[label])
            classaa = classified_data[label]
            classbb = np.vstack(classaa)  # 确保 classbb 是二维数组

            relabela = np.ones(len(classbb))
            relabela[:] = label

            if len(np.unique(relabela)) > 1:  # 确保 relabela 中有多个类别
                oversample = ADASYN(n_neighbors=2)
                result, _ = oversample.fit_resample(classbb, relabela)
            else:
                result = classbb  # 如果只有一个类别，直接使用原始数据

            plt.figure(figsize=(15, 5))
            for signal in classbb:
                plt.plot(signal, color='blue')
            for signal in result:
                plt.plot(signal, color='red')
            plt.title(f'Original and Generated Signals for Label {label}')
            plt.legend(['Original', 'Generated'])
            plt.show()

            all_generated_samples.append(result)
            all_generated_labels.extend([label] * len(result))

        Train_X = np.vstack(all_generated_samples)
        Train_Y = np.array(all_generated_labels)
        resu = Counter(Train_Y)
        print('增强后的Train_Y个数', resu)



    elif over_sampling == 'RANDOM':
        oversample = RandomOverSampler()
        Train_X, Train_Y = oversample.fit_resample(Train_X, Train_Y)
        all_generated_samples.append(Train_X)
        all_generated_labels.extend(Train_Y)

    elif over_sampling == 'none':
        resu = Counter(Train_Y)
        print('增强后的Train_Y个数', resu)
        return Train_X, Train_Y

    # 确保每个类别的数据量和label1的个数一致
    augmented_train_x = []
    augmented_train_y = []

    for label in np.unique(Train_Y):
        label_data = Train_X[Train_Y == label]
        num_to_generate = class_1_count - len(label_data)
        if num_to_generate > 0:
            label_data_df = pd.DataFrame(label_data)  # 转换为DataFrame
            sampled_data = label_data_df.sample(n=num_to_generate, replace=True, random_state=42).to_numpy()
            augmented_train_x.append(sampled_data)
            augmented_train_y.extend([label] * num_to_generate)

    if augmented_train_x:
        augmented_train_x = np.vstack(augmented_train_x)
        augmented_train_y = np.array(augmented_train_y)
        Train_X = np.vstack([Train_X, augmented_train_x])
        Train_Y = np.concatenate([Train_Y, augmented_train_y])

    return Train_X, Train_Y

def train_set_split_ssv(train_x, train_y, ssv_num):
    def split_dataset(X, Y, num_samples_per_class, random_state=42):
        if random_state is not None:
            np.random.seed(random_state)
        test_indices = []
        train_indices = []
        for label, num_samples in enumerate(num_samples_per_class):
            class_indices = np.where(Y == label+1)[0]
            # 随机选取指定数量的样本索引作为测试集
            test_indices.extend(np.random.choice(class_indices, size=num_samples, replace=False))
            # 剩余的作为训练集
            train_indices.extend([idx for idx in class_indices if idx not in test_indices])
        return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]
    num_samples_per_class = ssv_num * np.ones((9), dtype=int)

    X_train, X_test, Y_train, Y_test = split_dataset(np.array(train_x), np.array(train_y), num_samples_per_class)
    return X_train, Y_train, X_test



def preprocess_for_diagnosis(d_path, over_sampling='none', imbalance_ratio=1, ssv_size = 100):
    print(d_path)
    scaled_df = load_and_preprocess_data(d_path)
    sampled_data, sampled_labels = create_balanced_dataset(scaled_df)

    X = sampled_data.drop(columns=['label'])
    y = sampled_labels
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, y, test_size=0.1, stratify=sampled_data['label'],
                                                        random_state=42)
    Train_X, Train_Y, ssv_set = train_set_split_ssv(Train_X, Train_Y, ssv_size)
    # 检查Train_Y的标签分布
    print("Train_Y标签分布:")
    unique, counts = np.unique(Train_Y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {int(label)}: {count}")

    imbalanced_train_data, imbalanced_train_labels = create_imbalanced_dataset(pd.DataFrame(Train_X), Train_Y, imbalance_ratio)
    if imbalanced_train_data is None or imbalanced_train_labels is None:
        print("No class 1 data or insufficient data to create imbalanced dataset. Skipping imbalance processing.")
        imbalanced_train_data, imbalanced_train_labels = Train_X, Train_Y

    print("数据增强前不平衡中各个类型的个数:")
    unique, counts = np.unique(imbalanced_train_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {int(label)}: {count}")

    generate_dataX, generate_dataY = data_augmentation(pd.DataFrame(imbalanced_train_data), imbalanced_train_labels,
                                                       over_sampling)

    print("数据增强后Train_Y中各个类型的个数:")
    unique, counts = np.unique(generate_dataY, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"Label {int(label)}: {count}")

    Train_Y, Test_Y = generate_dataY, Test_Y
    Train_X = np.asarray(generate_dataX)
    Test_X = np.asarray(Test_X)
    # Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)

    # print("Test_Y中各个类型的个数:")
    # unique, counts = np.unique(np.argmax(Test_Y, axis=1), return_counts=True)
    # for label, count in zip(unique, counts):
    #     print(f"Label {int(label)}: {count}")

    return Train_X, Train_Y, Test_X, Test_Y, ssv_set

def create_alfa_dataset(train=True, ssv = False,  ssv_size=100, imbalance_factor = 1):

    Train_X, Train_Y, Test_X, Test_Y, ssv_set = preprocess_for_diagnosis(
        r'C:\Users\bobobob\Desktop\UAV-vibration_gan-master1\UAV-vibration_gan-master1\vibration_gan-master\tsne_output',
    over_sampling='none', imbalance_ratio=imbalance_factor, ssv_size=ssv_size
    )
    if(train):
        return CWRUdata(Train_X, Train_Y)
    elif(ssv):
        ssv_y = create_dcae_label(CWRUdata(ssv_set, np.zeros((ssv_set.shape[0], 1))))
        return CWRUdata(ssv_set, ssv_y)
    else:
        return CWRUdata(Test_X, Test_Y)