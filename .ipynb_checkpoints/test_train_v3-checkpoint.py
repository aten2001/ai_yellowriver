import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense
# 加载plt文件（2D数据）

input_directory = 'path/to/plt/files'  # 输入的plt文件所在目录
output_directory = 'path/to/save/npy/files'  # 保存npy文件的目录


def plt_to_npy(input_directory,output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith('.plt'):
            plt_path = os.path.join(input_directory, filename)
            output_filename = os.path.splitext(filename)[0] + '.npy'
            output_path = os.path.join(output_directory, output_filename)
            os.makedirs(output_path, exist_ok=True)
            # 读取plt文件数据
            data = np.loadtxt(plt_path)

            # 将数据保存为npy文件
            np.save(output_path, data)
    return output_path

plt_data= plt_to_npy('path/to/plt/files','path/to/save/npy/files')
original_height, original_width = plt_data.shape

# 定义插值后的高分辨率图像的尺寸
target_height = 2 * original_height
target_width = 2 * original_width

# 创建插值的网格
x = np.linspace(0, original_height-1, original_height)
y = np.linspace(0, original_width-1, original_width)
grid_x, grid_y = np.meshgrid(x, y, indexing='ij')

# 创建插值的坐标点
new_x = np.linspace(0, original_height-1, target_height)
new_y = np.linspace(0, original_width-1, target_width)
new_grid_x, new_grid_y = np.meshgrid(new_x, new_y, indexing='ij')

# 进行插值
interp_data = interpolate.griddata((grid_x.flatten(), grid_y.flatten()), plt_data.flatten(),
                                   (new_grid_x, new_grid_y), method='linear')

# 可以根据需求对插值后的数据进行进一步处理和分析

# 选择合适的时间步骤大小
time_steps = 150
height = 50
width = 50

# 数据预处理和准备
input_sequences = []
output_sequences = []
sequence_length = interp_data.shape[0] - time_steps   # 根据数据长度和时序长度计算
for i in range(sequence_length):
    input_sequences.append(interp_data[i:i+time_steps-1, :height, :width, np.newaxis])
    output_sequences.append(interp_data[i+time_steps-1, :height, :width, np.newaxis])

# 转换为数组并归一化处理（可根据具体需求调整）
input_data = np.array(input_sequences)
output_data = np.array(output_sequences)
input_data = input_data / np.max(input_data)  # 归一化处理
output_data = output_data / np.max(output_data)  # 归一化处理

# 划分训练集和测试集
train_ratio = 0.8
train_size = int(train_ratio * input_data.shape[0])
x_train, x_test = input_data[:train_size], input_data[train_size:]
y_train, y_test = output_data[:train_size], output_data[train_size:]

# 构建CNN模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(time_steps-1, height, width, 1)),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# 随机选择一个输入样本进行预测
random_index = np.random.randint(0, x_test.shape[0])
random_input = x_test[random_index:random_index+1]

# 预测第150个时间步
predicted_output = model.predict(random_input)

# 预测下一个时刻
next_input = np.concatenate((random_input[:, 1:, :, :, :], predicted_output), axis=1)
next_predicted_output = model.predict(next_input)