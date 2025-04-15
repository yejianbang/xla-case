import tensorflow as tf
import time

# 全局启用 XLA JIT 编译
tf.config.optimizer.set_jit(True)

# 定义一个函数并使用 XLA 编译
@tf.function(jit_compile=True)
def matmul(a, b):
    return tf.matmul(a, b)

# 创建两个随机矩阵
a = tf.random.uniform([100000, 10000])
b = tf.random.uniform([10000, 100000])

# 执行矩阵乘法
result = matmul(a, b)


start_time = time.time()
for i in range(10) :
    result = matmul(a, b)

end_time = time.time()
exe_time = end_time - start_time
print("time:", exe_time)
print("Result:", result)
