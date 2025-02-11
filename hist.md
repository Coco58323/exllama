before gemm
after gemm 0.026833295822143555
Test passed successfully!






before gemm
after gemm 0.030245542526245117
Test passed successfully!

before gemm
after gemm 0.0001246929168701172
Test passed successfully!


before gemm
after gemm 0.00012755393981933594
Test passed successfully!



before gemm
after gemm 0.03654003143310547
Test passed successfully!

before gemm
after gemm 0.00010251998901367188
Test passed successfully!
tensor(-6304., device='cuda:0', dtype=torch.float16)


---fp5:
before gemm
after gemm 0.04806780815124512
Test passed successfully!
tensor(-3910., device='cuda:0', dtype=torch.float16)

before gemm
after gemm 0.00012564659118652344
Test passed successfully!
tensor(-3910., device='cuda:0', dtype=torch.float16)

before gemm
after gemm 0.00012373924255371094
Test passed successfully!
tensor(-3910., device='cuda:0', dtype=torch.float16)



int5:
before gemm
after gemm 0.03354620933532715
Test passed successfully!
tensor(-4288., device='cuda:0', dtype=torch.float16)
before gemm
after gemm 0.0005466938018798828
Test passed successfully!
tensor(-4288., device='cuda:0', dtype=torch.float16)
before gemm
after gemm 0.00010251998901367188
Test passed successfully!
tensor(-4288., device='cuda:0', dtype=torch.float16)


fp5:
Round1: 0.0206601619720459
Round2: 5.7697296142578125e-05
Round3: 5.2928924560546875e-05


int5:
Round1: 0.02098536491394043
Round2: 5.340576171875e-05
Round3: 5.1021575927734375e-05



tensor(-10.5156, device='cuda:0', dtype=torch.float16)

tensor(-0.0163, device='cuda:0', dtype=torch.float16)

input: 1*16*4096
weight: 4096*4096
group: 128


1. mnk=16x14848x8192
2. mnk=16x8192x7424
fp5:
23.7768 TFLOPs 0.1637 ms
18.0307 TFLOPs 0.1079 ms

int5:
25.1089 TFLOPs 0.1550 ms
18.0999 TFLOPs 0.1075 ms

fp16:
0.0984ms 0.05ms


cutlass a8w4:
1. mnk=16x14848x8192
2. mnk=16x8192x7424
直接转换：
0.0907059 ms  0.0817361 ms
查表：
1.41715 ms   0.713534 ms

cutlass a8w4:
1. mnk=4096x14848x8192
2. mnk=4096x8192x7424
直接转换：
2.39079 ms  1.2016 ms
查表：
1.41785 ms   0.709342 ms

初步结论：
1. 小 batch 下，和您的实现不太对齐，可能是机器的问题
2. 小 batch 下，和
