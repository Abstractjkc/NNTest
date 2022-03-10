// NNTest.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
// 全连接网络判断坐标点所在的象限
/**
    考虑到全连接过程并不复杂，直接选择用面向过程做该实例
*/
#include <iostream>
#include <algorithm>
#include <cmath>
#include "Matrix.h"
using namespace std;
void getProbs(Matrix &Y);       // 获得输出对应标签的概率
double getLoss(const Matrix& probs, const Matrix& label);   // 获取loss值
Matrix affine_forward(const Matrix& X, const Matrix& W, const Matrix& b);   // 前向传播函数

// 反向传播函数
void affine_backward(const Matrix& out, const Matrix &X, const Matrix &W, const Matrix &b, Matrix &dX, Matrix &dW, Matrix &db);
// relu 激活函数
double relu(double num) {
    return num <= 0 ? 0 : num;
}

int main()
{
    // ==========数据初始化开始==========
    srand(time(NULL));
    double loss = 0, reg = 0.001, epsilon = 0.001;
    int input_dim = 2, num_classes = 4, hidden_dim = 50, num_test = 4;
    /*
        矩阵规模的初始化
        以及输入数据的初始化
        容易看到 选了四个点 (1, 1), (-1, 1), (-1, -1), (1, -1) 作为输出数据
    */
    Matrix input(num_test, input_dim);
    input[0][0] = 1, input[0][1] = 1;
    input[1][0] = -1, input[1][1] = 1;
    input[2][0] = -1, input[2][1] = -1;
    input[3][0] = 1, input[3][1] = -1;
    Matrix label(num_test, 1);
    label[0][0] = 1, label[1][0] = 2, label[2][0] = 3, label[3][0] = 4; // 四个点对应的正确标签
    Matrix W1(input_dim, hidden_dim), W2(hidden_dim, num_classes), b1(1, hidden_dim);
    Matrix b2(1, num_classes), H1(num_test, hidden_dim), Relu_Cache(num_test, hidden_dim);
    Matrix dW1(input_dim, hidden_dim), dW2(hidden_dim, num_classes), db1(1, hidden_dim);
    Matrix db2(1, num_classes), Y(num_test, num_classes), probs(num_test, num_classes);
    Matrix dH(num_test, hidden_dim), dInput(num_test, input_dim);
    W1.randMatrix(), W2.randMatrix();
    b1.fill(0), b2.fill(0);
    // ======数据初始化结束=========
    // =========训练==========
    int T = 10000;  // 迭代训练次数
    while (T--) {
        // 前向传播 激活 前向传播
        H1 = affine_forward(input, W1, b1); // 第一次前向传播
        H1.forEach(relu);   //激活
        Relu_Cache = H1;
        Y = affine_forward(H1, W2, b2); // 第二次前向传播
        probs = Y;
        getProbs(probs);    // 获得输出对应标签的概率
        loss = getLoss(probs, label);   // 获得loss值

        //反向传播 基本同上
        for (int i = 0; i < label.row; ++i) {
            probs[i][(int)label[i][0] - 1] -= 1;
        }
        probs = probs.mutipleByNumber(1.0 / num_test);
        affine_backward(probs, H1, W2, b2, dH, dW2, db2);
        for (int i = 0; i < Relu_Cache.row; ++i) {
            for (int j = 0; j < Relu_Cache.column; ++j) {
                if (Relu_Cache[i][j] <= 0) dH[i][j] = 0;
            }
        }
        affine_backward(dH, input, W1, b1, dInput, dW1, db1);

        // 参数的更新
        dW1 = dW1 + W1.mutipleByNumber(reg);
        dW2 = dW2 + W2.mutipleByNumber(reg);
        W1 = W1 - dW1.mutipleByNumber(epsilon);
        W2 = W2 - dW2.mutipleByNumber(epsilon);
        b1 = b1 - db1.mutipleByNumber(epsilon);
        b2 = b2 - db2.mutipleByNumber(epsilon);
    }
    cout  << loss / num_test<< "\n";

    // 测试数据
    Matrix test_data(4, 2);
    test_data[0][0] = -1, test_data[0][1] = 2;
    test_data[1][0] = 1, test_data[1][1] = 2;
    test_data[2][0] = 2, test_data[2][1] = -3;
    test_data[3][0] = -2, test_data[3][1] = -5;
    H1 = affine_forward(test_data, W1, b1);
    H1.forEach(relu);
    Y = affine_forward(H1, W2, b2);
    probs = Y;
    getProbs(probs);
    double maxx = 0;
    int ans;
    for (int i = 0; i < 4; ++i) {
        maxx = 0;
        ans = 0;
        for (int j = 0; j < 4; ++j) {
            if (probs[i][j] > maxx) {
                maxx = probs[i][j];
                ans = j + 1;
            }
        }
        cout << "(" << test_data[i][0] << "," << test_data[i][1] << ")"
            << " 在第 " << ans << " 象限\n";
    }
}

void getProbs(Matrix& Y)
{
    double maxx = -1e18, sum = 0;
    for (int i = 0; i < Y.row; ++i) {
        maxx = -1e18;
        for (int j = 0; j < Y.column; ++j) {
            maxx = max(maxx, Y[i][j]);
        }
        sum = 0;
        for (int j = 0; j < Y.column; ++j) {
            Y[i][j] -= maxx;
            Y[i][j] = exp(Y[i][j]);
            sum += Y[i][j];
        }
        for (int j = 0; j < Y.column; ++j) {
            Y[i][j] /= sum;
        }
    }
}

double getLoss(const Matrix& probs, const Matrix& label)
{
    double loss = 0.0;
    for (int i = 0; i < probs.row; ++i) {
        loss += -log(probs[i][(int)label[i][0] - 1]);
    }
    loss /= probs.row;
    return loss;
}

Matrix affine_forward(const Matrix& X, const Matrix& W, const Matrix& b)
{
    Matrix Y(X.row, W.column);
    Y = X * W + b;
    return Y;
}

void affine_backward(const Matrix& out, const Matrix& X, const Matrix& W, const Matrix& b, Matrix& dX, Matrix& dW, Matrix& db)
{
    dW = X.getTrans() * out;
    dX = out * W.getTrans();
    db.matrixO();
    for (int i = 0; i < out.column; ++i) {
        for (int j = 0; j < out.row; ++j) {
            db[0][i] += out[j][i];
        }
    }
}
