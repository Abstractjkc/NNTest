#pragma once
#include <iostream>
#include <ctime>
using namespace std;
class Matrix
{
	friend Matrix operator*(const Matrix& a, const Matrix& b);
	friend ostream& operator<<(ostream& out, const Matrix& mat);
	friend Matrix operator+(const Matrix& a, const Matrix& b);
	friend Matrix operator-(const Matrix& a, const Matrix& b);
private:
	double** a;		// 存储二维矩阵
public :
	int row, column;
	Matrix(int row, int column);	// 分配内存
	double*& operator[](int ind);	// 重载下标运算符， 方便修改矩阵
	const double*& operator[](int ind)const;
	Matrix& operator=(Matrix&& mat) noexcept;	// 移动赋值
	Matrix operator=(const Matrix& mat);	// 重载赋值
	Matrix(const Matrix& mat);		// 拷贝构造
	Matrix getTrans()const;	// 求矩阵转置
	~Matrix();	// 释放内存
	void randMatrix();		// 随机矩阵中的值
	void fill(double num); //将矩阵中所有元素填入一个值
	void matrixI(); // 构造一个单位矩阵
	void matrixO(); // 构造一个零矩阵
	void forEach(double (*f)(double));
	Matrix mutipleByNumber(double num)const;
	static double Rand(int l, int r);
};

