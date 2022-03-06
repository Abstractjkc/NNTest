#include "Matrix.h"

Matrix operator*(const Matrix& a, const Matrix& b)
{
    Matrix c(a.row, b.column);
     for (int i = 0; i < c.row; ++i) {
        for (int j = 0; j < c.column; ++j) {
            for (int k = 0; k < a.column; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

ostream& operator<<(ostream& out, const Matrix& mat)
{
    // TODO: 在此处插入 return 语句
    for (int i = 0; i < mat.row; ++i) {
        for (int j = 0; j < mat.column; ++j) {
            out << mat[i][j] << " ";
        }
        out << "\n";
    }
    return out;
}

Matrix operator+(const Matrix& a, const Matrix& b)
{
    // TODO: 在此处插入 return 语句
    Matrix c(a.row, a.column);
     for (int i = 0; i < c.row; ++i) {
        for (int j = 0; j < c.column; ++j) {
            c[i][j] = a[i][j] + b[i % b.row][j];
        }
     }
    return c;
}

Matrix operator-(const Matrix& a, const Matrix& b)
{
    Matrix c(a.row, a.column);
    for (int i = 0; i < c.row; ++i) {
        for (int j = 0; j < c.column; ++j) {
            c[i][j] = a[i][j] - b[i % b.row][j];
        }
    }
    return c;
}

double*& Matrix::operator[](int ind)
{
    // TODO: 在此处插入 return 语句
    return a[ind];
}

const double*& Matrix::operator[](int ind) const
{
    // TODO: 在此处插入 return 语句
    return (const double *&)a[ind];
}

Matrix& Matrix::operator=(Matrix&& mat) noexcept
{
    // TODO: 在此处插入 return 语句
    if (&mat == this) return *this;
    if (a != nullptr) {
        for (int i = 0; i < row; ++i) delete[] a[i];
        delete[] a;
    }
    row = mat.row;
    column = mat.column;
    a = mat.a;
    mat.a = nullptr;
    return *this;
}

Matrix Matrix::operator=(const Matrix& mat)
{
    if (a != nullptr) {
        for (int i = 0; i < row; ++i) delete[] a[i];
        delete a;
    }
    row = mat.row, column = mat.column;
    a = new double* [row];
    for (int i = 0; i < row; ++i) a[i] = new double[column];
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            a[i][j] = mat[i][j];
        }
    }
    return *this;
}

Matrix::Matrix(int row, int column):row(row), column(column)
{
    a = new double* [row];
    for (int i = 0; i < row; ++i) {
        a[i] = new double[column];
        for (int j = 0; j < column; ++j) a[i][j] = 0;
    }
}

Matrix::Matrix(const Matrix& mat):row(mat.row), column(mat.column)
{
    a = new double* [row];
    for (int i = 0; i < row; ++i) {
        a[i] = new double[column];
    }
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            a[i][j] = mat[i][j];
        }
    }
}

Matrix Matrix::getTrans() const 
{
    Matrix c = Matrix(column, row);
    for (int i = 0; i < c.row; ++i) {
        for (int j = 0; j < c.column; ++j) {
            c[i][j] = a[j][i];
        }
    }
    return c;
}

Matrix::~Matrix()
{
    if (a == nullptr) return;
    for (int i = 0; i < row; ++i) {
        delete[] a[i];
    }
    delete[] a;
}

void Matrix::randMatrix()
{
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            a[i][j] = Rand(-2, 2);
        }
    }
}

void Matrix::fill(double num)
{
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            a[i][j] = num;
        }
    }
}

void Matrix::matrixI()
{
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            a[i][j] = 0;
        }
        a[i][i] = 1;
    }
}

void Matrix::matrixO()
{
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            a[i][j] = 0;
        }
    }
}

void Matrix::forEach(double(*f)(double))
{
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            a[i][j] = f(a[i][j]);
        }
    }
}

Matrix Matrix::mutipleByNumber(double num) const
{
    Matrix c(row, column);
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            c[i][j] = num * a[i][j];
        }
    }
    return c;
}

double Matrix::Rand(int l, int r)
{
    return (0.0 + r - l + 1.0) * rand() / RAND_MAX + l;
}
