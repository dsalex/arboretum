#include <vector>
#include <algorithm>
#include <functional>
#include "io.h"

namespace arboretum {
  namespace io {
    using namespace std;

    DataMatrix::DataMatrix(int rows, int columns) : rows(rows), columns(columns)
    {
      _init = false;
      data.resize(columns);
      sorted_data.resize(columns);
      index.resize(columns);
      sorted_grad.resize(columns);
      grad.resize(rows);
      for(int i = 0; i < columns; ++i){
            data[i].resize(rows);
      }
    }

    void DataMatrix::Reorder(std::vector<size_t> &rowIndex2Node, std::vector<size_t> &aligned_row2node){
      std::vector<float> tmp_grad(rows);
      for(size_t i = 0; i < rows; ++i){
          tmp_grad[aligned_row2node[i]] = grad[i];
        }
      std::copy(tmp_grad.begin(), tmp_grad.end(), grad.begin());
      for(size_t i = 0; i < rows; ++i){
          tmp_grad[aligned_row2node[i]] = y[i];
        }
      std::copy(tmp_grad.begin(), tmp_grad.end(), y.begin());
      for(size_t i = 0; i < rows; ++i){
          tmp_grad[aligned_row2node[i]] = y_hat[i];
        }
      std::copy(tmp_grad.begin(), tmp_grad.end(), y_hat.begin());
      for(size_t j = 0; j < columns; ++j){
          std::vector<int> tmp_index(rows);
          std::vector<float> tmp_data(rows);

          for(size_t i = 0; i < rows; ++i){
              tmp_index[i] = aligned_row2node[index[j][i]];
              tmp_data[aligned_row2node[i]] = data[j][i];
            }
          std::copy(tmp_index.begin(), tmp_index.end(), index[j].begin());
          std::copy(tmp_data.begin(), tmp_data.end(), data[j].begin());
        }
    }

    void DataMatrix::Init(const float initial_y, std::function<float const(const float, const float)> func){
      if(!_init){
          _gradFunc = func;
          y.resize(y_hat.size(), initial_y);
          for(size_t i = 0; i < data.size(); ++i){
            index[i] = SortedIndex(i);
            std::vector<float> tmp(data[i].size());
            for(size_t j = 0; j < data[i].size(); ++j){
                tmp[j] = data[i][index[i][j]];
              }
            sorted_data[i] = tmp;
          }
          _init = true;
        }
    }

    std::vector<int> DataMatrix::SortedIndex(int column){
        std::vector<float>& v = data[column];
        size_t size = v.size();
        std::vector<int> idx(size);
        for(size_t i = 0; i < size; i ++)
          idx[i] = i;

        sort(idx.begin(), idx.end(),
             [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

        return idx;
    }

    void DataMatrix::UpdateGrad(){
      for(size_t i = 0; i < rows; ++i){
          grad[i] = _gradFunc(y[i], y_hat[i]);
        }
      for(size_t i = 0; i < columns; ++i){
          std::vector<float> tmp(data[i].size());
          for(size_t j = 0; j < data[i].size(); ++j){
              tmp[j] = grad[index[i][j]];
            }
          sorted_grad[i] = tmp;
        }

    }
  }
}
