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

    void DataMatrix::Reorder(std::vector<size_t> &sorting_order){
      for(size_t j = 0; j < columns; ++j){
          std::vector<float> tmp(rows);
          for(size_t i = 0; i < rows; ++i){
              tmp[i] = data[j][sorting_order[i]];
            }
          std::copy(tmp.begin(), tmp.end(), data[j].begin());

          for(size_t i = 0; i < rows; ++i){
              tmp[i] = sorted_data[j][sorting_order[i]];
            }
          std::copy(tmp.begin(), tmp.end(), sorted_data[j].begin());

          for(size_t i = 0; i < rows; ++i){
              tmp[i] = sorted_grad[j][sorting_order[i]];
            }
          std::copy(tmp.begin(), tmp.end(), sorted_grad[j].begin());
        }

      for(size_t j = 0; j < columns; ++j){
          std::vector<int> tmp(rows);
          for(size_t i = 0; i < rows; ++i){
              tmp[i] = index[j][sorting_order[i]];
            }
          std::copy(tmp.begin(), tmp.end(), index[j].begin());
        }

      std::vector<float> tmp(rows);
      for(size_t i = 0; i < rows; ++i){
          tmp[i] = grad[sorting_order[i]];
        }
      std::copy(tmp.begin(), tmp.end(), grad.begin());

      for(size_t i = 0; i < rows; ++i){
          tmp[i] = y[sorting_order[i]];
        }
      std::copy(tmp.begin(), tmp.end(), y.begin());

      for(size_t i = 0; i < rows; ++i){
          tmp[i] = y_hat[sorting_order[i]];
        }
      std::copy(tmp.begin(), tmp.end(), y_hat.begin());
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
