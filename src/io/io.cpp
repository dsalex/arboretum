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

    void DataMatrix::Reorder(std::vector<size_t> &rowIndex2Node, std::vector<int> &node_offset){
      printf("====== offset \n");
      for(size_t i = 0; i < node_offset.size(); ++i){
          printf("node %d offset %d \n", i, node_offset[i]);
        }

      for(size_t j = 0; j < columns; ++j){
          std::vector<int> counter(node_offset);
          std::vector<int> tmp_int(rows);
          std::vector<float> tmp_value(rows);
          std::vector<float> tmp_grad(rows);
          size_t node;
          size_t row_index;

          printf("resort for %d ======== \n", j);

          for(size_t i = 0; i < rows; ++i){
              row_index = index[j][i];
              node = rowIndex2Node[row_index];

              printf("initial row_index %d node %d value %f grad %f \n",
                     row_index, node, data[j][row_index], grad[row_index]);

              tmp_int[counter[node]] = counter[node];
              tmp_value[counter[node]] = sorted_data[j][i];
              tmp_grad[counter[node]] = sorted_grad[j][i];
              counter[node]++;
            }
          std::copy(tmp_int.begin(), tmp_int.end(), index[j].begin());
          std::copy(tmp_grad.begin(), tmp_grad.end(), sorted_grad[j].begin());
          std::copy(tmp_value.begin(), tmp_value.end(), sorted_data[j].begin());


          for(size_t i = 0; i < rows; ++i){
//              node = rowIndex2Node[i];
              printf("i %d value %f grad %f \n", i, data[j][index[j][i]], grad[index[j][i]]);
            }
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
