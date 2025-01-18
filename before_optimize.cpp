#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

using std::vector;

class FigureProcessor {
private:
  vector<vector<unsigned char>> figure;
  vector<vector<unsigned char>> result;
  const size_t size;

public:
  FigureProcessor(size_t size, size_t seed = 0) : size(size) {
    
    // !!! Please do not modify the following code !!!
    // 如果你需要修改内存的数据结构，请不要修改初始化的顺序和逻辑
    // 助教可能会通过指定某个初始化seed 的方式来验证你的代码
    // 如果你修改了初始化的顺序，可能会导致你的代码无法通过测试
    std::random_device rd;
    std::mt19937_64 gen(seed == 0 ? rd() : seed);
    std::uniform_int_distribution<unsigned char> distribution(0, 255);
    // !!! ----------------------------------------- !!!

    // 两个数组的初始化在这里，可以改动，但请注意 gen 的顺序是从上到下从左到右即可。
    for (size_t i = 0; i < size; ++i) {
      figure.push_back(vector<unsigned char>());
      for (size_t j = 0; j < size; ++j) {
        figure[i].push_back(static_cast<unsigned char>(distribution(gen)));
      }
    }

    for (size_t i = 0; i < size; ++i) {
      result.push_back(vector<unsigned char>());
      for (size_t j = 0; j < size; ++j) {
        result[i].push_back(0);
      }
    }
  }

  ~FigureProcessor() = default;

  // Gaussian filter
  // [[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16
  //FIXME: Feel free to optimize this function
  //Hint: You can use SIMD instructions to optimize this function
  void gaussianFilter() {
    // 处理内部区域
    for (size_t i = 1; i < size - 1; ++i) {
      for (size_t j = 1; j < size - 1; ++j) {
        result[i][j] =
            (figure[i - 1][j - 1] + 2 * figure[i - 1][j] +
             figure[i - 1][j + 1] + 2 * figure[i][j - 1] + 4 * figure[i][j] +
             2 * figure[i][j + 1] + figure[i + 1][j - 1] +
             2 * figure[i + 1][j] + figure[i + 1][j + 1]) /
            16;
      }
    }

    for (size_t i = 1; i < size - 1; ++i) {
      result[i][0] =
          (figure[i - 1][0] + 2 * figure[i - 1][0] + figure[i - 1][1] +
           2 * figure[i][0] + 4 * figure[i][0] + 2 * figure[i][1] +
           figure[i + 1][0] + 2 * figure[i + 1][0] + figure[i + 1][1]) /
          16;

      result[i][size - 1] =
          (figure[i - 1][size - 2] + 2 * figure[i - 1][size - 1] +
           figure[i - 1][size - 1] + 2 * figure[i][size - 2] +
           4 * figure[i][size - 1] + 2 * figure[i][size - 1] +
           figure[i + 1][size - 2] + 2 * figure[i + 1][size - 1] +
           figure[i + 1][size - 1]) /
          16;
    }

    for (size_t j = 1; j < size - 1; ++j) {
      result[0][j] =
          (figure[0][j - 1] + 2 * figure[0][j] + figure[0][j + 1] +
           2 * figure[0][j - 1] + 4 * figure[0][j] + 2 * figure[0][j + 1] +
           figure[1][j - 1] + 2 * figure[1][j] + figure[1][j + 1]) /
          16;

      result[size - 1][j] =
          (figure[size - 2][j - 1] + 2 * figure[size - 2][j] +
           figure[size - 2][j + 1] + 2 * figure[size - 1][j - 1] +
           4 * figure[size - 1][j] + 2 * figure[size - 1][j + 1] +
           figure[size - 1][j - 1] + 2 * figure[size - 1][j] +
           figure[size - 1][j + 1]) /
          16;
    }

    // 处理四个角点
    // 左上角
    result[0][0] = (4 * figure[0][0] + 2 * figure[0][1] + 2 * figure[1][0] +
                    figure[1][1]) /
                   9; 

    // 右上角
    result[0][size - 1] = (4 * figure[0][size - 1] + 2 * figure[0][size - 2] +
                           2 * figure[1][size - 1] + figure[1][size - 2]) /
                          9;

    // 左下角
    result[size - 1][0] = (4 * figure[size - 1][0] + 2 * figure[size - 1][1] +
                           2 * figure[size - 2][0] + figure[size - 2][1]) /
                          9;

    // 右下角
    result[size - 1][size - 1] =
        (4 * figure[size - 1][size - 1] + 2 * figure[size - 1][size - 2] +
         2 * figure[size - 2][size - 1] + figure[size - 2][size - 2]) /
        9;
  }

  // Power law transformation
  // FIXME: Feel free to optimize this function
  // Hint: LUT to optimize this function?
  void powerLawTransformation() {
    constexpr float gamma = 0.5f;
    
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        if(figure[i][j] == 0) {
          result[i][j] = 0;
          continue;
        }
        float normalized = (figure[i][j]) / 255.0f;
        result[i][j] = static_cast<unsigned char>(
            255.0f * std::pow(normalized, gamma) + 0.5f); 
      }
    }
  }

  // Run benchmark
  unsigned int calcChecksum() {
    unsigned int sum = 0;
    constexpr size_t mod = 1000000007;
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        sum += result[i][j];
        sum %= mod;
      }
    }
    return sum;
  }
  void runBenchmark() {
    auto start = std::chrono::high_resolution_clock::now();
    gaussianFilter();
    auto middle = std::chrono::high_resolution_clock::now();

    unsigned int sum = calcChecksum();

    auto middle2 = std::chrono::high_resolution_clock::now();
    powerLawTransformation();
    auto end = std::chrono::high_resolution_clock::now();
    
    sum += calcChecksum();
    sum %= 1000000007;
    std::cout << "Checksum: " << sum << "\n";

    auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(middle - start) +
        std::chrono::duration_cast<std::chrono::milliseconds>(end - middle2);
    std::cout << "Benchmark time: " << milliseconds.count() << " ms\n";
  }
};

// Main function
// !!! Please do not modify the main function !!!
int main(int argc, const char **argv) {
  constexpr size_t size = 16384;
  FigureProcessor processor(size, argc > 1 ? std::stoul(argv[1]) : 0);
  processor.runBenchmark();
  return 0;
}
