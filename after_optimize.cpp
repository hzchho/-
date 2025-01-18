#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>
#include <immintrin.h>
#include <cmath>

using std::vector;
    
class FigureProcessor {
private:
  const size_t size;
  unsigned char *figure;
  unsigned char *result;
    
public:
    FigureProcessor(size_t size, size_t seed = 0) : size(size) 
    {
    figure = new unsigned char[size * size];
    result = new unsigned char[size * size];
    // !!! Please do not modify the following code !!!
    // 如果你需要修改内存的数据结构，请不要修改初始化的顺序和逻辑
    // 助教可能会通过指定某个初始化seed 的方式来验证你的代码
    // 如果你修改了初始化的顺序，可能会导致你的代码无法通过测试
    std::random_device rd;
    std::mt19937_64 gen(seed == 0 ? rd() : seed);
    std::uniform_int_distribution<unsigned char> distribution(0, 255);
    // !!! ----------------------------------------- !!!

    // 两个数组的初始化在这里，可以改动，但请注意 gen 的顺序是从上到下从左到右即可。
    #pragma omp for
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                figure[i * size + j] = static_cast<unsigned char>(distribution(gen));
            }
        }

    #pragma omp for
        for (size_t i = 0; i < size; i++) {
            for (size_t j = 0; j < size; j++) {
                result[i * size + j] = 0;
            }
        }

    
  }

  ~FigureProcessor() = default;

  // Gaussian filter
  // [[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16
  //FIXME: Feel free to optimize this function
  //Hint: You can use SIMD instructions to optimize this function
  void gaussianFilter() {
    __m256i f = _mm256_set1_epi16(0x00FF);
    #pragma omp parallel shared(figure, result, f)
    {
        #pragma omp for collapse(2)
        // 内部区域
        for (size_t i = 1; i < size -1; ++i) {
            for (size_t j = 32; j < size - 32; j += 32) {
                // 提取元素
                __m256i left_up = _mm256_loadu_si256((__m256i*)&figure[(i - 1) * size + j - 1]);
                __m256i middle_up = _mm256_loadu_si256((__m256i*)&figure[(i - 1) * size + j]);
                __m256i right_up = _mm256_loadu_si256((__m256i*)&figure[(i - 1) * size + j + 1]);
                __m256i left_middle = _mm256_loadu_si256((__m256i*)&figure[(i * size) + j - 1]);
                __m256i middle_middle = _mm256_loadu_si256((__m256i*)&figure[(i * size) + j]);
                __m256i right_middle = _mm256_loadu_si256((__m256i*)&figure[(i * size) + j + 1]);
                __m256i left_under = _mm256_loadu_si256((__m256i*)&figure[(i + 1) * size + j - 1]);
                __m256i middle_under = _mm256_loadu_si256((__m256i*)&figure[(i + 1) * size + j]);
                __m256i right_under = _mm256_loadu_si256((__m256i*)&figure[(i + 1) * size + j + 1]);
                
                // 每16位的高位部分
                __m256i left_up2 = _mm256_srli_epi16(left_up, 8);
                __m256i middle_up2 = _mm256_srli_epi16(middle_up, 8);
                __m256i right_up2 = _mm256_srli_epi16(right_up, 8);
                __m256i left_middle2 = _mm256_srli_epi16(left_middle, 8);
                __m256i middle_middle2 = _mm256_srli_epi16(middle_middle, 8);
                __m256i right_middle2 = _mm256_srli_epi16(right_middle, 8);
                __m256i left_under2 = _mm256_srli_epi16(left_under, 8);
                __m256i middle_under2 = _mm256_srli_epi16(middle_under, 8);
                __m256i right_under2 = _mm256_srli_epi16(right_under, 8);
                
                // 每16位的低位部分
                left_up = _mm256_and_si256(left_up, f);
                middle_up = _mm256_and_si256(middle_up, f);
                right_up = _mm256_and_si256(right_up, f);
                left_middle = _mm256_and_si256(left_middle, f);
                middle_middle = _mm256_and_si256(middle_middle, f);
                right_middle = _mm256_and_si256(right_middle, f);
                left_under = _mm256_and_si256(left_under, f);
                middle_under = _mm256_and_si256(middle_under, f);
                right_under = _mm256_and_si256(right_under, f);

                // 由于其中有四个进行×2，一个进行×4
                // 可以先将中心点×2后再与四方相加，再进行×2，再加上四个角，仅需进行两次乘法
                middle_middle = _mm256_slli_epi16(middle_middle, 1);
                middle_middle = _mm256_adds_epu16(middle_middle, middle_up);
                middle_middle = _mm256_adds_epu16(middle_middle, left_middle);
                middle_middle = _mm256_adds_epu16(middle_middle, right_middle);
                middle_middle = _mm256_adds_epu16(middle_middle, middle_under);
                middle_middle = _mm256_slli_epi16(middle_middle, 1);

                middle_middle = _mm256_adds_epu16(middle_middle, left_up);
                middle_middle = _mm256_adds_epu16(middle_middle, right_up);
                middle_middle = _mm256_adds_epu16(middle_middle, left_under);
                middle_middle = _mm256_adds_epu16(middle_middle, right_under);
                
                // 原理同上
                middle_middle2 = _mm256_slli_epi16(middle_middle2, 1);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, middle_up2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, left_middle2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, right_middle2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, middle_under2);
                middle_middle2 = _mm256_slli_epi16(middle_middle2, 1);

                middle_middle2 = _mm256_adds_epu16(middle_middle2, left_up2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, right_up2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, left_under2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, right_under2);
                
                // 按16位进行右移4位(即除以16)
                middle_middle = _mm256_srli_epi16(middle_middle, 4);
                middle_middle2 = _mm256_srli_epi16(middle_middle2, 4);

                // 两个结果都仅保留低8位，并将原先的进行左移8位作为高位进行组合
                middle_middle = _mm256_and_si256(middle_middle, f);
                middle_middle2 = _mm256_and_si256(middle_middle2, f);
                middle_middle2 = _mm256_slli_epi16(middle_middle2, 8);
                middle_middle = _mm256_add_epi16(middle_middle, middle_middle2);

                // 将结果回传内存
                _mm256_storeu_si256((__m256i*)&result[i * size + j], middle_middle);

            }   
        }

        #pragma omp for
        // 左内边
            for (size_t i = 1; i < size -1; ++i) {
                size_t j = 1;
                // 提取元素
                __m256i left_up = _mm256_loadu_si256((__m256i*)&figure[(i - 1) * size + j - 1]);
                __m256i middle_up = _mm256_loadu_si256((__m256i*)&figure[(i - 1) * size + j]);
                __m256i right_up = _mm256_loadu_si256((__m256i*)&figure[(i - 1) * size + j + 1]);
                __m256i left_middle = _mm256_loadu_si256((__m256i*)&figure[(i * size) + j - 1]);
                __m256i middle_middle = _mm256_loadu_si256((__m256i*)&figure[(i * size) + j]);
                __m256i right_middle = _mm256_loadu_si256((__m256i*)&figure[(i * size) + j + 1]);
                __m256i left_under = _mm256_loadu_si256((__m256i*)&figure[(i + 1) * size + j - 1]);
                __m256i middle_under = _mm256_loadu_si256((__m256i*)&figure[(i + 1) * size + j]);
                __m256i right_under = _mm256_loadu_si256((__m256i*)&figure[(i + 1) * size + j + 1]);
                
                // 每16位的高位部分
                __m256i left_up2 = _mm256_srli_epi16(left_up, 8);
                __m256i middle_up2 = _mm256_srli_epi16(middle_up, 8);
                __m256i right_up2 = _mm256_srli_epi16(right_up, 8);
                __m256i left_middle2 = _mm256_srli_epi16(left_middle, 8);
                __m256i middle_middle2 = _mm256_srli_epi16(middle_middle, 8);
                __m256i right_middle2 = _mm256_srli_epi16(right_middle, 8);
                __m256i left_under2 = _mm256_srli_epi16(left_under, 8);
                __m256i middle_under2 = _mm256_srli_epi16(middle_under, 8);
                __m256i right_under2 = _mm256_srli_epi16(right_under, 8);
                
                // 每16位的低位部分
                left_up = _mm256_and_si256(left_up, f);
                middle_up = _mm256_and_si256(middle_up, f);
                right_up = _mm256_and_si256(right_up, f);
                left_middle = _mm256_and_si256(left_middle, f);
                middle_middle = _mm256_and_si256(middle_middle, f);
                right_middle = _mm256_and_si256(right_middle, f);
                left_under = _mm256_and_si256(left_under, f);
                middle_under = _mm256_and_si256(middle_under, f);
                right_under = _mm256_and_si256(right_under, f);

                // 由于其中有四个进行×2，一个进行×4
                // 可以先将中心点×2后再与四方相加，再进行×2，再加上四个角，仅需进行两次乘法
                middle_middle = _mm256_slli_epi16(middle_middle, 1);
                middle_middle = _mm256_adds_epu16(middle_middle, middle_up);
                middle_middle = _mm256_adds_epu16(middle_middle, left_middle);
                middle_middle = _mm256_adds_epu16(middle_middle, right_middle);
                middle_middle = _mm256_adds_epu16(middle_middle, middle_under);
                middle_middle = _mm256_slli_epi16(middle_middle, 1);

                middle_middle = _mm256_adds_epu16(middle_middle, left_up);
                middle_middle = _mm256_adds_epu16(middle_middle, right_up);
                middle_middle = _mm256_adds_epu16(middle_middle, left_under);
                middle_middle = _mm256_adds_epu16(middle_middle, right_under);
                
                // 原理同上
                middle_middle2 = _mm256_slli_epi16(middle_middle2, 1);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, middle_up2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, left_middle2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, right_middle2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, middle_under2);
                middle_middle2 = _mm256_slli_epi16(middle_middle2, 1);

                middle_middle2 = _mm256_adds_epu16(middle_middle2, left_up2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, right_up2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, left_under2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, right_under2);
                
                // 按16位进行右移4位(即除以16)
                middle_middle = _mm256_srli_epi16(middle_middle, 4);
                middle_middle2 = _mm256_srli_epi16(middle_middle2, 4);

                // 两个结果都仅保留低8位，并将原先的进行左移8位作为高位进行组合
                middle_middle = _mm256_and_si256(middle_middle, f);
                middle_middle2 = _mm256_and_si256(middle_middle2, f);
                middle_middle2 = _mm256_slli_epi16(middle_middle2, 8);
                middle_middle = _mm256_add_epi16(middle_middle, middle_middle2);

                // 将结果回传内存
                _mm256_storeu_si256((__m256i*)&result[i * size + j], middle_middle);

            }   

        #pragma omp for
        // 右内边
            for (size_t i = 1; i < size -1; ++i) {
                size_t j = size - 33;
                // 提取元素
                __m256i left_up = _mm256_loadu_si256((__m256i*)&figure[(i - 1) * size + j - 1]);
                __m256i middle_up = _mm256_loadu_si256((__m256i*)&figure[(i - 1) * size + j]);
                __m256i right_up = _mm256_loadu_si256((__m256i*)&figure[(i - 1) * size + j + 1]);
                __m256i left_middle = _mm256_loadu_si256((__m256i*)&figure[(i * size) + j - 1]);
                __m256i middle_middle = _mm256_loadu_si256((__m256i*)&figure[(i * size) + j]);
                __m256i right_middle = _mm256_loadu_si256((__m256i*)&figure[(i * size) + j + 1]);
                __m256i left_under = _mm256_loadu_si256((__m256i*)&figure[(i + 1) * size + j - 1]);
                __m256i middle_under = _mm256_loadu_si256((__m256i*)&figure[(i + 1) * size + j]);
                __m256i right_under = _mm256_loadu_si256((__m256i*)&figure[(i + 1) * size + j + 1]);
                
                // 每16位的高位部分
                __m256i left_up2 = _mm256_srli_epi16(left_up, 8);
                __m256i middle_up2 = _mm256_srli_epi16(middle_up, 8);
                __m256i right_up2 = _mm256_srli_epi16(right_up, 8);
                __m256i left_middle2 = _mm256_srli_epi16(left_middle, 8);
                __m256i middle_middle2 = _mm256_srli_epi16(middle_middle, 8);
                __m256i right_middle2 = _mm256_srli_epi16(right_middle, 8);
                __m256i left_under2 = _mm256_srli_epi16(left_under, 8);
                __m256i middle_under2 = _mm256_srli_epi16(middle_under, 8);
                __m256i right_under2 = _mm256_srli_epi16(right_under, 8);
                
                // 每16位的低位部分
                left_up = _mm256_and_si256(left_up, f);
                middle_up = _mm256_and_si256(middle_up, f);
                right_up = _mm256_and_si256(right_up, f);
                left_middle = _mm256_and_si256(left_middle, f);
                middle_middle = _mm256_and_si256(middle_middle, f);
                right_middle = _mm256_and_si256(right_middle, f);
                left_under = _mm256_and_si256(left_under, f);
                middle_under = _mm256_and_si256(middle_under, f);
                right_under = _mm256_and_si256(right_under, f);

                // 由于其中有四个进行×2，一个进行×4
                // 可以先将中心点×2后再与四方相加，再进行×2，再加上四个角，仅需进行两次乘法
                middle_middle = _mm256_slli_epi16(middle_middle, 1);
                middle_middle = _mm256_adds_epu16(middle_middle, middle_up);
                middle_middle = _mm256_adds_epu16(middle_middle, left_middle);
                middle_middle = _mm256_adds_epu16(middle_middle, right_middle);
                middle_middle = _mm256_adds_epu16(middle_middle, middle_under);
                middle_middle = _mm256_slli_epi16(middle_middle, 1);

                middle_middle = _mm256_adds_epu16(middle_middle, left_up);
                middle_middle = _mm256_adds_epu16(middle_middle, right_up);
                middle_middle = _mm256_adds_epu16(middle_middle, left_under);
                middle_middle = _mm256_adds_epu16(middle_middle, right_under);
                
                // 原理同上
                middle_middle2 = _mm256_slli_epi16(middle_middle2, 1);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, middle_up2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, left_middle2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, right_middle2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, middle_under2);
                middle_middle2 = _mm256_slli_epi16(middle_middle2, 1);

                middle_middle2 = _mm256_adds_epu16(middle_middle2, left_up2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, right_up2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, left_under2);
                middle_middle2 = _mm256_adds_epu16(middle_middle2, right_under2);
                
                // 按16位进行右移4位(即除以16)
                middle_middle = _mm256_srli_epi16(middle_middle, 4);
                middle_middle2 = _mm256_srli_epi16(middle_middle2, 4);

                // 两个结果都仅保留低8位，并将原先的进行左移8位作为高位进行组合
                middle_middle = _mm256_and_si256(middle_middle, f);
                middle_middle2 = _mm256_and_si256(middle_middle2, f);
                middle_middle2 = _mm256_slli_epi16(middle_middle2, 8);
                middle_middle = _mm256_add_epi16(middle_middle, middle_middle2);

                // 将结果回传内存
                _mm256_storeu_si256((__m256i*)&result[i * size + j], middle_middle);

            }   

        #pragma omp for
        for (size_t i = 1; i < size - 1; ++i) {
            result[i * size] =
                (figure[(i - 1) * size] + 2 * figure[(i - 1) * size] +
                figure[(i - 1) * size + 1] + 2 * figure[i * size] + 4 * figure[i * size] +
                2 * figure[i * size + 1] + figure[(i + 1) * size] +
                2 * figure[(i + 1) * size] + figure[(i + 1) * size + 1]) /
                16;

            result[i * size + size - 1] =
                (figure[i * size - 2] + 2 * figure[i * size - 1] +
                figure[i * size - 1] + 2 * figure[i * size + size - 2] + 4 * figure[i * size + size - 1] +
                2 * figure[i * size + size - 1] + figure[(i + 1) * size + size - 2] +
                2 * figure[(i + 1) * size + size - 1] + figure[(i + 1) * size + size - 1]) /
                16;
        }

        #pragma omp for
        for (size_t j = 1; j < size - 1; ++j) {
            result[j] =
                (figure[j - 1] + 2 * figure[j] + figure[j + 1] +
                2 * figure[j - 1] + 4 * figure[j] + 2 * figure[j + 1] +
                figure[size + j - 1] + 2 * figure[size + j] + figure[size + j + 1]) /
                16;

            result[(size - 1) * size + j] =
                (figure[(size - 2) * size + j - 1] + 2 * figure[(size - 2) * size + j] +
                figure[(size - 2) * size + j + 1] + 2 * figure[(size - 1) * size + j - 1] +
                4 * figure[(size - 1) * size + j] + 2 * figure[(size - 1) * size + j + 1] +
                figure[(size - 1) * size + j - 1] + 2 * figure[(size - 1) * size + j] +
                figure[(size - 1) * size+ j + 1]) /
                16;
        }

    }
     
    // 处理四个角点
    // 左上角
    result[0] = (4 * figure[0] + 2 * figure[1] + 2 * figure[size] +
                    figure[size + 1]) /
                   9; 

    // 右上角
    result[size - 1] = (4 * figure[size - 1] + 2 * figure[size - 2] +
                           2 * figure[2 * size - 1] + figure[2 * size - 2]) /
                          9;

    // 左下角
    result[(size - 1) * size] = (4 * figure[(size - 1) * size] + 2 * figure[(size - 1) * size + 1] +
                           2 * figure[(size - 2) * size] + figure[(size - 2) * size + 1]) /
                          9;

    // 右下角
    result[size * size - 1] =
        (4 * figure[size * size - 1] + 2 * figure[size * size - 2] +
         2 * figure[(size - 1) * size - 1] + figure[(size - 1) * size - 2]) /
        9;
  }

  // Power law transformation
  // FIXME: Feel free to optimize this function
  // Hint: LUT to optimize this function?
  void powerLawTransformation() {
    unsigned char LUT[256];
    constexpr float gamma = 0.5f;
    const float delta = std::pow(255.0f, gamma);
    for (size_t i = 0; i < 256; ++i) {
        LUT[i] = static_cast<unsigned char>(255.0f * std::pow(i, gamma) / delta + 0.5f);
    }
    
    #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; j += 4) {
                result[i * size + j] = LUT[figure[i * size + j]]; 
                result[i * size + j + 1] = LUT[figure[i * size + j + 1]];
                result[i * size + j + 2] = LUT[figure[i * size + j + 2]];
                result[i * size + j + 3] = LUT[figure[i * size + j + 3]];
            }
        }
    
  }
  // Run benchmark
  unsigned int calcChecksum() {
    unsigned int sum = 0;
    constexpr size_t mod = 1000000007;
    #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < size; ++i) {
            for (size_t j = 0; j < size; ++j) {
                sum += result[i * size + j];
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

    delete []result;
    delete []figure;
  }
};

// Main function
// !!! Please do not modify the main function !!!
int main(int argc, const char **argv) {
  omp_set_num_threads(4);
  printf ( "Number of threads = %d\n", omp_get_max_threads ( ) );
  constexpr size_t size = 16384;
  FigureProcessor processor(size, argc > 1 ? std::stoul(argv[1]) : 0);
  processor.runBenchmark();
  return 0;
}