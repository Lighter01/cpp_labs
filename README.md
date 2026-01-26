# Отчет: SIMD vs Scalar + ThreadPool (RGBA alpha blending и histogram)

Работу выполнил: Тонка Петр

## 1. Введение
SIMD и многопоточность дают ускорение разными способами. SIMD повышает плотность вычислений внутри одного потока за счет векторных инструкций (AVX2/FMA), а ThreadPool масштабирует нагрузку по ядрам. В отчете объединены результаты двух лабораторных: SIMD-реализации (ЛР2) и пул потоков с data-parallel исполнением (ЛР3) для задач альфа-смешивания RGBA и построения 8-битной гистограммы яркости.

## 2. Цель и задачи
**Цель:** ускорить обработку изображений с помощью SIMD и пула потоков, сравнить варианты и оценить накладные расходы.

**Задачи:**
- Реализовать scalar версии alpha blending и histogram.
- Реализовать SIMD версии (AVX2/FMA) этих задач.
- Разработать ThreadPool и policy-обертки (SeqExec/ParExec).
- Интегрировать ThreadPool в pipeline alpha blending.
- Провести измерения, построить графики и оценить speedup/efficiency.
- Проанализировать overhead распараллеливания.

## 3. Краткая теория
### 3.1 SIMD
SIMD обрабатывает несколько элементов одной инструкцией. В AVX2 256-битный регистр содержит 8 float (или 8xint32). Ускорение возможно при линейной памяти, минимуме ветвлений и аккуратном обращении с выравниванием. FMA сокращает число инструкций вида `a*b + c`.

### 3.2 Alpha blending
Базовая формула:
`out_rgb = fg_rgb * a + bg_rgb * (1 - a)`, где `a` - альфа переднего слоя.
Корректное смешивание выполняется в linear RGB, поэтому в pipeline есть:
1) sRGB -> linear (через LUT),
2) premultiply,
3) blending kernel,
4) revert premultiply,
5) linear -> sRGB (через LUT).

### 3.3 Гистограмма яркости
Нужно подсчитать количества пикселей в bins 0..255. Прямой SIMD scatter для `hist[vec]++` сложен, поэтому используется компромисс: частные локальные гистограммы и последующая редукция.

### 3.4 ThreadPool
ThreadPool реализует модель producer/consumer: очередь задач, рабочие потоки, `condition_variable_any` и остановка через `std::stop_token`. Data-parallel достигается разбиением диапазона пикселей на чанки.

## 4. Реализация
### 4.1 Форматы данных
Базовый контейнер `Image<T, Channels>` хранит `width`, `height`, `std::vector<T> data` (AoS, RGBA подряд). Используются типы `ImageRGBA8`, `ImageRGBAf`, `ImageGray8` из `include/alpha_hist/image.hpp`.

### 4.2 Alpha blending (scalar)
Pipeline реализован в `src/alpha_blend.cpp` через `alpha_blend_pipeline_templ<Impl, ExecMode>`:
- preprocess: sRGB -> linear + premultiply,
- blend: kernel Over/In/Out/Atop/Xor,
- postprocess: revert premultiply + linear -> sRGB.
Каждая стадия измеряется по времени и циклам (`BlendStageTiming`, `BlendStageCycles`).

### 4.3 Alpha blending (SIMD)
В SIMD-ядре обрабатываются 8 пикселей за итерацию (4 вектора по 2 пикселя). Для режима Over альфа реплицируется внутри 128-битных половин, затем используется FMA:

```cpp
// src/alpha_blend.cpp (blend_simd_chunk<BlendMode::Over>)
__m256 fg_scaled_0 = _mm256_mul_ps(fg_v0, glob_op);
__m256 fg_a_scaled_0 = _mm256_shuffle_ps(fg_scaled_0, fg_scaled_0, 255);
__m256 F_b0 = _mm256_sub_ps(one, fg_a_scaled_0);
rgba_out_0 = _mm256_fmadd_ps(F_b0, bg_v0, fg_scaled_0);
```

Хвост диапазона обрабатывается скалярно, а после SIMD-части вызывается `_mm256_zeroupper()` для исключения штрафа AVX->SSE.

### 4.4 Histogram (scalar)
В `src/histogram.cpp` используются 4 локальных массива `h0..h3` для уменьшения зависимости в инкрементах, затем выполняется редукция в итоговый `out[256]`.

### 4.5 Histogram (SIMD)
SIMD-версия загружает блоки `__m256i`, извлекает байты и считает частные гистограммы по 4 массивам, затем делает SIMD-редукцию:

```cpp
// src/histogram.cpp
static inline void consume32_4hist(std::array<std::array<std::uint32_t, CSIZE>, 4>& h, __m256i v) {
    std::uint64_t p0_31   = _mm256_extract_epi64(v, 0);
    std::uint64_t p32_63  = _mm256_extract_epi64(v, 1);
    std::uint64_t p64_95  = _mm256_extract_epi64(v, 2);
    std::uint64_t p96_127 = _mm256_extract_epi64(v, 3);
    extract_chars(h[0], p0_31);
    extract_chars(h[1], p32_63);
    extract_chars(h[2], p64_95);
    extract_chars(h[3], p96_127);
}
```

Такой подход избегает scatter-обновлений, но сохраняет скалярные инкременты.

### 4.6 ThreadPool
ThreadPool (`include/alpha_hist/thread_pool.hpp`, `src/thread_pool.cpp`) использует `std::jthread` и `stop_token`:

```cpp
// src/thread_pool.cpp
cv_.wait(lock, st, [this]{ return !tasks_.empty(); });
if (tasks_.empty()) {
    return;
}
task = std::move(tasks_.front());
tasks_.pop();
```

Остановка производится через `accepting_ = false`, `request_stop()` и `notify_all()`; `jthread` делает join при `workers_.clear()`.

### 4.7 Execution policies (SeqExec/ParExec)
`SeqExec` запускает один диапазон. `ParExec` делит диапазон на чанки, выравнивает границы под SIMD и использует `std::latch`:

```cpp
// include/alpha_hist/parallel_exec.hpp
size_t base = (n + target_chunks - 1) / target_chunks;
base = std::max(base, size_t(256 * 1024));
step = round_up(base, use_align);
std::latch done(num_tasks);
pool.enqueue([&, b, e] { fn(b, e); done.count_down(); });
done.wait();
```

### 4.8 Интеграция ThreadPool
Каждый этап alpha blending использует `exec.for_pixels(n, ...)` и одинаковый шаблон разбиения на чанки. Это дает возможность комбинировать SIMD и многопоточность.

## 5. Эксперименты
### 5.1 Данные
Использованы синтетические изображения из `data/png_test` (stripe, circle, sine, blobs, gauss, etc.) с размерами 1, 2, 4, 8, 16 и 32 MPix.

### 5.2 Окружение
- CPU: AMD Ryzen 7 6800H, 8C/16T.
- OS: Linux 6.6 (WSL2).
- Compiler: GCC 14.1.0 (`/usr/local/gcc-14.1.0/bin/g++-14.1.0`).
- Сборка: `-O3 -mavx2 -mfma -fno-tree-loop-vectorize -fno-tree-slp-vectorize` (C++20).
- CPU affinity: `taskset -c 0-15` (см. `utils/benchmark.sh`).

### 5.3 Метрики
- Для alpha blending: `preprocess_ns`, `blend_ns`, `postprocess_ns`, `total_ns`, а также циклы через `_rdtsc`.
- Для histogram: `timing_ns` и `cycles`.
- Speedup: `S = T_scalar / T_simd`, `S_par = T_seq / T_par`.
- Throughput: `MPix/s = N / (time_sec)`.

### 5.4 Методика измерений
Для alpha blending использовались 10 итераций, первая итерация отбрасывалась как warm-up (см. `tests/test_alpha_blend.cpp`). Для histogram применяется аналогичный warm-up в `tests/test_histogram.cpp`. В отчете приведены медианные значения для alpha blending (из `plots_out/tables`) и средние значения для histogram (по CSV).

## 6. Результаты и анализ
### 6.1 SIMD vs Scalar (alpha blending, seq)
Медианные времена из `plots_out/tables/simd_benefit.csv`:

| MPix | scalar seq ms | simd seq ms | speedup (scalar/simd) |
|---:|---:|---:|---:|
| 1 | 18.19 | 16.14 | 1.127 |
| 2 | 36.41 | 31.09 | 1.171 |
| 4 | 172.33 | 163.12 | 1.057 |
| 8 | 344.58 | 319.97 | 1.077 |
| 16 | 726.93 | 680.22 | 1.069 |
| 32 | 2001.51 | 1930.82 | 1.037 |

- Средний выигрыш SIMD по размерам: ~1.09x, максимум ~1.17x на 2 MPix.
- Ускорение ограничено памятью и затратами на LUT-конверсии.

Дополнительно показаны раздельные метрики по стадиям (preprocess/blend/postprocess), а также распределения и процентильные кривые total time.

![Alpha blending SIMD vs scalar (total)](plots_out/plots_simd_vs_scalar/alpha_blend/blend_metric_vs_mp_total_ns.png)
![Alpha blending SIMD vs scalar (preprocess)](plots_out/plots_simd_vs_scalar/alpha_blend/blend_metric_vs_mp_preprocess_ns.png)
![Alpha blending SIMD vs scalar (blend)](plots_out/plots_simd_vs_scalar/alpha_blend/blend_metric_vs_mp_blend_ns.png)
![Alpha blending SIMD vs scalar (postprocess)](plots_out/plots_simd_vs_scalar/alpha_blend/blend_metric_vs_mp_postprocess_ns.png)
![Alpha blending stage boxplots](plots_out/plots_simd_vs_scalar/alpha_blend/blend_stage_boxplots_ns.png)
![Alpha blending stage mean (stacked)](plots_out/plots_simd_vs_scalar/alpha_blend/blend_stage_mean_stacked.png)
![Alpha blending stage mean (pies)](plots_out/plots_simd_vs_scalar/alpha_blend/blend_stage_mean_pies.png)
![Alpha blending total ns boxplot](plots_out/plots_simd_vs_scalar/alpha_blend/blend_total_ns_boxplot.png)
![Alpha blending total ns by index](plots_out/plots_simd_vs_scalar/alpha_blend/blend_total_ns_by_index.png)
![Alpha blending total ns cumulative](plots_out/plots_simd_vs_scalar/alpha_blend/blend_total_ns_cumulative.png)
![Alpha blending total ns hist KDE](plots_out/plots_simd_vs_scalar/alpha_blend/blend_total_ns_hist_kde.png)
![Alpha blending total ns percentile](plots_out/plots_simd_vs_scalar/alpha_blend/blend_total_ns_percentile.png)

### 6.2 Histogram (scalar vs SIMD)
Средние значения по CSV из `results_release/hist_results_*`:

| MPix | scalar ms (mean) | simd ms (mean) | scalar MPix/s | simd MPix/s | speedup (scalar/simd) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.307 | 0.431 | 3256.7 | 2322.0 | 0.713 |
| 2 | 0.598 | 0.851 | 3345.5 | 2351.0 | 0.703 |
| 4 | 1.164 | 1.685 | 3437.8 | 2374.4 | 0.691 |
| 8 | 2.270 | 3.401 | 3526.3 | 2353.6 | 0.667 |
| 16 | 4.357 | 6.788 | 3673.0 | 2357.4 | 0.642 |
| 32 | 8.528 | 13.593 | 3752.9 | 2354.5 | 0.627 |

- SIMD-реализация здесь медленнее (speedup < 1), так как приходится извлекать байты и делать скалярные инкременты.
- Scalar стабильно держит ~3.3-3.8 Gpix/s, SIMD ~2.3 Gpix/s.

Ниже добавлены разные виды распределений (boxplot/percentile/cumulative) для сравнения стабильности и хвостов.

![Histogram timing](plots_out/plots_simd_vs_scalar/histogram/hist_timing_ns_boxplot.png)
![Histogram timing cumulative](plots_out/plots_simd_vs_scalar/histogram/hist_timing_ns_cumulative.png)
![Histogram timing hist KDE](plots_out/plots_simd_vs_scalar/histogram/hist_timing_ns_hist_kde.png)
![Histogram timing percentile](plots_out/plots_simd_vs_scalar/histogram/hist_timing_ns_percentile.png)

### 6.3 ThreadPool scaling (alpha blending, par)
Лучшие значения по потокам из `plots_out/tables/best_threads_per_mpix.csv`:

| MPix | scalar best threads | scalar speedup vs seq | simd best threads | simd speedup vs seq | simd speedup vs scalar seq |
|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 1.520 | 32 | 1.450 | 1.633 |
| 2 | 1 | 0.979 | 2 | 1.163 | 1.362 |
| 4 | 16 | 1.205 | 16 | 1.145 | 1.209 |
| 8 | 16 | 1.215 | 8 | 1.133 | 1.220 |
| 16 | 16 | 1.218 | 4 | 1.144 | 1.223 |
| 32 | 8 | 1.237 | 8 | 1.231 | 1.276 |

- Средний speedup vs seq: ~1.23x (scalar) и ~1.21x (simd).
- Лучшие значения достигаются при 8-16 потоках на больших размерах.

Ниже представлены все графики из `plots_out/plots`, включая best-of метрики и boxplot по стадиям для малого и большого размера.

![Speedup vs threads](plots_out/plots/speedup_vs_threads_grid.png)
![Best threads vs MPix](plots_out/plots/best_threads_vs_mpix.png)
![Best total time (overall)](plots_out/plots/best_of_total_time.png)
![Best ns per pixel (overall)](plots_out/plots/ns_per_pixel_best_of.png)
![Preprocess boxplot (1 MP)](plots_out/plots/box_preprocess_1p0005mpix.png)
![Preprocess boxplot (32 MP)](plots_out/plots/box_preprocess_32p004949mpix.png)
![Blend boxplot (1 MP)](plots_out/plots/box_blend_1p0005mpix.png)
![Blend boxplot (32 MP)](plots_out/plots/box_blend_32p004949mpix.png)
![Postprocess boxplot (1 MP)](plots_out/plots/box_postprocess_1p0005mpix.png)
![Postprocess boxplot (32 MP)](plots_out/plots/box_postprocess_32p004949mpix.png)

### 6.4 SIMD + Threads vs baseline
- Максимальный выигрыш относительно scalar seq: 1.63x (1 MPix, simd par, 32 потока).
- Для больших изображений (32 MPix) итоговый выигрыш ~1.28x относительно scalar seq.
- На больших размерах ограничение смещается в сторону пропускной способности памяти.

## 7. Заключение
- Реализованы scalar и SIMD версии alpha blending и histogram, а также ThreadPool и execution policies.
- SIMD дает умеренный выигрыш для alpha blending (~1.09x в среднем), но для histogram текущая SIMD-версия медленнее из-за дорогостоящего извлечения байтов.
- Многопоточность дает стабильный рост на больших изображениях (~1.2-1.3x), но ограничена bandwidth и overhead на малых размерах.
- Комбинация SIMD + Threads дает максимум 1.63x ускорения относительно scalar seq на текущем стенде.
