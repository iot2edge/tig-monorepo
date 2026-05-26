#include <stdint.h>

__attribute__((weak, visibility("default"))) uint64_t __fuel_remaining = 0;
__attribute__((weak, visibility("default"))) uint64_t __runtime_signature = 0;
__attribute__((weak, visibility("default"))) uint64_t __curr_memory_usage = 0;
__attribute__((weak, visibility("default"))) uint64_t __total_memory_usage = 0;
__attribute__((weak, visibility("default"))) uint64_t __max_memory_usage = 0;
__attribute__((weak, visibility("default"))) uint64_t __max_allowed_memory_usage = 0;
__thread __attribute__((weak, visibility("default"))) uint64_t __thread_local_fuel_used = 0;
__thread __attribute__((weak, visibility("default"))) uint64_t __thread_local_runtime_signature = 0;
__thread __attribute__((weak, visibility("default"))) uint64_t __thread_local_curr_memory_usage = 0;
__thread __attribute__((weak, visibility("default"))) uint64_t __thread_local_total_memory_usage = 0;
__thread __attribute__((weak, visibility("default"))) uint64_t __thread_local_max_memory_usage = 0;
__thread __attribute__((weak, visibility("default"))) uint64_t __thread_local_max_allowed_memory_usage = 0;
