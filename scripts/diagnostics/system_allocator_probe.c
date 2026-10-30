/* Standalone valid malloc/free stress inspired by JDK-8350455/JDK-8350375.
 * No pricing, Python, Rust, sanitizer, or deliberate invalid access is linked.
 * A negative run cannot prove that an intermittent allocator fault is absent. */
#include <dlfcn.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

_Static_assert(sizeof(_Atomic uint64_t) == 8, "probe requires 8-byte atomic words");
_Static_assert(ATOMIC_LONG_LOCK_FREE == 2 && ATOMIC_LLONG_LOCK_FREE == 2,
               "probe requires lock-free 64-bit words");

struct worker {
    unsigned index;
    size_t bytes;
    uint64_t iterations;
};

static atomic_bool started;
static atomic_bool stopped;

static double monotonic(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) abort();
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static void *work(void *argument) {
    struct worker *worker = argument;
    while (!atomic_load_explicit(&started, memory_order_acquire)) sched_yield();
    uint64_t count = 0;
    while (!atomic_load_explicit(&stopped, memory_order_relaxed)) {
        _Atomic uint64_t *allocation = malloc(worker->bytes);
        if (allocation == NULL) abort();
        const uint64_t token = ((uint64_t)(worker->index + 1) << 48) ^ count;
        for (size_t i = 0; i < worker->bytes / sizeof(uint64_t); ++i)
            atomic_store_explicit(&allocation[i], token + i, memory_order_relaxed);
        for (size_t i = 0; i < worker->bytes / sizeof(uint64_t); ++i) {
            const uint64_t observed = atomic_load_explicit(&allocation[i], memory_order_relaxed);
            if (observed != token + i) {
                fprintf(stderr, "live allocation changed: worker=%u pointer=%p word=%zu expected=%" PRIu64 " observed=%" PRIu64 "\n",
                        worker->index, (void *)allocation, i, token + i, observed);
                abort();
            }
        }
        free(allocation);
        ++count;
    }
    worker->iterations = count;
    return NULL;
}

static unsigned number(const char *text, unsigned maximum) {
    char *end;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || *text == '\0' || *end != '\0' || value == 0 || value > maximum) {
        fprintf(stderr, "invalid probe argument: %s\n", text);
        exit(2);
    }
    return (unsigned)value;
}

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    const unsigned workers = number(argv[1], 64);
    const unsigned rounds = number(argv[2], 64);
    const unsigned seconds = number(argv[3], 30);
    if (rounds * 2 * seconds > 900) return 2;
    struct rlimit core_limit = {0, 0};
    if (setrlimit(RLIMIT_CORE, &core_limit) != 0) return 2;
    /* A deadlock is an incomplete diagnostic, never an unbounded run. */
    alarm(rounds * 2 * seconds + 60);
    FILE *report = fopen(argv[4], "w");
    if (report == NULL) return 2;
    unsigned (*nano)(void) = (unsigned (*)(void))dlsym(RTLD_DEFAULT, "malloc_engaged_nano");
    bool (*secure)(void) = (bool (*)(void))dlsym(RTLD_DEFAULT, "malloc_engaged_secure_allocator");
    fprintf(report, "{\"event\":\"start\",\"workers\":%u,\"rounds\":%u,\"seconds\":%u,\"nano_mode\":", workers, rounds, seconds);
    if (nano) fprintf(report, "%u", nano()); else fputs("null", report);
    fputs(",\"secure_allocator\":", report);
    if (secure) fprintf(report, "%s", secure() ? "true" : "false"); else fputs("null", report);
    fputs("}\n", report);
    fflush(report);
    const size_t sizes[] = {8, 64};
    for (unsigned round = 0; round < rounds; ++round) {
        for (unsigned size_index = 0; size_index < 2; ++size_index) {
            fprintf(report, "{\"event\":\"case-start\",\"round\":%u,\"bytes\":%zu}\n", round, sizes[size_index]);
            fflush(report);
            pthread_t threads[64];
            struct worker state[64] = {0};
            atomic_store(&started, false);
            atomic_store(&stopped, false);
            for (unsigned i = 0; i < workers; ++i) {
                state[i].index = i;
                state[i].bytes = sizes[size_index];
                if (pthread_create(&threads[i], NULL, work, &state[i]) != 0) abort();
            }
            const double before = monotonic();
            atomic_store_explicit(&started, true, memory_order_release);
            struct timespec remaining = {(time_t)seconds, 0};
            while (nanosleep(&remaining, &remaining) != 0) {
                if (errno != EINTR) abort();
            }
            atomic_store(&stopped, true);
            for (unsigned i = 0; i < workers; ++i)
                if (pthread_join(threads[i], NULL) != 0) abort();
            fprintf(report, "{\"event\":\"case\",\"round\":%u,\"bytes\":%zu,\"elapsed_seconds\":%.9f,\"iterations\":[", round, sizes[size_index], monotonic() - before);
            for (unsigned i = 0; i < workers; ++i) {
                if (state[i].iterations == 0) abort();
                fprintf(report, "%s%" PRIu64, i ? "," : "", state[i].iterations);
            }
            fputs("]}\n", report);
            fflush(report);
        }
    }
    fprintf(report, "{\"event\":\"completed\",\"cases\":%u}\n", rounds * 2);
    if (fclose(report) != 0) return 2;
    puts("system allocator probe completed");
    return 0;
}
