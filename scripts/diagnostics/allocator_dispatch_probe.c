/* Valid allocation and libdispatch activity only; no deliberate invalid access.
 * Supplemental diagnostic: a negative run cannot waive failed pricing cases. */
#include <dispatch/dispatch.h>
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

enum { SLOTS = 256, MAX_WORKERS = 16, MAX_TASKS = 512 };
struct slot { _Atomic uint64_t *p; size_t words; uint64_t token; };
struct worker { unsigned id; uint64_t operations; };
struct task { unsigned id; uint64_t round; };
static atomic_bool started, stopped;
static atomic_uint_fast64_t tasks_done;

static double now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) abort();
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}
static void verify(const struct slot *s) {
    if (!s->p) return;
    for (size_t i = 0; i < s->words; ++i) {
        uint64_t actual = atomic_load_explicit(&s->p[i], memory_order_relaxed);
        if (actual != s->token + i) {
            fprintf(stderr, "live payload changed pointer=%p word=%zu expected=%" PRIu64 " actual=%" PRIu64 "\n",
                    (void *)s->p, i, s->token + i, actual);
            abort();
        }
    }
}
static void fill(struct slot *s, uint64_t token) {
    s->token = token;
    for (size_t i = 0; i < s->words; ++i)
        atomic_store_explicit(&s->p[i], token + i, memory_order_relaxed);
}
static uint64_t next(uint64_t *s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s;
}
static void *allocate(void *arg) {
    struct worker *w = arg;
    struct slot slots[SLOTS] = {0};
    uint64_t rng = UINT64_C(0x9e3779b97f4a7c15) ^ (uint64_t)(w->id + 1);
    uint64_t count = 0;
    while (!atomic_load_explicit(&started, memory_order_acquire)) sched_yield();
    while (!atomic_load_explicit(&stopped, memory_order_relaxed)) {
        uint64_t choice = next(&rng);
        struct slot *s = &slots[choice % SLOTS];
        verify(s);
        if (s->p && ((choice >> 12) & 3) == 0) {
            free(s->p); s->p = NULL; s->words = 0;
        } else {
            size_t words = (size_t)1 << ((choice >> 16) % 6);
            void *p = s->p ? realloc(s->p, words * sizeof(uint64_t)) : malloc(words * sizeof(uint64_t));
            if (!p) abort();
            s->p = p; s->words = words;
            fill(s, ((uint64_t)(w->id + 1) << 56) ^ count);
        }
        ++count;
    }
    for (unsigned i = 0; i < SLOTS; ++i) { verify(&slots[i]); free(slots[i].p); }
    w->operations = count;
    return NULL;
}
static void dispatched(void *arg) {
    const struct task *t = arg;
    struct slot slots[16] = {0};
    for (unsigned i = 0; i < 16; ++i) {
        slots[i].words = (size_t)1 << (i % 6);
        slots[i].p = malloc(slots[i].words * sizeof(uint64_t));
        if (!slots[i].p) abort();
        fill(&slots[i], (t->round << 32) ^ ((uint64_t)t->id << 16) ^ i);
    }
    /* Allow multiple workqueue callbacks to overlap, then become idle. */
    struct timespec pause = {0, 2000000};
    while (nanosleep(&pause, &pause) && errno == EINTR) {}
    for (unsigned i = 0; i < 16; ++i) { verify(&slots[i]); free(slots[i].p); }
    atomic_fetch_add_explicit(&tasks_done, 1, memory_order_relaxed);
}
static unsigned number(const char *s, unsigned max) {
    char *end; errno = 0; unsigned long v = strtoul(s, &end, 10);
    if (errno || !*s || *end || !v || v > max) exit(2);
    return (unsigned)v;
}
int main(int argc, char **argv) {
    if (argc != 6) return 2;
    unsigned workers = number(argv[1], MAX_WORKERS), seconds = number(argv[2], 600);
    unsigned task_count = number(argv[3], MAX_TASKS), idle = number(argv[4], 30);
    if (getenv("MallocNanoZone") || getenv("MallocScribble") || getenv("MallocStackLogging") ||
        getenv("MallocGuardEdges") || getenv("DYLD_INSERT_LIBRARIES")) return 2;
    alarm(seconds + idle + 60);
    FILE *report = fopen(argv[5], "w"); if (!report) return 2;
    fprintf(report, "{\"event\":\"start\",\"workers\":%u,\"seconds\":%u,\"tasks_per_burst\":%u,\"idle_seconds\":%u,\"live_slots_per_worker\":%u}\n",
            workers, seconds, task_count, idle, SLOTS);
    fflush(report);
    pthread_t threads[MAX_WORKERS];
    struct worker state[MAX_WORKERS] = {0};
    for (unsigned i = 0; i < workers; ++i) {
        state[i].id = i;
        if (pthread_create(&threads[i], NULL, allocate, &state[i])) abort();
    }
    double before = now(), deadline = before + seconds;
    atomic_store_explicit(&started, true, memory_order_release);
    uint64_t bursts = 0;
    while (now() < deadline) {
        dispatch_group_t group = dispatch_group_create();
        if (!group) abort();
        struct task tasks[MAX_TASKS];
        for (unsigned i = 0; i < task_count; ++i) {
            tasks[i].id = i; tasks[i].round = bursts;
            dispatch_group_async_f(group, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), &tasks[i], dispatched);
        }
        if (dispatch_group_wait(group, dispatch_time(DISPATCH_TIME_NOW, 30 * NSEC_PER_SEC))) abort();
        dispatch_release(group);
        ++bursts;
        uint64_t completed = atomic_load_explicit(&tasks_done, memory_order_relaxed);
        if (completed != bursts * task_count) abort();
        fprintf(report, "{\"event\":\"burst\",\"burst\":%" PRIu64 ",\"tasks_completed\":%" PRIu64 ",\"elapsed_seconds\":%.6f}\n", bursts, completed, now() - before);
        fflush(report);
        struct timespec pause = {(time_t)idle, 0};
        while (nanosleep(&pause, &pause) && errno == EINTR) {}
    }
    atomic_store_explicit(&stopped, true, memory_order_relaxed);
    for (unsigned i = 0; i < workers; ++i)
        if (pthread_join(threads[i], NULL) || state[i].operations == 0) abort();
    fprintf(report, "{\"event\":\"completed\",\"bursts\":%" PRIu64 ",\"tasks_completed\":%" PRIuFAST64 ",\"elapsed_seconds\":%.6f,\"worker_operations\":[",
            bursts, atomic_load(&tasks_done), now() - before);
    for (unsigned i = 0; i < workers; ++i) fprintf(report, "%s%" PRIu64, i ? "," : "", state[i].operations);
    fputs("]}\n", report);
    if (fclose(report)) return 2;
    puts("allocator/dispatch diagnostic completed without reproduction");
    return 0;
}
