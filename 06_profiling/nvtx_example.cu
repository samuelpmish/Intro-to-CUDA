#include <nvtx3/nvToolsExt.h>
#include <thread>
#include <chrono>

void foo() {
    nvtxRangePushA("foo");
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    nvtxRangePop();
}

void bar() {
    nvtxRangePushA("bar");
    std::this_thread::sleep_for(std::chrono::milliseconds{3});
    nvtxRangePop();
}

void baz() {
    nvtxRangePushA("baz");
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    nvtxRangePop();
}

void some_function() {
  nvtxMark("beginning");

  for (int i = 0; i < 10; ++i) {
    nvtxRangePushA("loop");
    foo();
    bar();
    baz();

    if (i == 7) {
      nvtxMark("Hello world!");
    }
    nvtxRangePop();
  }

  nvtxMark("end");
}

int main() {
  some_function();
  return 0;
}