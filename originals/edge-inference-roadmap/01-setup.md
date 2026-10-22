# Setup and first commands

## Computer and phone roles

Use your computer for editing, building, exporting checkpoints and running Python evaluation. Use the phone as a measured Android target. Start without root. Do not change kernel settings or thermal protections to complete this curriculum.

Record the exact handset model and Android build. The supplied Snapdragon 8 Elite Gen 5 and 16 GB RAM are the planned target, but available RAM, OEM libraries and supported accelerators must be observed. Disable marketing RAM expansion in your capacity reasoning: it is not additional physical memory.

## Host prerequisites

Install a supported C++ compiler (Clang or GCC), debugger, CMake, a build tool such as Ninja, Git, and a Python virtual environment. Later add Android Studio, SDK platform tools and a compatible Android NDK. Use official installers or your operating system's supported packages. Vendor tools may require a particular supported host environment; follow that release's requirements.

For Windows, a native Android Studio environment is sufficient for the Android side. A Linux development environment can simplify host C++ and vendor workflows, but do not assume every USB or vendor-tool step works identically inside WSL. Start with the smallest host build and a working native ADB connection.

## First C++ program

Create `hello.cpp` yourself containing a main function that prints a short line. On a compatible host with Clang, a basic command is:

```bash
clang++ -std=c++20 -Wall -Wextra -Wpedantic -g hello.cpp -o hello
./hello
```

The equivalent GCC command uses `g++`. On Windows use the correct executable name or IDE terminal for your compiler. Explain the language standard, warnings, symbols, input and output arguments. Add conversion warnings later and resolve them deliberately rather than globally suppressing them.

## Minimal CMake scaffold

```cmake
cmake_minimum_required(VERSION 3.20)
project(edge_learning LANGUAGES CXX)
add_executable(hello hello.cpp)
target_compile_features(hello PRIVATE cxx_std_20)
```

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug
```

Single-configuration generators use `CMAKE_BUILD_TYPE`; multi-configuration generators also need the appropriate build configuration. Keep debug, release, sanitizer and Android outputs in separate directories. Use your installed stable CMake guide rather than assuming the newest online tutorial matches it.

## First ADB checks

After installing platform tools, enable developer options and USB debugging on the phone, authorize your computer, then use:

```bash
adb devices
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.product.cpu.abi
adb shell cat /proc/meminfo
```

These inspect the device. Memory output is system-wide and does not establish an app's safe allocation budget. Do not paste private device identifiers into public reports.

## Runtime setup

Follow [upstream Android instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md) for CPU deployment and [the Snapdragon guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md) for the selected accelerator path. Pin a commit and record it. Use the commands from that checkout because binary names, flags and SDK requirements change.

Choose a small official checkpoint and a compatible model artifact. Prefer approximately 1B parameters for runtime baselines and approximately 100–400M for your own educational engine. Do not assume every architecture is supported by every backend or that all quants are accepted. Download model assets manually according to their terms; this pack contains no model weights.

## Suggested learning repository

Create these directories yourself: `cpp-basics`, `tensor-core`, `python-reference`, `native-engine`, `android-app`, `benchmarks`, `evaluation`, `fixtures`, and `notes`. Add a root README explaining what belongs where. Keep large model files and generated build outputs out of Git.

Your benchmark script should use subprocess argument arrays, explicit timeouts, and captured stderr. Do not build a shell command by concatenating untrusted prompts or file paths. Preserve the exact raw output before attempting to parse it.

## Initial decisions to record

- Host operating system, compiler, CMake, Python and Android tooling versions.
- Phone model, Android version, firmware identifier and available storage.
- Runtime commit, backend and build configuration.
- Model source, revision, architecture, tokenizer, chat template, quant and hash.
- Context limit, maximum output tokens and sampling settings.
- Which steps are already verified and which are still assumptions.
