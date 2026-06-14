# Engine Build

Enter the project dev shell from the repository root:

```sh
nix develop
```

Configure the CMake build directory:

```sh
cmake -S src/engine -B build/engine -G Ninja
```

Build the engine:

```sh
cmake --build build/engine
```

Run the server:

```sh
./build/engine/model-server
```

Test the health endpoint:

```sh
curl http://127.0.0.1:8080/health
```

After the first configure step, changes to C++ source files usually only need:

```sh
cmake --build build/engine
```

Rerun the configure command after changing `CMakeLists.txt`.
