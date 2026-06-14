{
  description = "Development shell for YA-GPT";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfreePredicate =
              pkg:
              builtins.elem (nixpkgs.lib.getName pkg) [
                "cuda_nvcc"
              ];
          };
          python = pkgs.python313;
          cudaNvcc = pkgs.cudaPackages.cuda_nvcc;
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              cudaNvcc
              pkgs.clang
              pkgs.cmake
              pkgs.ninja
              pkgs.uv
              pkgs.git
              pkgs.pkg-config
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
            ];

            UV_PYTHON = "${python}/bin/python";
            UV_PYTHON_DOWNLOADS = "never";
            TRITON_LIBCUDA_PATH = "/run/opengl-driver/lib";
            TRITON_PTXAS_PATH = "${cudaNvcc}/bin/ptxas";
            LD_LIBRARY_PATH = nixpkgs.lib.makeLibraryPath [
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
            ] + ":/run/opengl-driver/lib";

            shellHook = ''
              echo "Python: $(python --version)"
              echo "uv: $(uv --version)"
              echo "Run: uv sync"
            '';
          };
        }
      );
    };
}
