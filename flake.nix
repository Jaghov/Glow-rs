{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, ... } @ inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = ["x86_64-linux"];
      perSystem = { config, system, lib, ... }: let
        overlays = [
          (import inputs.rust-overlay)
        ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        # Set the CUDA version to match libtorch's CUDA version
        cudaVersion = "12.1";
        cudaPkg = pkgs.cudaPackages_12_1.cudatoolkit;
        libtorch = pkgs.libtorch-bin.override { cudaSupport = true; };
      in {
        devShells.default = pkgs.mkShell rec {
          packages = with pkgs; [
            pkg-config
            openssl
            glxinfo
            vscode-extensions.llvm-org.lldb-vscode
            taplo
            mdbook
            glib-networking
            cudaPkg
            cudaPackages.cudnn
            rust-bin.stable.latest.default
            typos
            libxkbcommon
            libGL
            wayland
            vulkan-tools
            vulkan-loader
            python311Packages.onnx
            libtorch          # Runtime libraries
            libtorch.dev      # Development files (headers)
          ];
          LD_LIBRARY_PATH = "${lib.makeLibraryPath packages}";
          # Set TORCH_CUDA_VERSION to match the CUDA version
          TORCH_CUDA_VERSION = "cu${builtins.replaceStrings ["." ] ["" ] cudaVersion}";
          PATH = "~/.cargo/bin:$PATH";
          # Point LIBTORCH to the dev output (includes headers)
          LIBTORCH = "${libtorch}";
          # Set CXXFLAGS to include header paths
          CXXFLAGS = "-I${libtorch.dev}/include -I${libtorch.dev}/include/torch/csrc/api/include";
        };
        formatter = pkgs.alejandra;
      };
    };
}
