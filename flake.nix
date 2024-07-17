{
  description = "Rust flake with nightly and CUDA 12.1 support";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
  };
  outputs = { self, flake-utils, rust-overlay, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = (import nixpkgs) {
          inherit system overlays;
          config.allowUnfree = true; # Needed for CUDA
        };
      in
      with pkgs;
      {
        devShells.default = mkShell rec {
          nativeBuildInputs = [
            (rust-bin.stable.latest.default.override {
              extensions = [ "rust-analyzer" "clippy" "rust-src" ];
            })
            cudaPackages.cuda_12_1
            wget
            unzip
          ];
          buildInputs = [
            udev alsa-lib vulkan-loader
            vulkan-tools vulkan-headers vulkan-validation-layers
            pkg-config
            xorg.libX11 xorg.libXcursor xorg.libXi xorg.libXrandr
            libxkbcommon wayland
          ];
          LD_LIBRARY_PATH = lib.makeLibraryPath buildInputs;
          VULKAN_SDK = "${vulkan-headers}";
          VK_LAYER_PATH = "${vulkan-validation-layers}/share/vulkan/explicit_layer.d";
          
          # CUDA and LibTorch specific configurations
          CUDA_PATH = "${cudaPackages.cuda_12_1}";
          LIBTORCH = "$HOME/.local/lib/libtorch";
          
          shellHook = ''
            export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
            
            if [ ! -d "${LIBTORCH}" ]; then
              echo "Downloading and extracting LibTorch..."
              wget -O libtorch.zip https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcu121.zip
              unzip libtorch.zip -d ${LIBTORCH}
              rm libtorch.zip
            fi
          '';
        };
      }
    );
}
