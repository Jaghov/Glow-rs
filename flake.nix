{
  description = "Rust flake with nightly";

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
        };
      in
      with pkgs;
      {
        devShells.default = mkShell rec {
          nativeBuildInputs = [
            (rust-bin.stable.latest.default.override {
              extensions = [ "rust-analyzer" "clippy" "rust-src" ];
            })
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
        };
      }
    );
}
