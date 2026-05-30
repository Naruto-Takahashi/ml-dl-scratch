{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "ml-dl-scratch-env";

  buildInputs = with pkgs; [
    python312
    python312Packages.numpy
    python312Packages.matplotlib
    python312Packages.jupyterlab
  ];

  shellHook = ''
    echo "========================================================="
    echo "  Welcome to the Machine Learning & Deep Learning Scratch  "
    echo "  Development Environment (Nix Shell)                     "
    echo "========================================================="
    echo "Available tools:"
    echo "  - Python: $(python --version)"
    echo "  - Jupyter Lab: Start with 'jupyter lab'"
    echo "========================================================="
  '';
}
