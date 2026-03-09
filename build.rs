fn main() {
    cc::Build::new()
        .cpp(true)
        // can't get this to work
        //.flag("-fsanitize=undefined")
        //.flag("-static-libasan")
        .flag("-march=native")
        .file("src/ncc.cpp")
        .compile("ncc");
    // why isn't this already happening?
    println!("cargo:rerun-if-changed=src/ncc.cpp");
}
