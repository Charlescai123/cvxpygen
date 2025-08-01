
/*
Auto-generated by CVXPYgen on July 17, 2025 at 15:09:13.
Content: Declarations for Python binding with pybind11.
*/

// User-defined parameters
struct CPG_Params_cpp_t {
    std::array<double, 3> A;
    std::array<double, 3> b;
};

// Flags for updated user-defined parameters
struct CPG_Updated_cpp_t {
    bool A;
    bool b;
};

// Primal solution
struct CPG_Prim_cpp_t {
    std::array<double, 2> x;
};

// Dual solution
struct CPG_Dual_cpp_t {
    std::array<double, 2> d0;
};

// Solver information
struct CPG_Info_cpp_t {
    double obj_val;
    int iter;
    int status;
    double pri_res;
    double dua_res;
    double time;
};

// Solution and solver information
struct CPG_Result_cpp_t {
    CPG_Prim_cpp_t prim;
    CPG_Dual_cpp_t dual;
    CPG_Info_cpp_t info;
};

// Main solve function
CPG_Result_cpp_t solve_cpp(struct CPG_Updated_cpp_t& CPG_Updated_cpp, struct CPG_Params_cpp_t& CPG_Params_cpp);
