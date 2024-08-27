get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cc.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cc.inc -gen-dialect-defs)
add_public_tablegen_target(SandboxOpsIncGen)

set(LLVM_LINK_COMPONENTS
    Support
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include(CMakePrintHelpers)
cmake_print_variables(LLVM_HAS_RTTI)
add_executable(sandbox_run
    sandbox_run.x.cc
    AST.cc
    Dialect.cc
    LowerToLlvm.cc
    MlirGen.cc)

target_link_libraries(sandbox_run
  PRIVATE
    MLIRAnalysis
    MLIRArithDialect
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLLVMDialect
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRTransforms)

add_dependencies(sandbox_run
    SandboxOpsIncGen)
