#include "AST.h"
#include "Dialect.h"
#include "MlirGen.h"
#include "Passes.h"
#include "SampleAst_2.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    std::cout << "sandbox_run" << std::endl;
    sandbox::ast::Module* astModule = sandbox::ast::sampleAst_2();
    std::cout << "AST dump:" << std::endl;
    sandbox::dump(std::cout, *astModule);

    mlir::MLIRContext context {};
    context.getOrLoadDialect<mlir::sandbox::SandboxDialect>();
    context.printOpOnDiagnostic(true);
    context.printStackTraceOnDiagnostic(true);
    mlir::OwningOpRef<mlir::ModuleOp> mlirModule = sandbox::mlirGen(context, *astModule);
    if (!mlirModule)
    {
        std::cerr << "Could not convert AST module to MLIR" << std::endl;
        return -1;
    }
    std::cout << std::string(32, '*') << std::endl;
    std::cout << "MLIR dump:" << std::endl;
    mlirModule->dump();

    std::cout << std::string(32, '*') << std::endl;
    std::cout << "Will now apply passes" << std::endl;
    mlir::PassManager pm(mlirModule.get()->getName());
    pm.addPass(mlir::sandbox::createLoweringPass());
    if (mlir::failed(pm.run(*mlirModule)))
    {
        std::cerr << "Could not run passes" << std::endl;
    }
    std::cout << "MLIR dump after pass:" << std::endl;
    mlirModule->dump();

    return 0;
}
