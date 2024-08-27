#pragma once

namespace mlir {
class MLIRContext;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace sandbox {

namespace ast {
class Module;
} // namespace ast

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext& a_context,
    ast::Module& a_moduleAst);

} // namespace sandbox
