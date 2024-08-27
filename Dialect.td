#ifndef SANDBOX_DIALECT
#define SANDBOX_DIALECT

include "mlir/IR/DialectBase.td"

def Sandbox_Dialect : Dialect {
    let name = "sandbox";
    let cppNamespace = "::mlir::sandbox";
}

#endif // SANDBOX_DIALECT
