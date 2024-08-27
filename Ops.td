#ifndef sandbox_OPS
#define sandbox_OPS

include "Dialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class, inheriting from Op (in OpBase.td)
class Sandbox_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sandbox_Dialect, mnemonic, traits>;

def ConstantOp : Sandbox_Op<"constant", [Pure]> {
    let summary = "constant";
    let description = [{
        Constant operation turns a literal into an SSA value. The value is attached
        to the operation as an attribute. For example:

        ```mlir
            %0 = sandbox.constant RD:FIXME
        ```
    }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyTypeOf<[UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64, F32, F64]>:$output);
    let assemblyFormat = "`(` $value `)` attr-dict `:` type($output)";
    let hasVerifier = 1;
}

def AddOp : Sandbox_Op<"add", [Pure]> {
    let summary = "add operation";
    let description = [{
        The 'add' operation adds its two inputs
    }];

    let arguments = (ins AnyTypeOf<[UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64, F32, F64]>:$lhs, AnyTypeOf<[UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64, F32, F64]>:$rhs);
    let results = (outs AnyTypeOf<[UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64, F32, F64]>);
    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def MulOp : Sandbox_Op<"mul", [Pure]> {
    let summary = "mul operation";
    let description = [{
        The 'mul' operation multiplies its two inputs
    }];

    // FIXME:RD temporarily accepting I32 below
    let arguments = (ins AnyTypeOf<[UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64, F32, F64, I32]>:$lhs, AnyTypeOf<[UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64, F32, F64, I32]>:$rhs);
    let results = (outs AnyTypeOf<[UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64, F32, F64]>);
    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def PrintOp : Sandbox_Op<"print"> {
    let summary = "print operation";
    let description = [{
        The 'print' builtin operation prints a given scalar, and produces no result
    }];

    let arguments = (ins AnyTypeOf<[UI8, UI16, UI32, UI64, SI8, SI16, SI32, SI64, F32, F64]>:$value);
}

def FuncOp : Sandbox_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "user defined function operation";
    let description = [{
        The "sandbox.func" operation represents a user defined function. These are
        callable SSA-region operations that contain sandbox computations.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [
        OpBuilder<(ins "StringRef":$name, "FunctionType":$type, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
    ];

    let extraClassDeclaration = [{
        //===-----------------------------------------===//
        // FunctionOpInterface methods
        //===-----------------------------------------===//

        /// Returns the argument types of this function
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        Region* getCallableRegion() { return &getBody(); }
    }];

    let skipDefaultBuilders = 1;
}

def ReturnOp : Sandbox_Op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
    let summary = "return operation";
    let description = [{
        The 'return' operation represents a return operation within a function.
        It takes an optional operand and produces no results.
        The operand type must match the signature of the function that contains
        the operation.
    }];

    let arguments = (ins Variadic<AnyType>:$input);

    // Allow building a ReturnOp with no return operand
    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];

    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    let hasVerifier = 1;
}

#endif // sandbox_OPS
