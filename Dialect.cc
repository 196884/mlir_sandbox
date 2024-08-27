#include "Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <string>

using namespace mlir;
using namespace mlir::sandbox;

#include "Dialect.cc.inc"

void SandboxDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "Ops.cc.inc"
        >();
}

static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser& a_parser,
    mlir::OperationState& a_result)
{
    mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    mlir::SMLoc operandsLoc = a_parser.getCurrentLocation();
    mlir::Type type;
    if (a_parser.parseOperandList(operands, /*requiredOperandCount=*/2) || a_parser.parseOptionalAttrDict(a_result.attributes) || a_parser.parseColonType(type))
    {
        return mlir::failure();
    }

    // If the type is a function type, it contains the input and result types of this operation
    if (mlir::FunctionType funcType = llvm::dyn_cast<mlir::FunctionType>(type))
    {
        if (a_parser.resolveOperands(operands, funcType.getInputs(), operandsLoc, a_result.operands))
        {
            return mlir::failure();
        }
        a_result.addTypes(funcType.getResults());
        return mlir::success();
    }

    if (a_parser.resolveOperands(operands, type, a_result.operands))
    {
        return mlir::failure();
    }
    a_result.addTypes(type);
    return mlir::success();
}

static void printBinaryOp(mlir::OpAsmPrinter& a_printer, mlir::Operation* a_op)
{
    a_printer << " " << a_op->getOperands();
    a_printer.printOptionalAttrDict(a_op->getAttrs());
    a_printer << " : ";

    mlir::Type resultType = *a_op->result_type_begin();
    if (llvm::all_of(a_op->getOperandTypes(), [=](mlir::Type type) { return type == resultType; }))
    {
        a_printer << resultType;
        return;
    }

    a_printer.printFunctionalType(a_op->getOperandTypes(), a_op->getResultTypes());
}

void mlir::sandbox::FuncOp::build(mlir::OpBuilder& a_builder,
    mlir::OperationState& a_state,
    llvm::StringRef a_name,
    mlir::FunctionType a_type,
    llvm::ArrayRef<mlir::NamedAttribute> a_attrs)
{
    // FunctionOpInterface provides a convenient 'build' method that will populate
    // the state of our FuncOp, and create an entry block.
    buildWithEntryBlock(a_builder, a_state, a_name, a_type, a_attrs, a_type.getInputs());
}

void mlir::sandbox::AddOp::build(mlir::OpBuilder& a_builder,
    mlir::OperationState& a_state,
    mlir::Value a_lhs,
    mlir::Value a_rhs)
{
    // NOTE: the toy example also adds a type here
    a_state.addTypes(a_builder.getIntegerType(32, false));
    a_state.addOperands({a_lhs, a_rhs});
}

mlir::ParseResult mlir::sandbox::AddOp::parse(mlir::OpAsmParser& a_parser,
    mlir::OperationState& a_result)
{
    return parseBinaryOp(a_parser, a_result);
}

void mlir::sandbox::AddOp::print(mlir::OpAsmPrinter& a_printer)
{
    printBinaryOp(a_printer, *this);
}

void mlir::sandbox::MulOp::build(mlir::OpBuilder& a_builder,
    mlir::OperationState& a_state,
    mlir::Value a_lhs,
    mlir::Value a_rhs)
{
    // NOTE: the toy example also adds a type here
    a_state.addTypes(a_builder.getIntegerType(32, false));
    a_state.addOperands({a_lhs, a_rhs});
}

mlir::ParseResult mlir::sandbox::MulOp::parse(mlir::OpAsmParser& a_parser,
    mlir::OperationState& a_result)
{
    return parseBinaryOp(a_parser, a_result);
}

void mlir::sandbox::MulOp::print(mlir::OpAsmPrinter& a_printer)
{
    printBinaryOp(a_printer, *this);
}

mlir::LogicalResult mlir::sandbox::ConstantOp::verify()
{
    if (FloatAttr floatAttr = llvm::dyn_cast<FloatAttr>(getValue()))
    {
        return success();
    }
    if (IntegerAttr intAttr = llvm::dyn_cast<IntegerAttr>(getValue()))
    {
        return success();
    }
    return emitOpError() << "ConstantOp::verify failed";
}

mlir::LogicalResult mlir::sandbox::ReturnOp::verify()
{
    // We know that the parent operation is a function, because of the 'HasParent'
    // trait attached to the operation definition
    auto function = cast<FuncOp>((*this)->getParentOp());

    if (getNumOperands() > 1)
    {
        return emitOpError() << "expects at most 1 return operand";
    }

    // The operand number and types must match the function signature.
    const auto& results = function.getFunctionType().getResults();
    if (getNumOperands() != results.size())
    {
        return emitOpError() << "does not return the same number of values ("
                             << getNumOperands() << ") as the enclosing function ("
                             << results.size() << ")";
    }

    if (!hasOperand())
    {
        return mlir::success();
    }

    auto inputType = *operand_type_begin();
    auto resultType = results.front();

    if (inputType == resultType)
    {
        return mlir::success();
    }

    return emitOpError() << "type of return operand (" << inputType
                         << ") does not match function result type (" << resultType
                         << ")";
}

#define GET_OP_CLASSES
#include "Ops.cc.inc"
