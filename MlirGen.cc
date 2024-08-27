#include "MlirGen.h"

#include "AST.h"
#include "Dialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

using llvm::ScopedHashTableScope;
using llvm::StringRef;

namespace  {

class MlirGenImpl
{
public:
    MlirGenImpl(mlir::MLIRContext& a_context)
        : m_builder(&a_context)
    {}

    mlir::ModuleOp mlirGen(sandbox::ast::Module& a_moduleAst)
    {
        m_module = mlir::ModuleOp::create(m_builder.getUnknownLoc());

        for (auto& f: a_moduleAst)
        {
            mlirGen(*f);
        }

        if (failed(mlir::verify(m_module)))
        {
            m_module.emitError("module verification error");
            return nullptr;
        }

        return m_module;
    }

private:
    mlir::Location loc(const sandbox::Location& a_loc)
    {
        return mlir::FileLineColLoc::get(m_builder.getStringAttr(a_loc.filename), a_loc.line, a_loc.column);
    }

    mlir::LogicalResult declare(llvm::StringRef a_var, mlir::Value a_value)
    {
        if (m_symbolTable.count(a_var))
        {
            return mlir::failure();
        }
        m_symbolTable.insert(a_var, a_value);
        return mlir::success();
    }

    mlir::sandbox::FuncOp mlirGen(sandbox::ast::FunctionPrototype& a_prototype)
    {
        auto location = loc(a_prototype.location());

        // This is a generic function, the return type will be inferred later.
        // FIXME:RD retrieve proper type
        //llvm::SmallVector<mlir::Type, 4> argTypes(a_prototype.arguments().size(), getType(UI64 {}));
        llvm::SmallVector<mlir::Type, 4> argTypes(a_prototype.arguments().size(), m_builder.getIntegerType(32, false));
        auto funcType = m_builder.getFunctionType(argTypes, std::nullopt);
        return m_builder.create<mlir::sandbox::FuncOp>(location, a_prototype.name(), funcType);
    }

    mlir::sandbox::FuncOp mlirGen(sandbox::ast::Function& a_functionAst)
    {
        // Create a scope in the symbol table to hold variable declarations
        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(m_symbolTable);

        // Create an MLIR function for the given prototype
        m_builder.setInsertionPointToEnd(m_module.getBody());
        mlir::sandbox::FuncOp function = mlirGen(*a_functionAst.prototype());
        if (!function)
        {
            return nullptr;
        }

        // Let's start the body of the function now
        mlir::Block& entryBlock = function.front();
        auto& protoArgs = a_functionAst.prototype()->arguments();

        // Declare all the function arguments in the symbol table
        for (const auto nameValue: llvm::zip(protoArgs, entryBlock.getArguments()))
        {
            if (failed(declare(std::get<0>(nameValue)->name(), std::get<1>(nameValue))))
            {
                return nullptr;
            }
        }

        // Set the insertion point in the builder to the beginning of the function body, it
        // will be used throughout the codegen to create operations in this function
        m_builder.setInsertionPointToEnd(&entryBlock);

        // Emit the body of the function
        if (mlir::failed(mlirGen(*a_functionAst.body())))
        {
            function.erase();
            return nullptr;
        }

        // Implicitly return void if no return statement was emitted
        mlir::sandbox::ReturnOp returnOp;
        if (!entryBlock.empty())
        {
            returnOp = llvm::dyn_cast<mlir::sandbox::ReturnOp>(entryBlock.back());
        }
        if (!returnOp)
        {
            m_builder.create<mlir::sandbox::ReturnOp>(loc(a_functionAst.prototype()->location()));
        }
        else if (returnOp.hasOperand())
        {
            // FIXME:RD type
            //function.setType(m_builder.getFunctionType(function.getFunctionType().getInputs(), getType(UI64 {})));
            function.setType(m_builder.getFunctionType(function.getFunctionType().getInputs(), m_builder.getIntegerType(32, false)));
        }

        return function;
    }

    mlir::LogicalResult mlirGen(sandbox::ast::Return_Expr& a_ret)
    {
        auto location = loc(a_ret.location());

        // 'return' takes an optional expression, handle that case here.
        mlir::Value expr = nullptr;
        if (a_ret.value().has_value())
        {
            if (!(expr = mlirGen(**a_ret.value())))
            {
                return mlir::failure();
            }
        }

        m_builder.create<mlir::sandbox::ReturnOp>(location, expr ? llvm::ArrayRef(expr) : llvm::ArrayRef<mlir::Value>());

        return mlir::success();
    }

    mlir::Value mlirGen(sandbox::ast::VariableDeclaration_Expr& a_expr)
    {
        sandbox::ast::Expr* val = a_expr.value();
        if (nullptr == val)
        {
            emitError(loc(a_expr.location()), "error: no value for '") << a_expr.name() << "'";
            return nullptr;
        }

        mlir::Value value = mlirGen(*val);
        if (!value)
        {
            return nullptr;
        }

        // NOTE: what should we do with the type at this point?
        if (failed(declare(a_expr.name(), value)))
        {
            return nullptr;
        }

        return value;
    }

    mlir::Value mlirGen(sandbox::ast::BinaryOp_Expr& a_expr)
    {
        mlir::Value lhs = mlirGen(*a_expr.lhs());
        if (!lhs)
        {
            return nullptr;
        }

        mlir::Value rhs = mlirGen(*a_expr.rhs());
        if (!rhs)
        {
            return nullptr;
        }

        auto location = loc(a_expr.location());
        switch (a_expr.getOperator())
        {
            case '+':
                return m_builder.create<mlir::sandbox::AddOp>(location, lhs, rhs);
            case '*':
                return m_builder.create<mlir::sandbox::MulOp>(location, lhs, rhs);
            default:
                ;
        }

        emitError(location, "invalid binary operator '") << a_expr.getOperator() << "'";
        return nullptr;
    }

    mlir::Value mlirGen(sandbox::ast::ScalarLiteral_Expr& a_expr)
    {
        const sandbox::ast::TypeInfo& typeInfo = a_expr.typeInfo();
        if (sandbox::ast::Type::Scalar != typeInfo.type())
        {
            emitError(loc(a_expr.location())) << "ScalarLiteral_Expr with non scalar value found";
            return nullptr;
        }
        switch (typeInfo.scalarType())
        {
            case sandbox::ast::ScalarType::UInt8:
                return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), m_builder.getIntegerType(8, false), m_builder.getIntegerAttr(m_builder.getIntegerType(8, false), std::get<uint8_t>(a_expr.value())));
                break;
            case sandbox::ast::ScalarType::UInt16:
                return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), m_builder.getIntegerType(16, false), m_builder.getIntegerAttr(m_builder.getIntegerType(16, false), std::get<uint16_t>(a_expr.value())));
                break;
            case sandbox::ast::ScalarType::UInt32:
                return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), m_builder.getIntegerType(32, false), m_builder.getIntegerAttr(m_builder.getIntegerType(32, false), std::get<uint32_t>(a_expr.value())));
                break;
            case sandbox::ast::ScalarType::UInt64:
                return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), m_builder.getIntegerType(64, false), m_builder.getIntegerAttr(m_builder.getIntegerType(64, false), std::get<uint64_t>(a_expr.value())));
                break;
            case sandbox::ast::ScalarType::Int8:
                return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), m_builder.getIntegerType(8, true), m_builder.getIntegerAttr(m_builder.getIntegerType(8, true), std::get<int8_t>(a_expr.value())));
                break;
            case sandbox::ast::ScalarType::Int16:
                return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), m_builder.getIntegerType(16, true), m_builder.getIntegerAttr(m_builder.getIntegerType(16, true), std::get<int16_t>(a_expr.value())));
                break;
            case sandbox::ast::ScalarType::Int32:
                return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), m_builder.getIntegerType(32, true), m_builder.getIntegerAttr(m_builder.getIntegerType(32, true), std::get<int32_t>(a_expr.value())));
                break;
            case sandbox::ast::ScalarType::Int64:
                return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), m_builder.getIntegerType(64, true), m_builder.getIntegerAttr(m_builder.getIntegerType(64, true), std::get<int64_t>(a_expr.value())));
                break;
            case sandbox::ast::ScalarType::Float32:
                return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), m_builder.getF32Type(), m_builder.getFloatAttr(m_builder.getF32Type(), std::get<float>(a_expr.value())));
                break;
            case sandbox::ast::ScalarType::Float64:
                return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), m_builder.getF64Type(), m_builder.getFloatAttr(m_builder.getF64Type(), std::get<double>(a_expr.value())));
                break;
            default:
                ;
        }

        emitError(loc(a_expr.location())) << "Unhandled ScalarType found in ScalarLiteral_Expr";
        return nullptr;

        // type, then attribute
        //return m_builder.create<mlir::sandbox::ConstantOp>(loc(a_expr.location()), , 
    }

    mlir::Value mlirGen(sandbox::ast::VariableReference_Expr& a_expr)
    {
        if (auto variable = m_symbolTable.lookup(a_expr.name()))
        {
            return variable;
        }

        emitError(loc(a_expr.location()), "error: unknown variable '") << a_expr.name() << "'";
        return nullptr;
    }

    mlir::Value mlirGen(sandbox::ast::Expr& a_expr)
    {
        switch (a_expr.kind())
        {
            case sandbox::ast::ExprKind::BinaryOp:
                return mlirGen(llvm::cast<sandbox::ast::BinaryOp_Expr>(a_expr));
            case sandbox::ast::ExprKind::ScalarLiteral:
                return mlirGen(llvm::cast<sandbox::ast::ScalarLiteral_Expr>(a_expr));
            case sandbox::ast::ExprKind::VariableReference:
                return mlirGen(llvm::cast<sandbox::ast::VariableReference_Expr>(a_expr));
            default:
                emitError(loc(a_expr.location())) << "MLIR codegen encountered an unhandled exp kind '" << (int) a_expr.kind() << "'";
            return nullptr;
        }
    }

    mlir::LogicalResult mlirGen(sandbox::ast::ExprList& a_blockAst)
    {
        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(m_symbolTable);
        for (auto& expr: a_blockAst)
        {
            // Specific handling for variable declarations, return statement, and print.
            // These can only appear in block list and not in nested expressions
            if (auto* ret = llvm::dyn_cast<sandbox::ast::Return_Expr>(expr.get()))
            {
                return mlirGen(*ret);
            }
            if (auto* decl = llvm::dyn_cast<sandbox::ast::VariableDeclaration_Expr>(expr.get()))
            {
                if (!mlirGen(*decl))
                {
                    return mlir::failure();
                }
                continue;
            }
            // FIXME:RD there are a few more specific cases!
            std::cerr << "FIXME:RD - to implement - type: " << (int) expr->kind() << std::endl;
        }
        return mlir::success();
    }

private:
    mlir::ModuleOp  m_module;
    mlir::OpBuilder m_builder;
    llvm::ScopedHashTable<StringRef, mlir::Value> m_symbolTable;

};  // class MlirGenImpl

} // namespace 

namespace sandbox {

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext& a_context,
    ast::Module& a_moduleAst)
{
    return MlirGenImpl(a_context).mlirGen(a_moduleAst);
}

} // namespace sandbox
