#pragma once

#include "AST.h"

namespace sandbox::ast {

Module* sampleAst_2()
{
    Module* result = new Module();

    Location loc;
    loc.filename = "SampleAst_2.h";
    loc.line     = 42;
    loc.column   = 666;
    std::vector<std::unique_ptr<VariableReference_Expr>> abcArgs {};
    abcArgs.emplace_back(new VariableReference_Expr("a", loc));
    abcArgs.emplace_back(new VariableReference_Expr("b", loc));
    abcArgs.emplace_back(new VariableReference_Expr("c", loc));
    std::unique_ptr<ExprList> abcBody = std::make_unique<ExprList>();
    std::unique_ptr<Expr> bRef(new VariableReference_Expr("b", loc));
    const TypeInfo dTypeInfo {ScalarType::UInt32};

    std::unique_ptr<Expr> dLhs(new VariableReference_Expr("a", loc));
    std::unique_ptr<Expr> dRhs(new VariableReference_Expr("b", loc));
    std::unique_ptr<Expr> dValue(new BinaryOp_Expr('*', std::move(dLhs), std::move(dRhs), loc));
    abcBody->emplace_back(new VariableDeclaration_Expr("d", dTypeInfo, std::move(dValue), loc));

    std::unique_ptr<Expr> eLhs(new ScalarLiteral_Expr(static_cast<uint32_t>(3), loc));
    std::unique_ptr<Expr> eRhs(new VariableReference_Expr("c", loc));
    std::unique_ptr<Expr> eValue(new BinaryOp_Expr('*', std::move(eLhs), std::move(eRhs), loc));
    abcBody->emplace_back(new VariableDeclaration_Expr("e", dTypeInfo, std::move(eValue), loc));

    std::unique_ptr<Expr> fLhs(new VariableReference_Expr("d", loc));
    std::unique_ptr<Expr> fRhs(new VariableReference_Expr("e", loc));
    std::unique_ptr<Expr> fValue(new BinaryOp_Expr('+', std::move(fLhs), std::move(fRhs), loc));
    abcBody->emplace_back(new VariableDeclaration_Expr("f", dTypeInfo, std::move(fValue), loc));

    std::unique_ptr<Expr> fRef(new VariableReference_Expr("f", loc));
    std::optional<std::unique_ptr<Expr>> fRefOpt = std::move(fRef);
    abcBody->emplace_back(new Return_Expr(std::move(fRefOpt), loc));

    std::unique_ptr<FunctionPrototype> abcProto(new FunctionPrototype("abc", std::move(abcArgs), loc));
    std::unique_ptr<Function> abcFun(new Function(std::move(abcProto), std::move(abcBody)));
    result->addFunction(std::move(abcFun));
    return result;
}

} // namespace sandbox::ast
