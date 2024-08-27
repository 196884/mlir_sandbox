#pragma once

#include "AST.h"

namespace sandbox::ast {

Module* sampleAst_3()
{
    Module* result = new Module();

    Location loc;
    loc.filename = "SampleAst_3.h";
    loc.line     = 42;
    loc.column   = 666;
    std::vector<std::unique_ptr<VariableReference_Expr>> abcArgs {};
    abcArgs.emplace_back(new VariableReference_Expr("a", loc));
    abcArgs.emplace_back(new VariableReference_Expr("b", loc));
    abcArgs.emplace_back(new VariableReference_Expr("c", loc));
    std::unique_ptr<ExprList> abcBody = std::make_unique<ExprList>();
    const TypeInfo dTypeInfo {ScalarType::UInt32};

    std::unique_ptr<Expr> cstValue(new ScalarLiteral_Expr(static_cast<uint32_t>(3), loc));
    abcBody->emplace_back(new VariableDeclaration_Expr("d", dTypeInfo, std::move(cstValue), loc));
    std::unique_ptr<Expr> cstRef(new VariableReference_Expr("d", loc));
    std::optional<std::unique_ptr<Expr>> cstRefOpt = std::move(cstRef);
    abcBody->emplace_back(new Return_Expr(std::move(cstRefOpt), loc));

    std::unique_ptr<FunctionPrototype> abcProto(new FunctionPrototype("abc", std::move(abcArgs), loc));
    std::unique_ptr<Function> abcFun(new Function(std::move(abcProto), std::move(abcBody)));
    result->addFunction(std::move(abcFun));
    return result;
}

} // namespace sandbox::ast
