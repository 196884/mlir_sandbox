#pragma once

#include "AST.h"

namespace sandbox::ast {

Module* sampleAst_1()
{
    Module* result = new Module();

    Location loc;
    loc.filename = "SampleAst_1.h";
    loc.line     = 42;
    loc.column   = 666;
    std::vector<std::unique_ptr<VariableReference_Expr>> abcArgs {};
    abcArgs.emplace_back(new VariableReference_Expr("a", loc));
    abcArgs.emplace_back(new VariableReference_Expr("b", loc));
    abcArgs.emplace_back(new VariableReference_Expr("c", loc));
    std::unique_ptr<ExprList> abcBody = std::make_unique<ExprList>();
    std::unique_ptr<Expr> bRef(new VariableReference_Expr("b", loc));
    std::optional<std::unique_ptr<Expr>> bRefOpt = std::move(bRef);
    abcBody->emplace_back(new Return_Expr(std::move(bRefOpt), loc));
    std::unique_ptr<FunctionPrototype> abcProto(new FunctionPrototype("abc", std::move(abcArgs), loc));
    std::unique_ptr<Function> abcFun(new Function(std::move(abcProto), std::move(abcBody)));
    result->addFunction(std::move(abcFun));
    return result;
}

} // namespace sandbox::ast
