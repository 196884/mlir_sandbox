#pragma once

#include "llvm/IR/Function.h"

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

namespace sandbox {

struct Location
{
    std::string filename {};
    uint32_t    line {};
    uint32_t    column {};
};  // struct Location

} // namespace sandbox

namespace sandbox::ast {

enum class Type : uint8_t
{
    Unknown = 0,
    Scalar,
    Tensor,
};

enum class ScalarType : uint8_t
{
    Invalid = 0,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
};

inline
void dump(std::ostream& a_os, ScalarType a_type)
{
    switch (a_type)
    {
        case ScalarType::UInt8:
            a_os << "UInt8";
            return;
        case ScalarType::UInt16:
            a_os << "UInt16";
            return;
        case ScalarType::UInt32:
            a_os << "UInt32";
            return;
        case ScalarType::UInt64:
            a_os << "UInt64";
            return;
        case ScalarType::Int8:
            a_os << "Int8";
            return;
        case ScalarType::Int16:
            a_os << "Int16";
            return;
        case ScalarType::Int32:
            a_os << "Int32";
            return;
        case ScalarType::Int64:
            a_os << "Int64";
            return;
        case ScalarType::Float32:
            a_os << "Float32";
            return;
        case ScalarType::Float64:
            a_os << "Float64";
            return;
        default:
            a_os << "Invalid";
            return;
    }
}

template <typename T>
ScalarType getScalarType()
{
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        return ScalarType::UInt8;
    }
    if constexpr (std::is_same_v<T, uint16_t>)
    {
        return ScalarType::UInt16;
    }
    if constexpr (std::is_same_v<T, uint32_t>)
    {
        return ScalarType::UInt32;
    }
    if constexpr (std::is_same_v<T, uint64_t>)
    {
        return ScalarType::UInt64;
    }
    if constexpr (std::is_same_v<T, int8_t>)
    {
        return ScalarType::Int8;
    }
    if constexpr (std::is_same_v<T, int16_t>)
    {
        return ScalarType::Int16;
    }
    if constexpr (std::is_same_v<T, int32_t>)
    {
        return ScalarType::Int32;
    }
    if constexpr (std::is_same_v<T, int64_t>)
    {
        return ScalarType::Int64;
    }
    if constexpr (std::is_same_v<T, float>)
    {
        return ScalarType::Float32;
    }
    if constexpr (std::is_same_v<T, double>)
    {
        return ScalarType::Float64;
    }
    if constexpr (std::is_same_v<T, uint8_t>)
    {
        return ScalarType::UInt8;
    }
    return ScalarType::Invalid;
}

using ScalarVariant = std::variant<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double>;

using TensorShape = std::vector<uint32_t>;

class TypeInfo
{
public:
    TypeInfo() = default;

    TypeInfo(ScalarType a_scalarType)
        : m_type(Type::Scalar)
        , m_scalarType(a_scalarType)
    {}

    TypeInfo(ScalarType a_scalarType, TensorShape a_tensorShape)
        : m_type(Type::Tensor)
        , m_scalarType(a_scalarType)
        , m_tensorShape(std::move(a_tensorShape))
    {}

    Type type() const { return m_type; }
    ScalarType scalarType() const { return m_scalarType; }
    const TensorShape& tensorShape() const { return m_tensorShape; }

private:
    Type m_type {Type::Unknown};
    ScalarType m_scalarType {};
    TensorShape m_tensorShape {};
};

enum class ExprKind : uint8_t
{
    BinaryOp,
    FunctionCall,
    Return,
    ScalarLiteral,
    VariableDeclaration,
    VariableReference,
};

class Expr
{
public:
    Expr(ExprKind a_kind, Location a_location)
    : m_kind(a_kind)
    , m_location(std::move(a_location))
    {}
    virtual ~Expr() = default;

    ExprKind kind() const { return m_kind; }
    const Location& location() const { return m_location; }

protected:
    ExprKind m_kind {};
    Location m_location {};
};

using ExprList = std::vector<std::unique_ptr<Expr>>;

class VariableReference_Expr : public Expr
{
public:
    VariableReference_Expr(const std::string& a_name, Location a_location)
        : Expr(ExprKind::VariableReference, std::move(a_location))
        , m_name(a_name)
    {}

    const std::string& name() const { return m_name; }

    static bool classof(const Expr* a_e) { return a_e->kind() == ExprKind::VariableReference; }

private:
    std::string m_name {};
};

class VariableDeclaration_Expr : public Expr
{
public:
    VariableDeclaration_Expr(const std::string& a_name, TypeInfo a_typeInfo, std::unique_ptr<Expr> a_value, Location a_location)
    : Expr(ExprKind::VariableDeclaration, std::move(a_location))
    , m_name(a_name)
    , m_typeInfo(std::move(a_typeInfo))
    , m_value(std::move(a_value))
    {}

    const std::string& name() const { return m_name; }
    const TypeInfo& typeInfo() const { return m_typeInfo; }
    Expr* value() { return m_value.get(); }

    static bool classof(const Expr* a_e) { return a_e->kind() == ExprKind::VariableDeclaration; }

private:
    std::string m_name {};
    TypeInfo    m_typeInfo {};
    std::unique_ptr<Expr> m_value {};
};

class ScalarLiteral_Expr : public Expr
{
public:
    template <typename T>
    ScalarLiteral_Expr(T a_value, Location a_location)
    : Expr(ExprKind::ScalarLiteral, std::move(a_location))
    , m_typeInfo(getScalarType<T>())
    {
        m_value = a_value;
    }

    const TypeInfo& typeInfo() const { return m_typeInfo; }
    ScalarVariant& value() { return m_value; }

    static bool classof(const Expr* a_e) { return a_e->kind() == ExprKind::ScalarLiteral; }

private:
    TypeInfo      m_typeInfo {};
    ScalarVariant m_value {};
};

class BinaryOp_Expr : public Expr
{
public:
    BinaryOp_Expr(char a_operator, std::unique_ptr<Expr> a_lhs, std::unique_ptr<Expr> a_rhs, Location a_location)
    : Expr(ExprKind::BinaryOp, std::move(a_location))
    , m_operator(a_operator)
    , m_lhs(std::move(a_lhs))
    , m_rhs(std::move(a_rhs))
    {
    }

    char getOperator() const { return m_operator; }
    Expr* lhs() { return m_lhs.get(); }
    Expr* rhs() { return m_rhs.get(); }

    static bool classof(const Expr* a_e) { return a_e->kind() == ExprKind::BinaryOp; }

private:
    char m_operator {};
    std::unique_ptr<Expr> m_lhs {};
    std::unique_ptr<Expr> m_rhs {};
};

class FunctionCall_Expr : public Expr
{
public:
    FunctionCall_Expr(const std::string& a_callee, ExprList a_arguments, Location a_location)
    : Expr(ExprKind::FunctionCall, std::move(a_location))
    , m_callee(a_callee)
    , m_arguments(std::move(a_arguments))
    {}

    const std::string& callee() const { return m_callee; }
    ExprList& arguments() { return m_arguments; }

    static bool classof(const Expr* a_e) { return a_e->kind() == ExprKind::FunctionCall; }

private:
    std::string m_callee {};
    ExprList m_arguments {};
};

class Return_Expr : public Expr
{
public:
    Return_Expr(std::optional<std::unique_ptr<Expr>> a_value, Location a_location)
    : Expr(ExprKind::Return, std::move(a_location))
    , m_value(std::move(a_value))
    {}

    std::optional<Expr*> value()
    {
        if (m_value.has_value())
        {
            return m_value->get();
        }
        return std::nullopt;
    }

    static bool classof(const Expr* a_e) { return a_e->kind() == ExprKind::Return; }

private:
    std::optional<std::unique_ptr<Expr>> m_value {};
};

class FunctionPrototype
{
public:
    FunctionPrototype(const std::string& a_name, std::vector<std::unique_ptr<VariableReference_Expr>> a_arguments, Location a_location)
        : m_name(a_name)
        , m_arguments(std::move(a_arguments))
        , m_location(std::move(a_location))
    {}

    const std::string& name() const { return m_name; }
    std::vector<std::unique_ptr<VariableReference_Expr>>& arguments() { return m_arguments; }
    const Location& location() const { return m_location; }

private:
    std::string m_name {};
    std::vector<std::unique_ptr<VariableReference_Expr>> m_arguments {};
    Location m_location {};
};  // class FunctionPrototype

class Function
{
public:
    Function(std::unique_ptr<FunctionPrototype> a_prototype, std::unique_ptr<ExprList> a_body)
    : m_prototype(std::move(a_prototype))
    , m_body(std::move(a_body))
    {}

    FunctionPrototype* prototype() { return m_prototype.get(); }
    ExprList* body() { return m_body.get(); }

private:
    std::unique_ptr<FunctionPrototype> m_prototype {};
    std::unique_ptr<ExprList> m_body {};
};

class Module
{
public:
    Module() = default;

    void addFunction(std::unique_ptr<Function> a_function)
    {
        m_functions.push_back(std::move(a_function));
    }

    auto begin() { return m_functions.begin(); }
    auto end() { return m_functions.end(); }

private:
    std::vector<std::unique_ptr<Function>> m_functions {};
};  // class Module

} // namespace sandbox::ast

namespace sandbox {
void dump(std::ostream& a_stream, ast::Module& a_module);
} // namespace sandbox
