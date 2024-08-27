#include "AST.h"

namespace sandbox::ast {

class Indenter
{
public:
    Indenter(uint32_t& a_level)
    : m_level(a_level)
    {
        ++m_level;
    }

    ~Indenter()
    {
        --m_level;
    }

private:
    uint32_t& m_level;
};

#define RD_INDENT() \
    Indenter indenter(m_indentLevel); \
    indent();

class Dumper
{
public:
    Dumper(std::ostream& a_os)
    : m_os(a_os)
    {}

    void dump(Module& a_module);

private:
    void indent()
    {
        for (uint32_t i = 0; i < m_indentLevel; ++i)
        {
            m_os << "    ";
        }
    }

    void dump(Expr* a_expr)
    {
        switch (a_expr->kind())
        {
            case ExprKind::BinaryOp:
                return dump(static_cast<BinaryOp_Expr*>(a_expr));
            case ExprKind::FunctionCall:
                return dump(static_cast<FunctionCall_Expr*>(a_expr));
            case ExprKind::Return:
                return dump(static_cast<Return_Expr*>(a_expr));
            case ExprKind::ScalarLiteral:
                return dump(static_cast<ScalarLiteral_Expr*>(a_expr));
            case ExprKind::VariableDeclaration:
                return dump(static_cast<VariableDeclaration_Expr*>(a_expr));
            case ExprKind::VariableReference:
                return dump(static_cast<VariableReference_Expr*>(a_expr));
        }
        RD_INDENT();
        m_os << "Unknown expression (kind: " << ((uint32_t) a_expr->kind()) << ")" << std::endl;
    }

    void dump(BinaryOp_Expr* a_expr)
    {
        RD_INDENT();
        m_os << "BinaryOp: " << a_expr->getOperator() << std::endl;
        dump(a_expr->lhs());
        dump(a_expr->rhs());
    }

    void dump(FunctionCall_Expr* a_expr)
    {
        RD_INDENT();
        m_os << "FunctionCall: '" << a_expr->callee() << "' [" << std::endl;
        for (auto& e: a_expr->arguments())
        {
            dump(e.get());
        }
        indent();
        m_os << "]" << std::endl;
    }

    void dump(Return_Expr* a_expr)
    {
        RD_INDENT();
        m_os << "Return" << std::endl;
        if (a_expr->value().has_value())
        {
            dump(a_expr->value().value());
        }
        else
        {
            RD_INDENT();
            m_os << "(void)" << std::endl;
        }
    }

    void dump(ScalarLiteral_Expr* a_expr)
    {
        RD_INDENT();
        m_os << "ScalarLiteral type: ";
        sandbox::ast::dump(m_os, a_expr->typeInfo().scalarType());
        m_os << " value: ";
        std::visit([&](auto&& v) -> void { m_os << v; }, a_expr->value());
        m_os << std::endl;
    }

    void dump(VariableDeclaration_Expr* a_expr)
    {
        RD_INDENT();
        m_os << "VariableDeclaration '" << a_expr->name() << "'" << std::endl;
        dump(a_expr->value());
    }

    void dump(VariableReference_Expr* a_expr)
    {
        RD_INDENT();
        m_os << "VariableReference '" << a_expr->name() << "'" << std::endl;
    }

    void dump(ExprList* a_body)
    {
        RD_INDENT();
        m_os << "Block {" << std::endl;
        for (auto& e: *a_body)
        {
            dump(e.get());
        }
        indent();
        m_os << "} // Block" << std::endl;
    }

    void dump(FunctionPrototype* a_proto)
    {
        RD_INDENT();
        m_os << "Proto '" << a_proto->name() << "'" << std::endl;
        indent();
        m_os << "Params: [";
        bool addComma = false;
        for (auto& p: a_proto->arguments())
        {
            if (addComma)
            {
                m_os << ", ";
            }
            addComma = true;
            m_os << p->name();
        }
        m_os << "]" << std::endl;
    }

    void dump(Function* a_function)
    {
        RD_INDENT();
        m_os << "Function" << std::endl;
        dump(a_function->prototype());
        dump(a_function->body());
    }

private:
    std::ostream& m_os;
    uint32_t      m_indentLevel {};
};

void Dumper::dump(Module& a_module)
{
    RD_INDENT();
    m_os << "Module:" << std::endl;
    for (auto& f: a_module)
    {
        dump(f.get());
    }
}

} // namespace sandbox::ast

namespace sandbox {

void dump(std::ostream& a_os, ast::Module& a_module)
{
    ast::Dumper dumper(a_os);
    dumper.dump(a_module);
}

} // namespace sandbox
