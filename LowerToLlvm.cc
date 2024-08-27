#include "Passes.h"

#include "Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>

namespace mlir::sandbox {

struct ConstantOpLowering : public mlir::OpRewritePattern<ConstantOp>
{
    using ::mlir::OpRewritePattern<ConstantOp>::OpRewritePattern;

    ::mlir::LogicalResult matchAndRewrite(ConstantOp a_op,
        ::mlir::PatternRewriter& a_rewriter) const final
    {
        if (mlir::IntegerAttr iAttr = mlir::dyn_cast<mlir::IntegerAttr>(a_op.getValue()))
        {
            const auto width = a_op.getType().getIntOrFloatBitWidth();
            const mlir::Type newType = a_rewriter.getIntegerType(width);
            mlir::Value newOp = a_rewriter.create<mlir::arith::ConstantOp>(a_op.getLoc(), newType, a_rewriter.getIntegerAttr(newType, iAttr.getUInt()));
            //a_rewriter.replaceOp(a_op, newOp);
            a_rewriter.replaceAllUsesWith(a_op, newOp);
            return success();
        }
        if (mlir::FloatAttr fAttr = mlir::dyn_cast<mlir::FloatAttr>(a_op.getValue()))
        {
            mlir::Value newOp = a_rewriter.create<mlir::arith::ConstantOp>(a_op.getLoc(), fAttr);
            //a_rewriter.replaceOp(a_op, newOp);
            a_rewriter.replaceAllUsesWith(a_op, newOp);
            return success();
        }
        return failure();
    }
};

} // namespace sandbox

namespace {

struct SandboxLoweringPass : public mlir::PassWrapper<SandboxLoweringPass, mlir::OperationPass<mlir::ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SandboxLoweringPass)

    void getDependentDialects(mlir::DialectRegistry& a_registry) const override
    {
        a_registry.insert<mlir::arith::ArithDialect>();
    }

    void runOnOperation() final;
};

}

void SandboxLoweringPass::runOnOperation()
{
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addIllegalOp<mlir::sandbox::ConstantOp>();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<mlir::sandbox::ConstantOpLowering>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::sandbox::createLoweringPass()
{
    return std::make_unique<SandboxLoweringPass>();
}
