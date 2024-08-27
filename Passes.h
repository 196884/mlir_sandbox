#pragma once

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir::sandbox {

std::unique_ptr<mlir::Pass> createLoweringPass();

}
