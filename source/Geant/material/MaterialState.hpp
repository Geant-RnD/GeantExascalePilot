
#pragma once

namespace geantphysics {
/**
 * @brief   Strongly typed enum class to represent material states.
 * @enum    geantx::MaterialState
 * @author  M Novak, A Ribon
 * @date    december 2015
 *
 */
enum class MaterialState {
  kStateUndefined, /**< material state is not defined */
  kStateSolid,     /**< material is solid  */
  kStateLiquid,    /**< material is liquid */
  kStateGas        /**< material is gaseous */
};

} // namespace geantphysics
