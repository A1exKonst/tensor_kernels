#pragma once

namespace matmul {
	enum class MatmulType {
		NAIVE,
		CACHE_FRIENDLY,
		TILING,
		VECTORIZED
	};
}