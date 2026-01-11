//go:build !amd64 || js

package main

import "math/bits"

func getWinsAndLossesAVX2(b, e uint64) (w, l uint64) {
	return getWinsAndLossesGo(b, e)
}

func selectBestEdgeAVX2(qs []float32, us []float32, coeff float32) int {
	if len(qs) == 0 {
		return -1
	}
	bestIdx := 0
	bestScore := qs[0] + coeff*us[0]
	for i := 1; i < len(qs); i++ {
		score := qs[i] + coeff*us[i]
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}
	return bestIdx
}

// SelectBit64 returns the position (0-63) of the k-th set bit in v.
// k is 0-indexed.
// Uses a hierarchical bit-counting approach (logarithmic steps) which is
// significantly faster than iterating or using software pdep on WASM.
func SelectBit64(v uint64, k int) int {
	b := 0
	// 32-bit step
	if n := bits.OnesCount64(v & 0xFFFFFFFF); k >= n {
		k -= n
		b += 32
		v >>= 32
	}
	// 16-bit step
	if n := bits.OnesCount64(v & 0xFFFF); k >= n {
		k -= n
		b += 16
		v >>= 16
	}
	// 8-bit step
	if n := bits.OnesCount64(v & 0xFF); k >= n {
		k -= n
		b += 8
		v >>= 8
	}
	// 4-bit step
	if n := bits.OnesCount64(v & 0xF); k >= n {
		k -= n
		b += 4
		v >>= 4
	}
	// 2-bit step
	if n := bits.OnesCount64(v & 0x3); k >= n {
		k -= n
		b += 2
		v >>= 2
	}
	// 1-bit step
	if n := bits.OnesCount64(v & 0x1); k >= n {
		b += 1
	}
	return b
}