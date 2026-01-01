// +build amd64

// func pdep(src, mask uint64) uint64
TEXT ·pdep(SB), $0-24
    MOVQ src+0(FP), AX
    MOVQ mask+8(FP), CX
    PDEPQ CX, AX, AX
    MOVQ AX, ret+16(FP)
    RET
