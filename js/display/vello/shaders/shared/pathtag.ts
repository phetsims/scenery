/* eslint-disable */

export default `struct _ac{_bs:u32,
_dq:u32,_au:u32,_dp:u32,_N:u32}const _fd=3u;const _gC=1u;const _cG=2u;const _dr=3u;const _gB=8u;const _ib=0x20u;const _gA=0x10u;const _ia=0x40u;fn _gD()->_ac{return _ac();}fn _by(a:_ac,b:_ac)->_ac{var c:_ac;c._bs=a._bs+b._bs;c._dq=a._dq+b._dq;c._au=a._au+b._au;c._dp=a._dp+b._dp;c._N=a._N+b._N;return c;}fn _ec(_D:u32)->_ac{var c:_ac;let _gz=_D&0x3030303u;c._dq=countOneBits((_gz*7u)&0x4040404u);c._bs=countOneBits(_D&(_ib*0x1010101u));let _gy=_gz+((_D>>2u)&0x1010101u);var a=_gy+(_gy&(((_D>>3u)&0x1010101u)*15u));a+=a>>8u;a+=a>>16u;c._au=a&0xffu;c._N=countOneBits(_D&(_gA*0x1010101u));c._dp=countOneBits(_D&(_ia*0x1010101u));return c;}`
