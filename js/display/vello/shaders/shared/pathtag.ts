/* eslint-disable */

export default `struct _ac{_bs:u32,
_dr:u32,_au:u32,_dq:u32,_N:u32}const _fe=3u;const _gD=1u;const _cH=2u;const _ds=3u;const _gC=8u;const _id=0x20u;const _gB=0x10u;const _ic=0x40u;fn _gE()->_ac{return _ac();}fn _by(a:_ac,b:_ac)->_ac{var c:_ac;c._bs=a._bs+b._bs;c._dr=a._dr+b._dr;c._au=a._au+b._au;c._dq=a._dq+b._dq;c._N=a._N+b._N;return c;}fn _ed(_D:u32)->_ac{var c:_ac;let _gA=_D&0x3030303u;c._dr=countOneBits((_gA*7u)&0x4040404u);c._bs=countOneBits(_D&(_id*0x1010101u));let _gz=_gA+((_D>>2u)&0x1010101u);var a=_gz+(_gz&(((_D>>3u)&0x1010101u)*15u));a+=a>>8u;a+=a>>16u;c._au=a&0xffu;c._N=countOneBits(_D&(_gB*0x1010101u));c._dq=countOneBits(_D&(_ic*0x1010101u));return c;}`
