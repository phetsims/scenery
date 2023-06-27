/* eslint-disable */

export default `struct _aE{_N:u32,
_cr:u32,
_bq:u32,
_K:u32}const _du=0u;const _ff=0x44u;const _eh=0x114u;const _eg=0x29cu;const _ef=0x248u;const _dt=0x9u;const _ee=0x21u;fn _ie()->_aE{return _aE();}fn _ei(a:_aE,b:_aE)->_aE{var c:_aE;c._N=a._N+b._N;c._cr=a._cr+b._cr;c._bq=a._bq+b._bq;c._K=a._K+b._K;return c;}fn _gF(_D:u32)->_aE{var c:_aE;c._N=u32(_D !=_du);c._cr=_D&1u;c._bq=(_D>>2u)&0x07u;c._K=(_D>>6u)&0x0fu;return c;}`
