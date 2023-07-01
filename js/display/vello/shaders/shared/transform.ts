/* eslint-disable */

export default `struct _az{_i:vec4<f32>,_bc:vec2<f32>}fn _cM(_x:_az,p:vec2<f32>)->vec2<f32>{return _x._i.xy*p.x+_x._i.zw*p.y+_x._bc;}fn _fe(_x:_az)->_az{let _ii=1./(_x._i.x*_x._i.w-_x._i.y*_x._i.z);let _fd=_ii*vec4(_x._i.w,-_x._i.y,-_x._i.z,_x._i.x);let _ih=mat2x2(_fd.xy,_fd.zw)*-_x._bc;return _az(_fd,_ih);}fn _dw(a:_az,b:_az)->_az{return _az(a._i.xyxy*b._i.xxzz+a._i.zwzw*b._i.yyww,a._i.xy*b._bc.x+a._i.zw*b._bc.y+a._bc);}`
