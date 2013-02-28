
(function(){
  module( 'Miscellaneous' );
  
  test( 'ES5 Object.defineProperty get/set', function() {
    var ob = { _key: 5 };
    Object.defineProperty( ob, 'key', {
      enumerable: true,
      configurable: true,
      get: function() { return this._key; },
      set: function( val ) { this._key = val; }
    } );
    ob.key += 1;
    equal( ob._key, 6, 'incremented object value' );
  } );
})();
