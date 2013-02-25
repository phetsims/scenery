// Copyright 2002-2012, University of Colorado

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.LayerType = function( Constructor, name, backend, args ) {
    this.Constructor = Constructor;
    this.name = name;
    this.backend = backend;
    this.args = args;
  };
  var LayerType = scenery.LayerType;
  
  LayerType.prototype = {
    constructor: LayerType,
    
    supportsBackend: function( backend ) {
      return this.backend === backend;
    },
    
    supportsNode: function( node ) {
      var that = this;
      return _.some( node._supportedBackends, function( backend ) {
        return that.supportsBackend( backend );
      } );
    },
    
    createLayer: function( args ) {
      var Constructor = this.Constructor;
      return new Constructor( _.extend( {}, this.args, args ) ); // allow overriding certain arguments if necessary
    }
  };
  
  return LayerType;
} );


